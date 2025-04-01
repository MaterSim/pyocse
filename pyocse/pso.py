import logging
import xml.etree.ElementTree as ET
import numpy as np
from pyxtal.util import prettify
import os
import multiprocessing as mp
from ase import units
from lammps import PyLammps

# Helper: Initialize global variables for each worker
def worker_init(shared_template, shared_ref_data, shared_e_offset):
    global template, ref_data, e_offset
    template = shared_template
    ref_data = shared_ref_data
    e_offset = shared_e_offset

# Worker function for a single parameter evaluation
def worker(para_values):
    """
    Evaluate the objective function for a single parameter set.
    """
    score = obj_function(para_values, template, ref_data, e_offset, obj="R2")
    return score

def obj_function_par(para_values_list, template, ref_data, e_offset, ncpu):
    if ncpu == 1:  # No parallelization
        return [obj_function(vals, template, ref_data, e_offset, obj="R2") for vals in para_values_list]
    # Initialize a multiprocessing pool
    with mp.Pool(
        processes=ncpu,
        initializer=worker_init,
        initargs=(template, ref_data, e_offset)
    ) as pool:
        # Map the computation across workers
        results = pool.map(worker, para_values_list)
    print(f"Evaluating {len(para_values_list)} parameter sets using {ncpu} CPUs.")
    return results

#@timeit
def obj_function(para_value, template, ref_data, e_offset, obj='R2'):
    """
    Objective function for the PSO Optimizer.

    Args:
        para_value: 1D-Array of parameter values to evaluate.
        template: A dictionary to guide the order of coefficients
        ref_data: (eng, force, stress, numbers, mask_e, mask_f, mask_s)

    Returns:
        Objective score.
    """
    cpu_id = (mp.current_process()._identity[0] - 1) % mp.cpu_count() if mp.current_process()._identity else 0
    if cpu_id < 10:
        folder = f"par/cpu00{cpu_id}"
    elif cpu_id < 100:
        folder = f"par/cpu0{cpu_id}"
    else:
        folder = f"par/cpu{cpu_id}"
    os.makedirs(folder, exist_ok=True)
    lmp_in_file = os.path.join(folder, "lmp.in")

    strs = f"""
clear

units real
atom_style full

dimension 3
boundary p p p  # p p m for periodic and tilt box
atom_modify sort 0 0.0

bond_style hybrid harmonic
angle_style hybrid harmonic
dihedral_style hybrid charmm
special_bonds amber lj 0.0 0.0 0.5 coul 0.0 0.0 0.83333 angle yes dihedral no

neighbor 2.0 multi
neigh_modify every 2 delay 4 check yes

# Pair Style
pair_style lj/cut/coul/long 9.0 9.0
pair_modify mix arithmetic shift no tail yes

box tilt large

# Read unique Atoms and Box for the current structure
variable atomfile string structures/lmp_dat_${{index}}
read_data ${{atomfile}} #add append
"""
    # Dynamically add mass entries
    for key in sorted(template):
        if key.startswith("mass"):
            atom_type = key.split()[1]
            mass_val = template[key]
            strs += f"mass {atom_type} {mass_val:.8f}\n"

    for key in template.keys():
        items = template[key]
        if key.startswith('bond_coeff'):
            [id1, id2] = items
            strs += f'{key} harmonic {para_value[id1]} {para_value[id2]}\n'
        elif key.startswith('angle_coeff'):
            [id1, id2] = items
            strs += f'{key} harmonic {para_value[id1]} {para_value[id2]}\n'
        elif key.startswith('dihedral_coeff'):
            [id1, arg0, arg1, arg2] = items
            strs += f'{key} charmm {para_value[id1]} {arg0} {arg1} {arg2}\n'
        elif key.startswith('pair_coeff'):
            [id1, id2] = items
            fact = 2**(5/6)
            strs += f'{key} {para_value[id1]} {para_value[id2]*fact}\n'

    strs += f"""
# dynamically specify the mesh grid based on the box
# Read box dimensions
variable xdim equal xhi-xlo
variable ydim equal yhi-ylo
variable zdim equal zhi-zlo
# Set minimum FFT grid size
variable xfft equal ceil(${{xdim}})
variable yfft equal ceil(${{ydim}})
variable zfft equal ceil(${{zdim}})
# Apply a minimum grid size of 32
if "${{xfft}} < 32" then "variable xfft equal 32"
if "${{yfft}} < 32" then "variable yfft equal 32"
if "${{zfft}} < 32" then "variable zfft equal 32"
kspace_style pppm 0.0005
kspace_modify gewald 0.29202898720871845 mesh ${{xfft}} ${{yfft}} ${{zfft}} order 6

# Thermodynamic Variables
variable pxx equal pxx
variable pyy equal pyy
variable pzz equal pzz
variable pyz equal pyz
variable pxz equal pxz
variable pxy equal pxy
variable fx atom fx
variable fy atom fy
variable fz atom fz
"""

    with open(f"{lmp_in_file}", 'w') as f: f.write(strs)
    engs1, forces1, stress1 = execute_lammps(f"{lmp_in_file}", len(ref_data[0]))

    # Reference data
    (engs, forces, stress, numbers, mask_e, mask_f, mask_s) = ref_data
    engs0 = engs / numbers
    engs0 = engs0[mask_e]
    forces0 = forces[mask_f]
    stress0 = stress[mask_s]

    # FF data
    engs1 /= numbers
    engs1 = engs1[mask_e]
    engs1 += e_offset
    forces1 = forces1[mask_f]
    stress1 = stress1[mask_s]
    
    # Calculate objective score
    if obj == "MSE":
        score = np.sum((engs1 - engs0)**2) + np.sum((forces1 - forces0)**2) + np.sum((stress1 - stress0)**2)
    elif obj == "R2":
        score = -r2_score(engs1, engs0)
        score -= r2_score(forces1, forces0)
        score -= r2_score(stress1, stress0)
    return score

def execute_lammps(lmp_in, N_strucs):
    """
    Args:
        lmp_in (str): path of lmp_in
        N_strucs (int): Number of structures
    """
    cmdargs = ["-screen", "none", "-log", "none", "-nocite"]
    lmp = PyLammps(cmdargs=cmdargs)
    engs = []
    for id in range(N_strucs):
        lmp.command(f"variable index equal {id+1}")
        lmp.file(lmp_in)
        lmp.run(0)
        thermo = lmp.last_run[-1]
        energy = float(thermo.TotEng[-1]) * units.kcal / units.mol
        stress_vars = ['pxx', 'pyy', 'pzz', 'pyz', 'pxz', 'pxy']
        stress = np.array([lmp.variables[var].value for var in stress_vars])
        fx = np.frombuffer(lmp.variables['fx'].value)
        fy = np.frombuffer(lmp.variables['fy'].value)
        fz = np.frombuffer(lmp.variables['fz'].value)
        force = np.vstack((fx, fy, fz)).T.flatten() * units.kcal / units.mol
        stress = -stress * 101325 * units.Pascal
        engs.append(energy)
        if id > 0:
            stresses = np.append(stresses, stress)
            forces = np.append(forces, force)
        else:
            stresses = stress
            forces = force
    lmp.close()
    return np.array(engs), forces, stresses

def r2_score(y_true, y_pred):
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    return 1 - (rss / tss)

class PSO:
    def __init__(self,
                 obj_function,
                 obj_args,
                 bounds,
                 seed=None,
                 num_particles=30,
                 dimensions=2,
                 inertia=0.5,
                 cognitive=0.2,
                 social=0.8,
                 max_iter=100,
                 ncpu=1,
                 log_file="pso.log",
                 xml_file="pso.xml",
                 evaluate=True):
        """
        Initialize the PSO optimizer.

        Args:
            obj_function: Objective function to minimize.
            bounds: Tuple (lower_bounds, upper_bounds) for each dimension.
            num_particles: Number of particles.
            dimensions: Number of dimensions in the search space.
            inertia: Inertia weight for velocity update (PSO).
            cognitive: Cognitive (personal best) weight (PSO).
            social: Social (global best) weight (PSO).
            max_iter: Maximum number of iterations.
            ncpu: number of parallel processes
            log_file: log file name
            evalute: bool, whether evaluate the objective function or not
        """

        self.obj_function = obj_function
        self.obj_args = obj_args
        self.bounds = bounds
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iter = max_iter
        self.ncpu = ncpu
        self.log_file = log_file
        self.xml_file = xml_file
        self.init_logger()

        # Initialize bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])

        if evaluate: self.init_state(seed=seed)

    def init_state(self, seed=None):
        """
        Initialize the state of the PSO optimizer.

        Args:
            seed: Initial positions for particles.

        Returns:
            None
        """
        # Initialize particles
        self.positions = np.random.uniform(self.lower_bounds,
                                           self.upper_bounds,
                                           (self.num_particles, self.dimensions))
        self.positions -= self.lower_bounds
        self.positions /= (self.upper_bounds - self.lower_bounds)
        self.velocities = 0.1 * np.random.uniform(-1, 1, (self.num_particles, self.dimensions))
        if seed is not None: self.set_seeds(seed)

        # Init: evaluate the score for each particle
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.zeros(len(self.positions))
        scores = self.safe_evaluate_par()
        for i in range(self.num_particles):
            self.personal_best_scores[i] = scores[i]
            self.logger.info(f"{i} score: {scores[i]}")
            print(f"{i} score: {scores[i]}")

        min_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[min_idx]
        self.global_best_score = np.min(self.personal_best_scores)
        self.global_best_id = min_idx

    def init_logger(self):
        # Initialize logger
        logging.getLogger().handlers.clear()
        logging.basicConfig(format="%(asctime)s| %(message)s",
                            filename=self.log_file,
                            level=logging.INFO)
        self.logger = logging

    def __str__(self):
        strs = f"\n=====PSO Optimizer=====\n"
        strs += f"Num Particles: {self.num_particles}\n"
        strs += f"Dimensions:    {self.dimensions}\n"
        strs += f"Inertia:       {self.inertia}\n"
        strs += f"Cognitive:     {self.cognitive}\n"
        strs += f"Social:        {self.social}\n"
        strs += f"Max Iter:      {self.max_iter}\n"
        strs += f"NCPU:          {self.ncpu}\n"
        strs += f"Log File:      {self.log_file}\n"
        strs += f"XML File:      {self.xml_file}\n"
        return strs

    def set_seeds(self, seed):
        """Set initial positions using the provided seed."""
        n_seeds = len(seed)
        self.logger.info(f"Set Seeds {n_seeds}")
        self.positions[:n_seeds] = (seed - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def rescale(self, scaled_values):
        """
        Rescale values from (0, 1) back to original bounds.
        """
        return scaled_values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

    def safe_evaluate_par(self):
        p_actuals = []
        for p in self.positions:
            p_actual = self.rescale(p)
            p_actuals.append(p_actual)

        scores = self.obj_function(p_actuals, *self.obj_args)
        return scores

    def pso_step(self):
        """Perform one step of PSO."""
        for i in range(self.num_particles):
            # update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            self.positions[i] += self.velocities[i]
            v = (
                 self.inertia * self.velocities[i] +
                 self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                 self.social * r2 * (self.global_best_position - self.positions[i]) +
                 0.05 * np.random.uniform(-1, 1, (self.dimensions))
            )
            v /= np.abs(v).max()
            self.velocities[i] = 0.1 * v #(-0.1, 0.1)
            self.positions[i] = np.clip(self.positions[i], 0, 1)

        # Evaluate results in parallel
        scores = self.safe_evaluate_par()

        for i in range(self.num_particles):
            score = scores[i]
            if score < self.personal_best_scores[i]:
                strs = f"{i} score: {score:.4f}  pbest: {self.personal_best_scores[i]:.4f}"
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
                self.logger.info(strs)
                print(strs)

        min_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[min_idx] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[min_idx]
            self.global_best_position = self.personal_best_positions[min_idx]
            self.global_best_id = min_idx

    def optimize(self, x0=None):
        """Optimize with PSO and save results to XML."""
        if x0 is not None: self.positions = x0

        for iteration in range(self.max_iter):
            self.pso_step()
            self.logger.info(f"Iter {iteration+1}/{self.max_iter}, Best Score: {self.global_best_score:.4f}")
            print(f"Iter {iteration+1}/{self.max_iter}, Best Score: {self.global_best_score:.4f}")
        best_position = self.rescale(self.global_best_position)
        self.save()  # Save final optimized parameters
        return best_position, self.global_best_score


    def save(self):
        """
        Save current best parameters to an XML file in array format.
        """
        # Create root element
        root = ET.Element("pso")

        # Save parameters
        params = ET.SubElement(root, "parameters")
        params.set("num_particles", str(self.num_particles))
        params.set("dimensions", str(self.dimensions))
        params.set("inertia", str(self.inertia))
        params.set("cognitive", str(self.cognitive))
        params.set("social", str(self.social))
        params.set("max_iter", str(self.max_iter))
        params.set("ncpu", str(self.ncpu))
        params.set("log_file", str(self.log_file))
        params.set("xml_file", str(self.xml_file))

        # Save bounds
        bounds = ET.SubElement(root, "bounds")
        bounds.set("lower", str(self.lower_bounds.tolist()))
        bounds.set("upper", str(self.upper_bounds.tolist()))

        # Save global best values
        bounds = ET.SubElement(root, "bests")
        bounds.set("global", str(self.global_best_score))
        bounds.set("global_id", str(self.global_best_id))

        # Global Best
        gb_elem = ET.SubElement(root, "global_best")
        gb_elem.text = str(self.global_best_position.tolist())

        # Local Bests
        lb_elem = ET.SubElement(root, "local_bests")
        for i in range(self.num_particles):
            p_elem = ET.SubElement(lb_elem, f"particle_{i}")
            p_elem.text = str(self.personal_best_positions[i].tolist())

        # local best scores
        lbs_elem = ET.SubElement(root, "local_best_scores")
        for i in range(self.num_particles):
            p_elem = ET.SubElement(lbs_elem, f"particle_{i}")
            p_elem.text = str(self.personal_best_scores[i])

        # Positions
        pos_elem = ET.SubElement(root, "positions")
        for i in range(self.num_particles):
            p_elem = ET.SubElement(pos_elem, f"particle_{i}")
            p_elem.text = str(self.positions[i].tolist())

        # Velocities
        vel_elem = ET.SubElement(root, "velocities")
        for i in range(self.num_particles):
            v_elem = ET.SubElement(vel_elem, f"particle_{i}")
            v_elem.text = str(self.velocities[i].tolist())

        # Use prettify to get a pretty-printed XML string
        pretty_xml = prettify(root)

        # Write the pretty-printed XML to a file
        with open(self.xml_file, 'w') as f:
            f.write(pretty_xml)

    @classmethod
    def load(cls, filename, obj_function, obj_args):
        """
        Create a PSO instance from a saved XML file.

        Args:
            cls: PSO class
            filename: Path to the XML file containing saved PSO state
            obj_function: Objective function to minimize
            obj_args: Additional arguments to pass to the objective function

        Returns:
            PSO instance with loaded state
        """

        tree = ET.parse(filename)
        root = tree.getroot()

        # Load parameters
        num_particles = int(root.find("parameters").get("num_particles"))
        dimensions = int(root.find("parameters").get("dimensions"))
        inertia = float(root.find("parameters").get("inertia"))
        cognitive = float(root.find("parameters").get("cognitive"))
        social = float(root.find("parameters").get("social"))
        max_iter = int(root.find("parameters").get("max_iter"))
        ncpu = int(root.find("parameters").get("ncpu"))
        log_file = root.find("parameters").get("log_file")
        xml_file = root.find("parameters").get("xml_file")

        # Load bounds
        lower_bounds = np.array(
            np.fromstring(root.find("bounds").get("lower")[1:-1], sep=",")
        )
        upper_bounds = np.array(
            np.fromstring(root.find("bounds").get("upper")[1:-1], sep=",")
        )

        # Create PSO instance
        pso = cls(obj_function, obj_args, (lower_bounds, upper_bounds),
                  num_particles=num_particles,
                  dimensions=dimensions,
                  inertia=inertia,
                  cognitive=cognitive,
                  social=social,
                  max_iter=max_iter,
                  ncpu=ncpu,
                  log_file=log_file,
                  xml_file=xml_file,
                  evaluate=False)
        pso.lower_bounds = lower_bounds
        pso.upper_bounds = upper_bounds

        # Load global best score
        pso.global_best_score = float(root.find("bests").get("global"))
        pso.global_best_id = int(root.find("bests").get("global_id"))

        # Load global best position (convert string list to numpy array)
        pso.global_best_position = np.array(
            np.fromstring(root.find("global_best").text.strip()[1:-1], sep=",")
        )

        # Load personal best positions
        pso.personal_best_positions = np.zeros((pso.num_particles, pso.dimensions))
        for i, p_elem in enumerate(root.find("local_bests")):
            pso.personal_best_positions[i] = np.fromstring(p_elem.text.strip()[1:-1], sep=",")

        # Load personal best scores
        pso.personal_best_scores = np.zeros(pso.num_particles)
        for i, p_elem in enumerate(root.find("local_best_scores")):
            pso.personal_best_scores[i] = float(p_elem.text)

        # Load positions
        pso.positions = np.zeros((pso.num_particles, pso.dimensions))
        for i, p_elem in enumerate(root.find("positions")):
            pso.positions[i] = np.fromstring(p_elem.text.strip()[1:-1], sep=",")

        # Load velocities
        pso.velocities = np.zeros((pso.num_particles, pso.dimensions))
        for i, v_elem in enumerate(root.find("velocities")):
            pso.velocities[i] = np.fromstring(v_elem.text.strip()[1:-1], sep=",")

        pso.logger.info(f"Loaded PSO from {filename}")

        return pso
