import numpy as np
import logging
import xml.etree.ElementTree as ET
import os
from pyxtal.util import prettify

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
            - obj_function: Objective function to minimize.
            - bounds: Tuple (lower_bounds, upper_bounds) for each dimension.
            - num_particles: Number of particles.
            - dimensions: Number of dimensions in the search space.
            - inertia: Inertia weight for velocity update (PSO).
            - cognitive: Cognitive (personal best) weight (PSO).
            - social: Social (global best) weight (PSO).
            - max_iter: Maximum number of iterations.
            - ncpu: number of parallel processes
            - log_file: log file name
            - evalute: bool, whether evaluate the objective function or not
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
            - seed: Initial positions for particles.

        Returns:
            - None
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

        min_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[min_idx] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[min_idx]
            self.global_best_position = self.personal_best_positions[min_idx]
            self.global_best_id = min_idx


    def optimize(self, x0=None):
        """Optimize with PSO and save results to XML."""
        if x0 is not None:
            self.positions = x0

        for iteration in range(self.max_iter):
            self.pso_step()
            self.logger.info(f"Iter {iteration+1}/{self.max_iter}, Best Score: {self.global_best_score:.4f}")
        best_position = self.rescale(self.global_best_position)
        self.save()  # Save final optimized parameters
        return best_position, self.global_best_score


    def save(self, filename="pso.xml"):
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
        with open(filename, 'w') as f:
            f.write(pretty_xml)

    @classmethod
    def load(cls, filename, obj_function, obj_args):
        """
        Create a PSO instance from a saved XML file.

        Parameters:
        - cls: PSO class
        - filename: Path to the XML file containing saved PSO state
        - obj_function: Objective function to minimize
        - obj_args: Additional arguments to pass to the objective function

        Returns:
        - PSO instance with loaded state
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
