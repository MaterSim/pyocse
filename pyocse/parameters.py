import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import time
import numpy as np
from scipy.optimize import minimize

from ase import Atoms
from pyxtal import pyxtal
from pyxtal.util import prettify
from pyocse.utils import reset_lammps_cell, compute_r2, xml_to_dict_list, array_to_string
from pyocse.lmp import LAMMPSCalculator

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        t = end_time - start_time
        print(f"{method.__name__} took {t} seconds to execute.")
        return result
    return timed

def prepare_atoms(ref_dic):
    """
    A short routine to prepare the ASE Atoms object from the reference dictionary
    """
    structure = Atoms(numbers=ref_dic['numbers'],
                      positions=ref_dic['position'],
                      cell=ref_dic['lattice'],
                      pbc=[1, 1, 1])
    structure = reset_lammps_cell(structure)
    return structure #.cell.cellpar(), structure.get_positions()

def evaluate_ff_error_par(ref_dics, lmp_strucs, lmp_dats, lmp_in, e_offset,
        dir_name, max_E=1000.0, max_dE=1.25):
    """
    parallel version
    """
    pwd = os.getcwd()
    os.chdir(dir_name)

    ff_eng, ff_force, ff_stress = [], [], []
    ref_eng, ref_force, ref_stress = [], [], []

    for ref_dic, lmp_struc, lmp_dat in zip(ref_dics, lmp_strucs, lmp_dats):
        options = ref_dic['options']
        replicate = ref_dic['replicate']
        structure = prepare_atoms(ref_dic)
        lmp_struc.box = structure.cell.cellpar()
        lmp_struc.coordinates = structure.get_positions()

        if not hasattr(lmp_struc, 'ewald_error_tolerance'): lmp_struc.complete()
        calc = LAMMPSCalculator(lmp_struc, lmp_in=lmp_in, lmp_dat=lmp_dat)
        eng, force, stress = calc.express_evaluation()

        # Ignore the structures with unphysical energy values
        e_diff = eng/replicate + e_offset - ref_dic['energy']/replicate
        if eng < max_E and abs(e_diff) < max_dE:
            ff_eng.append(eng/replicate + e_offset)
            ref_eng.append(ref_dic['energy']/replicate)

            if options[1]:
                ff_force.extend(force.tolist())
                ref_force.extend(ref_dic['forces'].tolist())

            if options[2]:
                ff_stress.extend(stress.tolist())
                ref_stress.extend(ref_dic['stress'].tolist())
        else:
            print('Neglect reference due to energy', eng, abs(e_diff), ref_dic['tag'])
    os.chdir(pwd)
    return (ff_eng, ff_force, ff_stress, ref_eng, ref_force, ref_stress)

def compute_refs_par(*args_list):
    """
    Evaluate the reference structure with the ref_evaluator
    args_list: (structure, numMol, calculator, natoms_per_unit,
                options, tag, steps, fmax, logfile)
    """
    ref_dics = []
    strucs, numMols, calculator, natoms_per_unit, options, tags, steps = args_list
    for (structure, numMol, option, tag, step) in zip(strucs, numMols, options, tags, steps):
        ref_dic = compute_ref_single(structure,
                                   numMol,
                                   calculator,
                                   natoms_per_unit,
                                   options=option,
                                   tag=tag,
                                   steps=step)
        ref_dics.append(ref_dic)
    return ref_dics

def compute_ref_single(structure, numMol, calculator, natoms_per_unit,
                        options=[True, True, True], tag='minimum',
                        steps=0, fmax=0.1, logfile='ase.log'):
    """
    Evaluate the reference structure with the ref_evaluator

    Args:
        structure (ASE Atoms): ASE Atoms object
        numMol (int): number of molecules
        calculator (object): ASE calculator
        natoms_per_unit (int): number of atoms per unit cell
        options (list): [energy, force, stress]
        steps (int): number of steps for relaxation
        fmax (float): maximum force for relaxation
        logfile (str): ASE logfile
    """
    from ase.optimize.fire import FIRE
    from ase.constraints import UnitCellFilter, FixSymmetry

    structure = reset_lammps_cell(structure)
    ref_dic = {"numbers": structure.numbers,
               "lattice": structure.cell.array,
               "position": structure.positions,
               'energy': None,
               'forces': None,
               'stress': None,
               'replicate': len(structure) / natoms_per_unit,
               'options': options,
               'tag': tag,
               'numMols': numMol}
    structure.set_calculator(calculator)
    if steps > 0:
        if tag == 'minimum':
            structure.set_constraint(FixSymmetry(structure))
            ecf = UnitCellFilter(structure)
            dyn = FIRE(ecf, a=0.1, logfile=logfile)
            dyn.run(fmax=fmax, steps=steps)
            structure.set_constraint()
        else:
            dyn = FIRE(structure, a=0.1, logfile=logfile)
            dyn.run(fmax=fmax, steps=steps)

    if options[0]: # Energy
        ref_dic['energy'] = structure.get_potential_energy()
    if options[1]: # forces
        ref_dic['forces'] = structure.get_forces()
    if options[2]:
        ref_dic['stress'] = structure.get_stress()
    structure.set_calculator() # reset calculator to None
    print("#", tag, np.diag(structure.cell.array))

    return ref_dic

def add_strucs_par(strs, smiles):
    from pymatgen.core import Structure
    xtals = []
    for _str in strs:
        try:
            pmg = Structure.from_str(_str, fmt='cif')
            c0 = pyxtal(molecular=True)
            c0.from_seed(pmg, molecules=smiles)
            xtals.append(c0)
        except:
            print("Skip a structure due to reading error")
            print(_str)
    return xtals

def opt_obj_fun(x, fun_args):
    """
    Objective function for optimization.
    This function must be defined at the top level to be pickleable.
    """
    para, ref_dics, parameters0, obj, ids, terms, charges = fun_args

    # Ensure ids and terms match
    if len(ids) != len(terms):
        print(f"WARNING: Mismatch detected - ids ({len(ids)}) vs. terms ({len(terms)})")
        print(f"  ids: {ids}")
        print(f"  terms: {terms}")
        raise ValueError(f"Mismatch between ids and terms: ids({len(ids)}) vs terms({len(terms)})")

    # Correctly extract values from x using ids
    values = [x[ids[i - 1]:ids[i]] if i > 0 else x[:ids[0]] for i in range(len(ids))]

    # Handle charge term separately
    if charges is not None:
        charges = np.array(charges, dtype=np.float64)
        values.append(x[-1] * charges)  # Apply charge scaling

    # Update parameters with extracted values
    parameters = para.set_sub_parameters(values, terms, parameters0)
    para.update_ff_parameters(parameters)

    # Reset the LAMMPS input file ?
    # lmp_in = para.ff.get_lammps_in()
    _, _, mse_values, r2_values = para.evaluate_multi_references(ref_dics, parameters)
    e_coef = para.e_coef
    f_coef = para.f_coef
    s_coef = para.s_coef

    if obj == 'MSE':
        objective = e_coef * mse_values[0] + f_coef * mse_values[1] + s_coef * mse_values[2]
    elif obj == 'R2':
        objective = - (e_coef * r2_values[0] + f_coef * r2_values[1] + s_coef * r2_values[2])
    else:
        raise ValueError("Invalid obj_type. Choose 'MSE' or 'R2'.")

    print("Total Objective:", objective)

    return objective

"""
A class to handle the optimization of force field parameters
for molecular simulation.
"""
class ForceFieldParametersBase:
    def __init__(self,
                 smiles, # = ['CC(=O)OC1=CC=CC=C1C(=O)O'],
                 style = 'gaff',
                 chargemethod = 'am1bcc',
                 ff_evaluator = 'lammps',
                 e_coef = 1.0,
                 f_coef = 0.1,
                 s_coef = 1.0,
                 ncpu = 1,
                 verbose = True,
                 ):
        """
        Args:
            smiles (list): list of smiles strings
            style (str): 'gaff' or 'openff'
            chargemethod (str): 'mmff94', 'am1bcc', 'am1-mulliken', 'gasteiger'
            ff_evaluator (str): 'lammps' or 'charmm'
            e_coef (float): coefficients for energy
            f_coef (float): coefficients for forces
            s_coef (float): coefficients for stress
        """
        from pyocse.forcefield import forcefield

        self.smiles = smiles
        self.ff_style = style
        self.ff = forcefield(smiles, style, chargemethod)

        # only works for 1:1 ratio cocrystal for now (QZ: to check if it is true)
        self.natoms_per_unit = sum([len(mol.atoms) for mol in self.ff.molecules])
        params_init, constraints, bounds = self.get_default_ff_parameters()
        self.params_init = params_init
        self.constraints = constraints
        self.bounds = bounds

        self.ff_evaluator = ff_evaluator
        self.ase_templates = {}
        self.lmp_dat = {}
        self.e_coef = e_coef
        self.f_coef = f_coef
        self.s_coef = s_coef
        self.terms = ['bond', 'angle', 'proper', 'vdW', 'charge', 'offset']
        self.ncpu = ncpu
        self.verbose = verbose

    def __str__(self):
        s = "\n------Force Field Parameters------\n"
        s += "Bond:        {:3d}\n".format(self.N_bond)
        s += "Angle:       {:3d}\n".format(self.N_angle)
        s += "Proper:      {:3d}\n".format(self.N_proper)
        s += "Improper:    {:3d}\n".format(self.N_improper)
        s += "vdW:         {:3d}\n".format(self.N_vdW)
        s += "Charges:     {:3d}\n".format(self.N_charge)
        s += "Total:       {:3d}\n".format(len(self.params_init))
        s += "Constraints: {:3d}\n".format(len(self.constraints))
        s += "FF_code:    {:s}\n".format(self.ff_evaluator)
        if hasattr(self, 'ref_evaluator'):
            s += "Ref_code:   {:s}\n".format(self.ref_evaluator)
        s += "N_CPU:       {:3d}\n".format(self.ncpu)
        s += "E_coef:      {:.3f}\n".format(self.e_coef)
        s += "F_coef:      {:.3f}\n".format(self.f_coef)
        s += "S_coef:      {:.3f}\n".format(self.s_coef)
        return s

    def __repr__(self):
        return str(self)

    def get_default_ff_parameters(self, coefs=[0.5, 1.5], deltas=[-0.2, 0.2]):
        """
        Get the initial FF parameters/bounds/constraints
        1. Loop over molecule
        2. Loop over LJ, Bond, Angle, Torsion, Improper
        3. Loop over molecule
        4. Loop over charges
        """

        params = []
        bounds = []
        constraints = []
        N_bond, N_angle, N_proper, N_improper, N_vdW, N_charge = 0, 0, 0, 0, 0, 0

        # Bond (k, req)
        for molecule in self.ff.molecules:
            for bond_type in molecule.bond_types:
                k = bond_type.k
                req = bond_type.req
                params.append(k)
                params.append(req)
                bounds.append((k * coefs[0], k * coefs[1]))
                bounds.append((req + deltas[0], req + deltas[1]))
                N_bond += 2

        # Angle (k, theteq)
        for molecule in self.ff.molecules:
            for angle_type in molecule.angle_types:
                k = angle_type.k
                theteq = angle_type.theteq
                params.append(k)
                params.append(theteq)
                bounds.append((k * coefs[0], k * coefs[1]))
                bounds.append((theteq + deltas[0], theteq + deltas[1]))
                N_angle += 2

        # Proper (phi_k) # per=2, phase=180.000,  scee=1.200, scnb=2.000
        # phi_k can be negative in some cases
        for molecule in self.ff.molecules:
            for dihedral_type in molecule.dihedral_types:
                    phi_k = dihedral_type.phi_k
                    params.append(phi_k)
                    if phi_k > 0:
                        bounds.append((phi_k * coefs[0], phi_k * coefs[1]))
                    else:
                        bounds.append((phi_k * coefs[1], phi_k * coefs[0]))
                    N_proper += 1

        # Improper (phi_k) #  per=2, phase=180.000,  scee=1.200, scnb=2.000>
        # for molecule in self.ff.molecules:
        # for improper_type in ps.improper_periodic_types.keys():

        # nonbond vdW parameters (rmin, epsilon)
        # sigma is related to rmin * 2**(-1/6) * 2
        for molecule in self.ff.molecules:
            ps = molecule.get_parameterset_with_resname_as_prefix()
            for atom_type in ps.atom_types.keys():
                rmin = ps.atom_types[atom_type].rmin
                epsilon = ps.atom_types[atom_type].epsilon
                params.append(rmin)
                params.append(epsilon)
                bounds.append((rmin + deltas[0], rmin + deltas[1]))
                bounds.append((epsilon * coefs[0], epsilon * coefs[1]))
                N_vdW += 2

        # nonbond charges
        for molecule in self.ff.molecules:
            for at in molecule.atoms:
                chg = at.charge
                params.append(chg)
                bounds.append((chg + deltas[0], chg + deltas[1]))
            id1 = len(params) - len(molecule.atoms)
            id2 = len(params)
            sum_chg = sum(params[id1:id2])
            constraints.append((id1, id2, sum_chg))
            N_charge += len(molecule.atoms)

        # N_LJ, N_bond, N_angle, N_proper, N_improper, N_charge
        self.N_bond = N_bond
        self.N_angle = N_angle
        self.N_proper = N_proper
        self.N_improper = N_improper
        self.N_vdW = N_vdW
        self.N_charge = N_charge
        # This is for the offset
        params.append(0)

        return params, constraints, bounds

    def get_sub_parameters(self, parameters, terms):
        """
        Get the subparameters/bonds/constraints for optimization

        Args:
            parameters (list): input complete parameters
            termss (list): selected terms ['vdW', 'bond', .etc]

        Returns:
            sub_paras (list): list of sub_parameters
            sub_bounds (list): list of bounds for each sub_parameter
            sub_constraints (list): list of constraints for each sub_parameter
        """
        assert(len(parameters) == len(self.params_init))
        sub_paras = []
        sub_bounds = []
        sub_constraints = []
        do_charge = False
        count = 0
        for term in terms:
            if term == 'bond':
                id1 = 0
                id2 = id1 + self.N_bond
            elif term == 'angle':
                id1 = self.N_bond
                id2 = id1 + self.N_angle
            elif term == 'proper':
                id1 = self.N_bond + self.N_angle
                id2 = id1 + self.N_proper
            elif term == 'vdW':
                id1 = self.N_bond + self.N_angle + self.N_proper
                id2 = id1 + self.N_vdW
            elif term == 'charge':
                id1 = self.N_bond + self.N_angle + self.N_proper + self.N_vdW
                id2 = id1 + self.N_charge
                do_charge = True
            elif term == 'offset':
                id1 = self.N_bond + self.N_angle + self.N_proper + self.N_vdW + self.N_charge
                id2 = id1 + 1

            sub_paras.append(parameters[id1:id2])

            if term != 'charge':
                sub_bounds.append(self.bounds[id1:id2])
            else:
                sub_bounds.append([(0.5, 2.0)])

            count += id2 - id1
            if do_charge:
                shift = self.N_bond + self.N_angle + self.N_proper + self.N_vdW
                shift -= count
                for cons in self.constraints:
                    (_id1, _id2, chgsum) = cons
                    sub_constraints.append((_id1-shift, _id2-shift, chgsum))

        return sub_paras, sub_bounds, sub_constraints

    def set_sub_parameters(self, sub_parameters, terms, parameters0=None):
        """
        Get the subparameters/bonds/constraints for optimization

        Args:
            sub_parameters (list): list of sub_parameters
            terms (list): selected terms ['vdW', 'bond', .etc]
            parameters0 (1d array): input complete parameters

        Returns:
            parameters (1d array): updated parameters
        """
        if parameters0 is None: parameters0 = self.params_init
        parameters = parameters0.copy()

        # Handle 1D array
        if len(sub_parameters) != len(terms):
            sub_values = []
            count = 0
            for term in terms:
                N = getattr(self, 'N_'+term)
                sub_values.append(sub_parameters[count:count+N])
                count += N
            sub_parameters = sub_values
        assert(len(sub_parameters) == len(terms))
        for sub_para, term in zip(sub_parameters, terms):
            if term == 'bond':
                id1 = 0
                id2 = id1 + self.N_bond
            elif term == 'angle':
                id1 = self.N_bond
                id2 = id1 + self.N_angle
            elif term == 'proper':
                id1 = self.N_bond + self.N_angle
                id2 = id1 + self.N_proper
            elif term == 'vdW':
                id1 = self.N_bond + self.N_angle + self.N_proper
                id2 = id1 + self.N_vdW
            elif term == 'charge':
                id1 = self.N_bond + self.N_angle + self.N_proper + self.N_vdW
                id2 = id1 + self.N_charge
            elif term == 'offset':
                id1 = self.N_bond + self.N_angle + self.N_proper + self.N_vdW + self.N_charge
                id2 = id1 + 1

            parameters[id1:id2] = sub_para

        return parameters

    #@timeit
    def update_ff_parameters(self, parameters):
        """
        Update FF parameters in self.ff.molecules
        1. Loop over molecule
        2. Loop over Bond, Angle, Torsion, Improper, vdW, charges
        """
        assert(len(parameters) == len(self.params_init))
        self.ff.update_parameters(parameters)

        # remember the offset value
        self.params_init[-1] = parameters[-1]

        # reset the ase_lammps to empty
        self.ase_templates = {}
        self.lmp_dat = {}


    def get_lmp_inputs_from_ref_dics(self, ref_dics):
        """
        Get lmp_strucs and lmp_dats from ref_dics
        To explain later
        """
        lmp_strucs, lmp_dats = [], []
        for ref_dic in ref_dics:
            numMols = ref_dic['numMols']
            #structure = Atoms(numbers = ref_dic['numbers'],
            #                  positions = ref_dic['position'],
            #                  cell = ref_dic['lattice'],
            #                  pbc = [1, 1, 1])
            structure = prepare_atoms(ref_dic)

            lmp_struc, lmp_dat = self.get_lmp_input_from_structure(structure, numMols)
            lmp_strucs.append(lmp_struc)
            lmp_dats.append(lmp_dat)
        #print('Final'); print(lmp_strucs[0].box); print(lmp_strucs[-1].box); import sys; sys.exit()
        return lmp_strucs, lmp_dats

    def get_lmp_input_from_structure(self, structure, numMols=[1], set_template=True):
        """
        Get lmp_struc and lmp_dat from ase structure

        Args:
            structure (ASE Atoms): ASE Atoms object
            numMols (list): list of number of molecules
            set_template (bool): whether to set the template
        """

        replicate = len(structure) / self.natoms_per_unit
        if replicate in self.ase_templates.keys():
            lmp_struc = self.ase_templates[replicate]
            lmp_dat = self.lmp_dat[replicate]
        else:
            lmp_struc = self.ff.get_ase_lammps(structure, numMols)
            dat_head = lmp_struc._write_dat_head()
            dat_prm = lmp_struc._write_dat_parameters()
            dat_connect, _, _, _ = lmp_struc._write_dat_connects()
            lmp_dat = [dat_head, dat_prm, dat_connect]
            if set_template:
                self.lmp_dat[replicate] = lmp_dat
                self.ase_templates[replicate] = lmp_struc
        return lmp_struc, lmp_dat

    def write_lmp_dat_from_ref_dics(self, ref_dics, DIR='structures'):
        """
        Write lmp.dat for all structures from the ref_dics

        Args:
            ref_dics: list of dictionaries
            DIR: directory to write the lmp.dat files
        """
        os.makedirs(DIR, exist_ok=True)
        for i, ref_dic in enumerate(ref_dics):
            numMols = ref_dic['numMols']
            #structure = Atoms(numbers = ref_dic['numbers'],
            #                  positions = ref_dic['position'],
            #                  cell = ref_dic['lattice'],
            #                  pbc = [1, 1, 1])
            structure = prepare_atoms(ref_dic)
            lmp_struc = self.ff.get_ase_lammps(structure, numMols)
            lmp_struc.box = structure.cell.cellpar()
            #print(i, lmp_struc.box[:3], lmp_struc.fftgrid())
            lmp_struc.coordinates = structure.positions
            dat_head = lmp_struc._write_dat_head()
            dat_box = lmp_struc._write_dat_box()
            dat_atoms = lmp_struc._write_dat_atoms()
            dat_connect, _, _, _ = lmp_struc._write_dat_connects()
            with open(DIR+"/lmp_dat_"+str(i+1), "w") as f:
                f.write(dat_head)
                f.write(dat_box)
                f.write(dat_atoms)
                f.write(dat_connect)

    #@timeit
    def evaluate_ff_single(self, lmp_struc, numMols=[1], options=[True]*3,
                           lmp_dat=None,
                           lmp_in=None,
                           box=None,
                           positions=None,
                           parameters=None):
        """
        Evaluate the reference structure with the ff_evaluator
        Add explanation later

        Args:
            lmp_struc: ase structure
            numMols: list of num of molecules
            options (list): [energy, forces, stress]
            lmp_dat: lammps data file
            lmp_in: lammps input file
            box: cell parameters
            positions: atomic positions
            parameters: list of forcefield parameters
        """
        if parameters is not None: self.update_ff_parameters(parameters)

        if type(lmp_struc) == Atoms:
            self.ase_templates = {}
            self.lmp_dat = {}
            lmp_struc, lmp_dat = self.get_lmp_input_from_structure(lmp_struc, numMols)

        if box is not None: lmp_struc.box = box
        if positions is not None: lmp_struc.coordinates = positions

        replicate = len(lmp_struc.atoms) / self.natoms_per_unit
        ff_dic = {'energy': None,
                  'forces': None,
                  'stress': None,
                  'replicate': replicate,
                  'options': options,
                  'numMols': numMols,
                  }

        #eng, force, stress = get_lmp_efs(lmp_struc, lmp_in, lmp_dat)
        if not hasattr(lmp_struc, 'ewald_error_tolerance'): lmp_struc.complete()
        calc = LAMMPSCalculator(lmp_struc, lmp_in=lmp_in, lmp_dat=lmp_dat)
        eng, force, stress = calc.express_evaluation()

        if options[0]: ff_dic['energy'] = eng
        if options[1]: ff_dic['forces'] = force
        if options[2]: ff_dic['stress'] = stress

        return ff_dic

    #@timeit
    def get_opt_dict(self, terms=['vdW'], values=None, parameters=None):
        """
        Get the opt_dict as an input for optimization
        """
        if values is None:
            if parameters is None: parameters = self.params_init
            values, _, _ = self.get_sub_parameters(parameters, terms)
        else:
            assert(len(terms) == len(values))

        opt_dict = {}
        for i, term in enumerate(terms):
            if term in self.terms:
                opt_dict[term] = np.array(values[i])
            else:
                raise ValueError("Cannot the unknown FF term", term)
        return opt_dict

    def optimize_offset(self, ref_dics, parameters0=None, steps=50):
        """
        Approximate the offset energy between FF and Reference
        mean(engs_ref-engs_ff)

        Args:
            ref_dics (dict): reference data dictionary
            parameters0 (array): initial full parameters
            steps (int): optimization steps

        Returns:
            The optimized e_offset value
        """
        if parameters0 is None:
            parameters0 = self.params_init.copy()
        else:
            assert(len(parameters0) == len(self.params_init))

        results = self.evaluate_multi_references(ref_dics,
                                                 parameters0,
                                                 max_E=1000,
                                                 max_dE=1000)
        (ff_values, ref_values, _, _) = results
        (ff_eng, _, _) = ff_values
        (ref_eng, _, _) = ref_values

        x = parameters0[-1]
        if abs(x) < 1e-5:
            x = np.mean(ref_eng - ff_eng)
            print("Initial guess of offset", x)

        def obj_fun(x, ff_eng, ref_eng):
            return -compute_r2(ff_eng + x, ref_eng)

        res = minimize(
                       obj_fun,
                       [x],
                       method = 'Nelder-Mead',
                       args = (ff_eng, ref_eng),
                       options = {'maxiter': steps},
                      )

        parameters0[-1] += res.x[0]
        print("optimized offset", parameters0[-1])#; import sys; sys.exit()
        return parameters0[-1], parameters0

    def load_parameters(self, filename):
        """
        Load the parameters from a given xml file

        Args:
            filename: xml file to store the parameters information
        """
        if filename.endswith('.xml'):
            dics = xml_to_dict_list(filename)[0]
            parameters = []
            errors = {}
            for term in self.terms:
                parameters.extend(dics[term].tolist())
            for key in ['rmse_values', 'r2_values', 'ff_style']:
                if key in dics.keys():
                    errors[key] = dics[key]
            return np.array(parameters), errors
        else:
            raise ValueError("Unsupported file format")

    def export_parameters(self, filename='parameters.xml', parameters=None, err_dict=None):
        """
        Export the parameters to the xml file

        Args:
            filename: xml file to store the parameters information
            parameters: a numpy array of parameters
        """
        if parameters is None: parameters = self.params_init.copy()
        opt_dict = self.get_opt_dict(self.terms, parameters=parameters)

        # Export reference data to file
        root = ET.Element('library')
        ref_elem = ET.SubElement(root, 'structure')
        for key, val in opt_dict.items():
            child = ET.SubElement(ref_elem, key)
            if isinstance(val, np.ndarray):
                if val.ndim == 2: # Special handling for 2D arrays
                    val = array_to_string(val)
                else:  # For 1D arrays, convert to list
                    val = val.tolist()
            child.text = str(val)

        # Export error values
        if err_dict is not None:
            #ref_elem = ET.SubElement(root, 'error')
            for key, val in err_dict.items():
                child = ET.SubElement(ref_elem, key)
                child.text = str(list(val))
        child = ET.SubElement(ref_elem, 'FF_style')
        child.text = self.ff_style

        # Use prettify to get a pretty-printed XML string
        pretty_xml = prettify(root)

        # Write the pretty-printed XML to a file
        with open(filename, 'w') as f:
            f.write(pretty_xml)

    def load_references(self, filename):
        """
        Load the reference information

        Args:
            filename (str): path of reference file

        Returns:
            the list of reference dictionaries
        """
        ref_dics = []
        if filename.endswith(('.xml', '.db')):
            # Load reference data from file
            if filename.endswith('.xml'):
                dics = xml_to_dict_list(filename)
                for dic in dics:
                    dic0 = {
                            'numbers': dic['numbers'],
                            'lattice': dic['lattice'],
                            'position': dic['position'],
                            'energy': dic['energy'],
                            'forces': dic['forces'],
                            'stress': dic['stress'],
                            'replicate': dic['replicate'],
                            'options': dic['options'],
                            'tag': dic['tag'],
                            'numMols': [int(m) for m in dic['numMols']],
                           }
                    ref_dics.append(dic0)
            else:
                pass
        else:
            raise ValueError("Unsupported file format")

        return ref_dics

    def get_gs_from_ref_dics(self, ref_dics):
        """
        Get the ground state structure from the reference dictionaries
        """
        gs = []
        for ref_dic in ref_dics:
            if ref_dic['tag'] == 'minimum':
                data = ref_dic['lattice'].flatten()
                data = np.append(data, ref_dic['energy'])
                gs.append(data)
        gs = np.array(gs)
        return gs

    def get_reference_data_and_mask(self, ref_dics):
        """
        Get the reference data and mask for the objective function
        """
        eng_arr = []
        force_arr = []
        stress_arr = []
        number_arr = []
        mask_eng = []
        mask_force = []
        mask_stress = []

        for ref_dic in ref_dics:
            eng = ref_dic['energy']
            if ref_dic['forces'] is None:
                force = [0] * 3 * len(ref_dic['numbers'])
            else:
                force = ref_dic['forces'].flatten()
            if ref_dic['stress'] is None:
                stress = [0] * 6
            else:
                stress = ref_dic['stress'].flatten()
            replicate = ref_dic['replicate']
            eng_arr.append(eng)
            number_arr.append(replicate)
            force_arr.extend(force)
            stress_arr.extend(stress)
            N_force = len(force)

            if ref_dic['options'][0]:
                mask_eng.append(True)
            else:
                mask_eng.append(False)

            if ref_dic['options'][1]:
                mask_force.extend([True] * N_force)
            else:
                mask_force.extend([False] * N_force)

            if ref_dic['options'][2]:
                mask_stress.extend([True] * 6)
            else:
                mask_stress.extend([False] * 6)

        eng_arr = np.array(eng_arr)
        force_arr = np.array(force_arr)
        stress_arr = np.array(stress_arr)
        number_arr = np.array(number_arr)
        mask_eng = np.array(mask_eng)
        mask_force = np.array(mask_force)
        mask_stress = np.array(mask_stress)
        return eng_arr, force_arr, stress_arr, number_arr, mask_eng, mask_force, mask_stress

    def get_label(self, i):
        """
        A short utility to get the label for the folder.
        """
        if i < 10:
            folder = f"cpu00{i}"
        elif i < 100:
            folder = f"cpu0{i}"
        else:
            folder = f"cpu0{i}"
        return folder

    def run_lammps_evaluation(self, lmp_strucs, numMols, options, lmp_dats, lmp_in, box, coordinates):
        return self.evaluate_ff_single(lmp_strucs, numMols, options, lmp_dats, lmp_in, box, coordinates)

    def evaluate_multi_references(self, ref_dics, parameters, max_E, max_dE):
        """
        Calculate scores for multiple reference structures

        Args:
            ref_dics: list of references
            parameters: ff parameters array
            max_E: maximally allowed energy for FF
            max_dE: maximally allowed dE between FF and ref energy
        """
        self.update_ff_parameters(parameters)
        offset_opt = parameters[-1]

        ff_eng, ff_force, ff_stress = [], [], []
        ref_eng, ref_force, ref_stress = [], [], []

        lmp_strucs, lmp_dats = self.get_lmp_inputs_from_ref_dics(ref_dics)
        lmp_in = self.ff.get_lammps_in()

        #parallel process
        N_cycle = int(np.ceil(len(ref_dics)/self.ncpu))
        args_list = []
        for i in range(self.ncpu):
            folder = self.get_label(i)
            id1 = i * N_cycle
            id2 = min([id1 + N_cycle, len(ref_dics)])
            os.makedirs(folder, exist_ok=True)
            args_list.append((ref_dics[id1:id2],
                              lmp_strucs[id1:id2],
                              lmp_dats[id1:id2],
                              lmp_in,
                              offset_opt,
                              folder,
                              max_E,
                              max_dE))

        #with ProcessPoolExecutor(max_workers=self.ncpu) as executor:
        with ProcessPoolExecutor(max_workers=self.ncpu,
                                 mp_context=mp.get_context('spawn')) as executor:
            results = [executor.submit(evaluate_ff_error_par, *p) for p in args_list]

        for result in results:
            res = result.result()
            ff_eng.extend(res[0])
            if len(res[1]) > 0: ff_force.extend(res[1])
            if len(res[2]) > 0: ff_stress.extend(res[2])
            ref_eng.extend(res[3])
            if len(res[4]) > 0: ref_force.extend(res[4])
            if len(res[5]) > 0: ref_stress.extend(res[5])

        # Convert lists to numpy arrays
        ff_eng, ff_force, ff_stress = map(np.array, [ff_eng, ff_force, ff_stress])
        ref_eng, ref_force, ref_stress = map(np.array, [ref_eng, ref_force, ref_stress])

        # Compute RMSE and R² scores
        mse_values = [np.sqrt(np.mean((ff - ref) ** 2)) if len(ff) > 0 else 0 for ff, ref in zip((ff_eng, ff_force, ff_stress), (ref_eng, ref_force, ref_stress))]
        r2_values = [compute_r2(ff, ref) if len(ff) > 0 else 0 for ff, ref in zip((ff_eng, ff_force, ff_stress), (ref_eng, ref_force, ref_stress))]

        ff_values = (ff_eng, ff_force, ff_stress)
        ref_values = (ref_eng, ref_force, ref_stress)

        return ff_values, ref_values, mse_values, r2_values


    def _plot_ff_parameters(self, ax, params, term='bond-1', width=0.35):
        """
        Plot the individual parameters in bar plot style

        Args:
            ax: matplotlib axis
            params (list): list of FF parameter arrays
            term (str): e.g. 'bond-1', 'angles-1', 'vdW-1', 'charges'
        """
        term = term.split('-')
        if len(term) == 1: # applied to charge/proper
            term, seq = term[0], 0
        else: # applied for bond/angle/proper/vdW
            term, seq = term[0], int(term[1])

        for i, param in enumerate(params):
            label = 'FF' + str(i) + '-' + term
            subpara, _ , _ = self.get_sub_parameters(param, [term])
            if seq == 0:
                data = subpara[0]
            else:
                data = subpara[0][seq-1::2]
                label += '-' + str(seq)
            ind = np.arange(len(data))
            ax.bar(ind+i*width, data, width, label=label)

        if seq < 2:
            #ax.set_xlabel(term)
            ax.set_ylabel(term)
        ax.set_xticks([])
        ax.legend()

    def plot_ff_parameters(self, figname, params, figsize=(10, 16),
                terms=['bond', 'angle', 'proper', 'vdW', 'charge']):
        """
        Plot the whole FF parameters

        Args:
            figname (str): path of figname
            params: list of parameters array
            figsize:
            terms: list of FF terms
        """
        grid_size = (len(terms), 2)
        fig = plt.figure(figsize=figsize)
        for i, term in enumerate(terms):
            if term in ['charge', 'proper']:
                ax = plt.subplot2grid(grid_size, (i, 0), colspan=2, fig=fig)
                self._plot_ff_parameters(ax, params, term=term)
            else:
                ax1 = plt.subplot2grid(grid_size, (i, 0), fig=fig)
                ax2 = plt.subplot2grid(grid_size, (i, 1), fig=fig)
                self._plot_ff_parameters(ax1, params, term=term+'-1')
                self._plot_ff_parameters(ax2, params, term=term+'-2')
        plt.title('.'.join(self.smiles))
        plt.savefig(figname)
        plt.close('all')

    def plot_ff_results(self, figname, ref_dics, params, labels=None,
            max_E=1000, max_dE=1000):
        """
        Plot the ff performance results

        Args:
            figname (str): figname
            ref_dics (list): list of references
            params (list): list of parameter arrays
            labels: labels

        Return:
            performance figure and the error dictionaries
        """
        print("Number of reference structures", len(ref_dics))
        if len(params) == 1:
            if labels is None: labels = 'Opt'
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            _, err_dic = self._plot_ff_results(axes, params[0], ref_dics, labels, max_E, max_dE)
            plt.savefig(figname)
            plt.close('all')
            return [err_dic]
        else:
            if labels is None: labels = ['FF'+str(i) for i in range(len(params))]
            fig, axes = plt.subplots(len(params), 3, figsize=(16, 4*len(params)))
            err_dics = []
            for i, param in enumerate(params):
                _, err_dic = self._plot_ff_results(axes[i], param, ref_dics, labels[i], max_E, max_dE)
                err_dics.append(err_dic)
            plt.savefig(figname)
            plt.close('all')
            return err_dics

    def _plot_ff_results(self, axes, parameters, ref_dics, label,
            max_E=1000, max_dE=1000, size=None, verbose=False):
        """
        Plot the results of FF prediction as compared to the references in
        terms of Energy, Force and Stress values.
        Args:
            axes (list): list of matplotlib axes
            parameters (1D array): array of full FF parameters
            ref_dics (dict): reference data
            offset_opt (float): offset values for energy prediction
            label (str): label for the plot
            max_E (float): maximum energy value
            max_dE (float): maximum energy difference
            size (int): size of the scatter points
            verbose (bool): verbose mode to print the results
        """
        # Set up the ff engine
        self.update_ff_parameters(parameters)

        results = self.evaluate_multi_references(ref_dics, parameters, max_E, max_dE)
        (ff_values, ref_values, rmse_values, r2_values) = results
        (ff_eng, ff_force, ff_stress) = ff_values
        (ref_eng, ref_force, ref_stress) = ref_values
        (mse_eng, mse_for, mse_str) = rmse_values
        (r2_eng, r2_for, r2_str) = r2_values
        if verbose:
            print("r2 values", r2_values)
            print("ref_eng_values", ref_eng)
            print("ff_eng_values", ff_eng)

        label1 = '{:s}. Energy ({:d})\n'.format(label, len(ff_eng))
        label1 += 'Unit: [eV/mole]\n'
        label1 += 'RMSE: {:.4f}\n'.format(mse_eng)
        label1 += 'R2:   {:.4f}'.format(r2_eng)

        label2 = '{:s}. Forces ({:d})\n'.format(label, len(ff_force))
        label2 += 'Unit: [eV/A]\n'
        label2 += 'RMSE: {:.4f}\n'.format(mse_for)
        label2 += 'R2:   {:.4f}'.format(r2_for.mean())

        label3 = '{:s}. Stress ({:d})\n'.format(label, len(ff_stress))
        label3 += 'Unit: [GPa]\n'
        label3 += 'RMSE: {:.4f}\n'.format(mse_str)
        label3 += 'R2:   {:.4f}'.format(r2_str)

        print('\n', label1)
        print('\n', label2)
        print('\n', label3)
        print(f'\nMin_values: {ff_eng.min():.4f} {ref_eng.min():.4f}')
        axes[0].scatter(ref_eng, ff_eng, s=size, label=label1)
        axes[1].scatter(ref_force, ff_force, s=size, label=label2)
        axes[2].scatter(ref_stress, ff_stress, s=size, label=label3)

        for ax in axes:
            ax.set_xlabel('Reference')
            ax.set_ylabel('FF')
            ax.legend(loc=2)

        err_dict = {
                    'rmse_values': (mse_eng, mse_for, mse_str),
                    'r2_values': (r2_eng, r2_for, r2_str),
                    'min_values': (ff_eng.min(), ref_eng.min()),
                   }
        return axes, err_dict

    def remove_duplicate_structures(self, xml_file, overwrite=True):
        """
        Remove duplicate structures in the xml file by comparing the numbers,
        lattice and position of the structures.

        Args:
            xml_file (str): xml file containing the structures
            overwrite (bool): overwrite the original file
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        seen_structures = set()
        unique_structures = []

        for structure in root.findall("structure"):
            numbers = structure.find("numbers").text.strip()
            lattice = structure.find("lattice").text.strip().replace("\n", " ")
            position = structure.find("position").text.strip().replace("\n", " ")

            # Create a unique key for the structure
            structure_key = (numbers, lattice, position)
            if structure_key not in seen_structures:
                seen_structures.add(structure_key)
                unique_structures.append(structure)

        # Clear original structures
        root.clear()

        # Add only unique structures back
        for structure in unique_structures: root.append(structure)

        # Overwrite the original file
        if overwrite:
            tree.write(xml_file)
            print(f"Overwritten {xml_file} with {len(unique_structures)} structures.")

class ForceFieldParameters(ForceFieldParametersBase):
    def __init__(self,
                 smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O'],
                 style = 'gaff',
                 chargemethod = 'am1bcc',
                 ref_evaluator = None,
                 ff_evaluator = 'lammps',
                 e_coef = 1.0,
                 f_coef = 0.1,
                 s_coef = 1.0,
                 ncpu = 1,
                 verbose = True):

        ForceFieldParametersBase.__init__(
                self,
                smiles,
                style,
                chargemethod,
                ff_evaluator,
                e_coef,
                f_coef,
                s_coef,
                ncpu,
                verbose,
                )
        if ref_evaluator is not None:
            self.set_ref_evaluator(ref_evaluator)

    def set_ref_evaluator(self, ref_evaluator, device='cpu'):
        """
        Set the reference evaluator
        """
        self.ref_evaluator = ref_evaluator

        if ref_evaluator == 'mace':
            from mace.calculators import mace_mp
            self.calculator = mace_mp(model = "small",
                                      dispersion = True,
                                      default_dtype = "float32",
                                      device = device)
        elif ref_evaluator == 'ani':
            from torchani import models
            self.calculator = models.ANI2x().ase()
        else:
            raise ValueError("Unknown ref_evaluator")

    def check_validity(self, parameters):
        """
        Check if the input FF parameters are within the bound
        and satisfy the constaint
        """
        # last parameter is for the offset
        for i, _parameter in enumerate(parameters[:-1]):
            (lb, ub) = self.bounds[i]
            if _parameter >= ub:
                _parameter = ub
            elif _parameter <= lb:
                _parameter = lb

        # Rescale the partial charges
        for constraint in self.constraints:
            (id1, id2, sum_chg) = constraint
            diff = sum(parameters[id1:id2]) - sum_chg
            if abs(diff) > 1e-2:
                for id in range(id1, id2):
                    parameters[id] += diff/(id2-id1)
        return parameters


    def evaluate_single_reference(self, ref_dic, parameters):
        """
        Evaluate the FF performance for a single reference structure

        Args:
            ref_dic: reference data dictionary
            parameters: FF parameters array
        """
        f_mse, f_r2, s_mse, s_r2 = 0, 0, 0, 0
        self.update_ff_parameters(parameters)
        offset_opt = parameters[-1]

        structure = prepare_atoms(ref_dic)
        options = ref_dic['options']

        ff_dic = self.evaluate_ff_single(structure, ref_dic['numMols'], options, None)
        e_diff = ff_dic['energy']/ff_dic['replicate'] + offset_opt - ref_dic['energy']/ff_dic['replicate']
        print(ff_dic['energy'], ref_dic['energy'])
        if options[1]:
            f1 = ff_dic['forces'].flatten()
            f2 = ref_dic['forces'].flatten()
            f_mse = np.sum((f1-f2)**2)/len(f1)
            f_r2 = compute_r2(f1, f2)
        if options[2]:
            s1 = ff_dic['stress'].flatten()
            s2 = ref_dic['stress'].flatten()
            f_mse = np.sum((s1-s2)**2)/len(s1)
            f_r2 = compute_r2(s1, s2)
        return e_diff, f_mse, f_r2, s_mse, s_r2

    def same_lmp(self, struc1, struc2):
        """
        Quick comparison for two lmp structures

        Args:
            struc1: lmp structure
            struc2: lmp structure
        """
        for i in range(len(struc1.dihedrals)):
            d1 = struc1.dihedrals[i]
            d2 = struc2.dihedrals[i]
            id1 = [d1.atom1.idx, d1.atom2.idx, d1.atom3.idx, d1.atom4.idx]
            id2 = [d2.atom1.idx, d2.atom2.idx, d2.atom3.idx, d2.atom4.idx]
            if id1 != id2:
                print("Different structures were found, check 1.xyz and 2.xyz")
                struc1.to_ase('1.xyz', format='1.xyz')
                struc2.to_ase('2.xyz', format='2.xyz')
                return False
        return True

    #@timeit
    def optimize_init(self, ref_dics, opt_dict, parameters0=None, obj='MSE'):
        """
        Initialize optimization by setting up parameters, bounds, and constraints.
        """

        if parameters0 is None:
            _, parameters0 = self.optimize_offset(ref_dics)
        else:
            assert len(parameters0) == len(self.params_init)

        self.update_ff_parameters(parameters0)
        terms = list(opt_dict.keys())

        # Move 'charge' term to the end if present
        if 'charge' in terms:
            terms.remove('charge')
            terms.append('charge')
            charges = opt_dict['charge']
        else:
            charges = None

        # Prepare the input for optimization
        x = []
        ids = []

        for term in terms:
            N = getattr(self, f'N_{term}', None)  # Get number of parameters for this term
            if N is None:
                raise ValueError(f"Unknown term {term}. Expected attribute N_{term}.")

            if len(ids) > 0:
                x.extend(opt_dict[term])
                ids.append(ids[-1] + N)  # Append cumulative sum
            else:
                x.extend(opt_dict[term])
                ids.append(N)  # First term, just store N

        # Debugging to check correctness
        print(f"DEBUG: Final ids list: {ids}")
        print(f"DEBUG: Terms: {terms}")
        print(f"DEBUG: Expected charge range: {ids[-2]} to {ids[-1]}" if 'charge' in terms else "DEBUG: No charge term")

        _, sub_bounds, _ = self.get_sub_parameters(parameters0, terms)
        bounds = [item for sublist in sub_bounds for item in sublist]

        # Create function arguments bundle
        fun_args = (self, ref_dics, parameters0, obj, ids, terms, charges)

        # Use `partial` to create a function that expects only `x`
        obj_partial = partial(opt_obj_fun, fun_args=fun_args)

        return x, bounds, obj_partial, fun_args


    def optimize_post(self, x, ids, charges):
        """
        Post-process the optimized parameters to the list of values
        """
        # Rearrange the optimized parameters to the list of values
        #ids = fun_args[-2]
        values = []
        for i in range(len(ids)):
            if i == 0:
                id1 = 0
            else:
                id1 = ids[i-1]
            values.append(x[id1:ids[i]])
        # The last x value is the ratio of charge
        if charges is not None:
            try:
                charges= np.array(charges,dtype=np.float64)
            except ValueError as e:
                values.append(x[-1] * charges)
                #print(f"ERROR: Cannot convert `charges` to float. Current value: {charges}")
                raise e
        return values

    def optimize_global(self, ref_dics, opt_dict, parameters0=None, steps=100, obj='MSE', t0=100):
        """
        Simulated annealing optimization with batch parallel objective function evaluation.
        Obsolete: Don't use this function for optimization.
        """
        x, bounds, obj_fun, fun_args = self.optimize_init(ref_dics, opt_dict, parameters0, obj)
        bounds = np.array(bounds)  # Convert bounds to NumPy array for faster operations
        t = t0
        current_x = x
        # Use `partial` correctly before entering multiprocessing
        obj_partial = partial(obj_fun, fun_args=fun_args)

        # First function evaluation
        current_fun = obj_partial(current_x)
        best_x, best_fun = current_x, current_fun

        # Optimize batch size
        batch_size = min(self.ncpu, len(bounds))
        candidate_xs = np.tile(current_x, (batch_size, 1))  # Generate multiple candidate solutions

        with Pool(processes=self.ncpu) as pool:
            for i in range(steps):
                t_start = time.time()

                # Generate multiple candidates at once (batch processing)
                perturbation = 0.02 * np.random.uniform(-1, 1, size=candidate_xs.shape) * (bounds[:, 1] - bounds[:, 0])
                candidate_xs = np.clip(current_x + perturbation, bounds[:, 0], bounds[:, 1])

                # Evaluate all candidates in parallel
                t_obj_start = time.time()
                candidate_funs = pool.starmap(obj_partial, [(x,) for x in candidate_xs], chunksize=1)
                t_obj_end = time.time()

                # Select the best candidate from batch
                best_idx = np.argmin(candidate_funs)
                candidate_x = candidate_xs[best_idx]
                candidate_fun = candidate_funs[best_idx]

                # Accept the best candidate with probability
                if candidate_fun < best_fun:
                    best_x, best_fun = candidate_x.copy(), candidate_fun

                if np.random.random() < np.exp((current_fun - candidate_fun) / t):
                    current_x, current_fun = candidate_x, candidate_fun

                t = t0 / (1 + 0.1 * i)  # Faster cooling

                if self.verbose and i % 10 == 0:
                    print(f"Step {i:4d}, Temp: {t:.2f}, Best: {best_fun:.4f}, Candidate: {candidate_fun:.4f}")
                    print(f"Step {i} took {time.time() - t_start:.4f} sec (Obj: {t_obj_end - t_obj_start:.4f} sec)")

        print(f"Best results after {steps} steps: {best_fun:.4f}")

        values = self.optimize_post(best_x, fun_args[-3], fun_args[-1])
        return best_x, best_fun, values, steps

    def optimize_local(self, ref_dics, opt_dict, parameters0=None, steps=100, obj='MSE'):
        """
        FF parameters' local optimization using the Nelder-Mead algorithm
        Obsolete: Don't use this function for optimization.

        Args:
            ref_dics (dict): reference data dictionary
            opt_dict (dict): optimization terms and values
            parameters0 (array): initial full parameters
            steps (int): optimization steps
        Returns:
            The optimized values
        """
        x, bounds, obj_fun, fun_args = self.optimize_init(ref_dics, opt_dict, parameters0, obj)

        class CallbackFunction:
            def __init__(self):
                self.iteration = 0  # Initialize iteration count

            def callback(self, xk):
                self.iteration += 1  # Increment iteration count
                if self.iteration % 10 == 0:
                    print(f"Step {self.iteration:4d} {last_function_value:.4f}")

        callback = CallbackFunction() if self.verbose else None

        res = minimize(obj_fun,
                       x,
                       method = 'Nelder-Mead',
                       args = fun_args, #(ref_dics, paras, e_offset, ids, charges),
                       options = {'maxiter': steps, 'disp': True},
                       bounds = bounds,
                       callback = callback.callback,
                       )

        # Rearrange the optimized parameters to the list of values
        values = self.optimize_post(res.x, fun_args[-3], fun_args[-1])
        print("Final Obj", res.fun)
        return res.x, res.fun, values, res.nfev

    def cut_references(self, ref_dics, cutoff):
        """
        Cut the list of references by energy threshold

        Args:
            ref_dics: list of reference configuration in dict format
            cutoff: energy cutoff
        """
        N0 = len(ref_dics)
        engs = []
        for ref_dic in ref_dics:
            engs.append(ref_dic['energy']/ref_dic['replicate'])
        engs = np.array(engs)
        eng_min = np.min(engs)
        _ref_dics = []
        for i, ref_dic in enumerate(ref_dics):
            if engs[i] < eng_min + cutoff:
                _ref_dics.append(ref_dic)
        print("Reduce references {:d} => {:d}".format(N0, len(_ref_dics)))
        return _ref_dics

    def select_references(self, ref_dics, fields=['CSP', 'minimum']):
        """
        Cut the list of references by energy
        """
        assert(type(fields) == list)
        N0 = len(ref_dics)
        _ref_dics = []
        for ref_dic in ref_dics:
            if ref_dic['tag'] in fields:
                _ref_dics.append(ref_dic)
        print("Reduce references {:d} => {:d}".format(N0, len(_ref_dics)))
        return _ref_dics

    def export_references(self, ref_dics, filename='reference.xml'):
        """
        Export the reference configurations to xml or ase.db

        Args:
            ref_dics: list of reference configuration in dict format
            filename: filename
        """
        if filename.endswith(('.xml', '.db')):
            # Export reference data to file
            if filename.endswith('.xml'):
                root = ET.Element('library')
                for ref_dic in ref_dics:
                    ref_elem = ET.SubElement(root, 'structure')
                    for key, val in ref_dic.items():
                        if type(val) == Atoms:
                            lattice = array_to_string(val.cell.array)
                            position = array_to_string(val.positions)
                            numbers = str(val.numbers.tolist())
                            child1 = ET.SubElement(ref_elem, 'lattice')
                            child1.text = lattice
                            child2 = ET.SubElement(ref_elem, 'position')
                            child2.text = position
                            child3 = ET.SubElement(ref_elem, 'numbers')
                            child3.text = numbers
                        else:
                            child = ET.SubElement(ref_elem, key)
                            if isinstance(val, np.ndarray):
                                if val.ndim == 2: # Special handling for 2D arrays
                                    val = array_to_string(val)
                                else:  # For 1D arrays, convert to list
                                    val = val.tolist()
                            child.text = str(val)

                # Use prettify to get a pretty-printed XML string
                pretty_xml = prettify(root)

                # Write the pretty-printed XML to a file
                with open(filename, 'w') as f:
                    f.write(pretty_xml)

            elif filename.endswith('.db'):
                # Ase database
                pass
        else:
            raise ValueError("Unsupported file format")

    def get_ase_charmm(self, params):
        """
        Prepare the charmm input files with the updated params.

        Args:
            params: FF parameters array

        Returns:
            ase_atoms object with the charmm ff information
        """
        from pyocse.interfaces.parmed import ParmEdStructure
        from pyocse.charmm import CHARMMStructure

        self.ff.update_parameters(params)
        n_mols = [1] * len(self.ff.smiles)
        if sum(n_mols) == 1:
            pd_struc = self.ff.molecules[0].copy(cls=ParmEdStructure)
        else:
            from functools import reduce
            from operator import add
            mols = []
            for i, m in enumerate(n_mols):
                mols += [self.ff.molecules[i] * m]
            pd_struc = reduce(add, mols)

        ase_with_ff = CHARMMStructure.from_structure(pd_struc)
        #ase_with_ff.write_charmmfiles(base='pyxtal')
        return ase_with_ff

    def clean_ref_dics(self, ref_dics, criteria={"O-O": 2.0}):
        """
        Remove the unwanted reference structures by criteria like
        unwanted bonding, e.g., O-O

        Args:
            ref_dics: list of reference dics
            criteria: dictionary of bond type and tolerance

        Returns:
            ref_dics with the removed unwanted entries
        """
        from pyxtal.util import ase2pymatgen

        mols = [smi+'.smi' for smi in self.smiles]
        _ref_dics = []
        for i, ref_dic in enumerate(ref_dics):
            c = pyxtal(molecular=True)
            structure = prepare_atoms(ref_dic)
            pmg = ase2pymatgen(structure)
            try:
                c.from_seed(pmg, molecules=mols)
                if c.check_short_distances_by_dict(criteria) == 0:
                    _ref_dics.append(ref_dic)
            except:
                print(i, "Unable to convert to pyxtal", ref_dic['tag'])
        print("Removed {:d} entries by geometry".format(len(ref_dics)-len(_ref_dics)))
        return _ref_dics

    def generate_report(self, ref_dics, parameters):
        """
        Run quick report about the performance of each reference structure
        Add explanations about the performance of the FF later

        Args:
            ref_dics:
            parameters:

        Returns:
            Printed values in terms of Energy/Forces/Stress tensors.
        """
        self.update_ff_parameters(parameters)
        for i, ref_dic in enumerate(ref_dics):
            # Remove the templates
            self.ase_templates = {}
            self.lmp_dat = {}
            structure = prepare_atoms(ref_dic)
            ff_dic = self.evaluate_ff_single(structure, ref_dic['numMols'])
            e1 = ff_dic['energy']/ff_dic['replicate'] + parameters[-1]
            e2 = ref_dic['energy']/ff_dic['replicate']
            print('\nStructure {:3d}'.format(i))
            print('Energy_ff_ref: {:8.3f} {:8.3f} {:8.3f}'.format(e1, e2, e1-e2))

            if ref_dic['options'][1]:
                f1 = ff_dic['forces'].flatten()
                f2 = ref_dic['forces'].flatten()
                rmse = np.sum((f1-f2)**2) / len(f2)
                r2 = compute_r2(f1, f2)
                print('Forces-R2-MSE: {:8.3f} {:8.3f}'.format(r2, rmse))

            if ref_dic['options'][2]:
                s1 = ff_dic['stress']
                s2 = ref_dic['stress']
                rmse = np.sum((s1-s2)**2) / len(s2)
                r2 = compute_r2(s1, s2)
                print('Stress_ff    : {:8.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}'.format(*s1))
                print('Stress_ref   : {:8.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}'.format(*s2))
                print('Stress-R2-MSE: {:8.5f} {:8.5f}'.format(r2, rmse))

    def compute_references(self, strucs, numMols, options, tags, steps):
        """
        Evaluate the reference structures in parallel
        """
        N_cycle = int(np.ceil(len(strucs)/self.ncpu))
        args_list = []
        for i in range(self.ncpu):
            folder = self.get_label(i)
            id1 = i * N_cycle
            id2 = min([id1 + N_cycle, len(strucs)])
            os.makedirs(folder, exist_ok=True)
            # print("# parallel process", N_cycle, id1, id2)
            args_list.append((strucs[id1:id2],
                              numMols[id1:id2],
                              self.calculator,
                              self.natoms_per_unit,
                              options[id1:id2],
                              tags[id1:id2],
                              steps[id1:id2]))
        ref_dics0 = []
        with ProcessPoolExecutor(max_workers=self.ncpu,
                                 mp_context=mp.get_context('spawn')) as executor:
            results = [executor.submit(compute_refs_par, *p) for p in args_list]
            for result in results:
                res = result.result()
                ref_dics0.extend(res)
        return ref_dics0

    def add_references(self, xtals, ref_gs, N_max, steps=50, max_E=1000, min_dE=0.01):
        """
        Add references from the given structure pool

        Args:
            xtals: list of pyxtal structures
            ref_gs: list of ref_gs (10 numnbers of lattice parameters and energy)
            N_max: maximum number of references to add
            steps: number of steps for relaxation
            max_E: maximum energy for the reference
            min_dE: minimum energy difference for the reference

        Returns:
            list of reference dics
        """
        numMols = [xtal.numMols for xtal in xtals]
        strucs = [xtal.to_ase(resort=False) for xtal in xtals]
        N = len(strucs)
        print(f'# Process references (minimum): {N} / {self.ncpu}')
        ref_dics0 = self.compute_references(strucs, numMols,
                                            [[True, True, True]] * N,
                                            ['minimum'] * N,
                                            [steps] * N)
        #print("debug ref_gs", ref_gs)
        ref_dics = []
        for ref_dic in ref_dics0:
            ref_dics, ref_gs = self.process_ref_dic(ref_dic, ref_dics, ref_gs, max_E, min_dE)
            if len(ref_dics) >= N_max:
                break
        print(f'# Added references (minimum): {N}')
        return ref_dics

    def augment_references(self, refs, N_vibs=1, N_ela_relax=20):
        """
        Add augmented references to the reference pool

        Args:
            ref_dics: list of reference dics
            N_vibs (int): number of vibrational samples to be generated
            N_ela_relax (int): number of steps for relaxation of elastic samples
        Returns:
            list of reference dics
        """
        print(f'# Augment references from {len(refs)} structures')
        strucs, numMols, options, tags, steps = [], [], [], [], []
        for ref_dic in refs:
            ref_structure = prepare_atoms(ref_dic)
            print("# Reference", np.diag(ref_structure.cell.array), ref_dic['energy'])

            # Generate elastic samples
            elastics = self.generate_elastics(ref_structure)
            strucs.extend(elastics)
            numMols.extend([ref_dic['numMols']] * len(elastics))
            options.extend([[True, False, True]] * len(elastics))
            tags.extend(['elastic'] * len(elastics))
            steps.extend([N_ela_relax] * len(elastics))

            # Generate vibrational samples
            vibs = self.generate_vibs(ref_structure, N_vibs=N_vibs)
            strucs.extend(vibs)
            numMols.extend([ref_dic['numMols']] * len(vibs))
            options.extend([[True, True, False]] * len(vibs))
            tags.extend(['vibration'] * len(vibs))
            steps.extend([0] * len(vibs))

        print(f'# Augmented references: {len(strucs)}')
        ref_dics = self.compute_references(strucs, numMols, options, tags, steps)
        return ref_dics

    def process_ref_dic(self, ref_dic, ref_dics, ref_gs, max_E, min_dE=0.05):
        """
        Check if the reference structure is in the list of ground states
        Used by add_references.

        Args:
            ref_dic: reference dic
            ref_dics: list of reference dics
            ref_gs: list of ground states
            max_E: maximum energy for the reference
            min_dE: minimum energy difference for the reference

        Returns:
            Updated ref_dics, ref_gs
        """
        data = np.append(ref_dic['lattice'].flatten(), ref_dic['energy'])
        e_ref = ref_dic['energy']

        # Here we only count the low energy structures with large energy difference
        if ref_dic['energy'] < max_E:
            ff_dic = self.evaluate_ff_single(prepare_atoms(ref_dic), ref_dic['numMols'])
            e_ff = ff_dic['energy'] / ff_dic['replicate'] + self.params_init[-1]
            if abs(e_ref - e_ff) > min_dE:
                add = False
                # Further check if the structure has been added to the list
                if len(ref_gs) == 0:
                    ref_gs = np.array([data])
                    add = True
                elif not np.any(np.all(ref_gs == data, axis=1)):
                    ref_gs = np.append(ref_gs, [data], axis=0)
                    add = True
                else:
                    print("Duplicated structure found", data)

                if add:
                    #print("Add reference", ref_dic['tag'], ref_dic['energy'])
                    ref_dics.append(ref_dic)
                    ref_dic['tag'] = 'minimum'
        return ref_dics, ref_gs

    def generate_elastics(self, ref_structure, coefs_strain = [0.85, 0.92, 1.10, 1.25]):
        """
        Generate elastic configurations.
        The main idea is to apply various strains to the reference structure

        Args:
            ref_structure: reference structure in ASE format
            coefs_strain: list of strain coefficients
        """
        cell0 = ref_structure.cell.array
        elastics = []
        for ax in range(3):
            for coef in coefs_strain:
                structure = ref_structure.copy()
                cell = cell0.copy()
                cell[ax, ax] *= coef
                structure.set_cell(cell, scale_atoms=True)
                elastics.append(structure)
        print(f'# Get elastic configurations: {len(elastics)}/{len(coefs_strain)}')
        return elastics

    def generate_vibs(self, ref_structure, dxs = [0.01, 0.02, 0.03], N_vibs=1):
        """
        Generate vibration configurations.
        The main idea is to perturb the atomic positions with various dx values

        Args:
            ref_structure: reference structure in ASE format
            dxs: list of perturbation
            N_vibs: number of vibrational samples to be generated
        """
        print(f'# Get purturbation: {N_vibs} * {len(dxs)}')
        pos0 = ref_structure.get_positions()
        vibs = []
        for dx in dxs:
            structure = ref_structure.copy()
            pos = pos0.copy()
            pos += np.random.uniform(-dx, dx, size=pos0.shape)
            structure.set_positions(pos)
            vibs.append(structure)
        return vibs

    def add_references_from_cif(self, cif, ref_gs=[], N_max=10, augment=True, steps=120, N_vibs=3):
        """
        Add multiple references to training

        Args:
            cif (str): cif file
            N_max (int): maximum number of references
            augment (bool): augment the references or not
            steps (int): number of steps for relaxation
            N_vibs (int): number of vibrational samples to be generated

        Returns:
            list of reference dics
        """
        from pyxtal.util import parse_cif
        strs, engs = parse_cif(cif, eng=True)
        N_max = min([N_max, len(strs)])
        ids = np.argsort(engs)[:N_max]
        strs = [strs[id] for id in ids if engs[id] < 1000] # sort by eng
        smiles = [smi+'.smi' for smi in self.ff.smiles]
        xtals = []

        N_cycle = int(np.ceil(len(strs)/self.ncpu))
        args_list = []
        for i in range(self.ncpu):
            id1 = i * N_cycle
            id2 = min([id1+N_cycle, len(strs)])
            args_list.append((strs[id1:id2], smiles))

        with ProcessPoolExecutor(max_workers=self.ncpu,
                                 mp_context=mp.get_context('spawn')) as executor:
            results = [executor.submit(add_strucs_par, *p) for p in args_list]
            for result in results:
                res = result.result()
                if len(res) > 0: xtals.extend(res)

        ref_dics = self.add_references(xtals, ref_gs, N_max, steps, 1000)
        if augment:
            aug_dics = self.augment_references(ref_dics, ref_gs, N_vibs)
            ref_dics.extend(aug_dics)
        return ref_dics

    def cut_references_by_error(self, ref_dics, parameters, dE=4.0, FMSE=4.0, SMSE=5e-4):
        """
        Cut the list of references by error

        Args:
            ref_dics (list): all reference structures
            parameters (array): ff parmater
            dE (float): maximally allowed Energy error
            FMSE (float): maximally allowed Force error
            SMSE (float): maximally allowed Stress error
        """
        _ref_dics = []
        self.update_ff_parameters(parameters)
        for ref_dic in ref_dics:
            self.ase_templates = {}
            self.lmp_dat = {}
            structure = prepare_atoms(ref_dic)
            ff_dic = self.evaluate_ff_single(structure, ref_dic['numMols'])
            e1 = ff_dic['energy'] / ff_dic['replicate'] + parameters[-1]
            e2 = ref_dic['energy'] / ff_dic['replicate']
            if abs(e1-e2) < dE:
                add = True
                if ref_dic['options'][1]:
                    f1 = ff_dic['forces'].flatten()
                    f2 = ref_dic['forces'].flatten()
                    rmse = np.sum((f1-f2)**2) / len(f2)
                    if rmse > FMSE:
                        add = False
                if add and ref_dic['options'][2]:
                    s1 = ff_dic['stress']
                    s2 = ref_dic['stress']
                    rmse = np.sum((s1-s2)**2) / len(s2)
                    if rmse > SMSE:
                        add = False
                if add:
                    _ref_dics.append(ref_dic)

        print("Removed {:d} entries by error".format(len(ref_dics)-len(_ref_dics)))
        return _ref_dics


if __name__ == "__main__":
    from pyxtal.db import database

    # db = database('../HT-OCSP/benchmarks/Si.db')
    db = database("../HT-OCSP/benchmarks/test.db")
    style = 'gaff' #'openff'
    style = 'openff'
    #xtal = db.get_pyxtal("ACSALA")
    xtal = db.get_pyxtal("XATJOT") #"XAFQON")#
    #xtal = db.get_pyxtal("KONTIQ09")
    smiles = [mol.smile for mol in xtal.molecules]
    assert smiles[0] is not None
    params = ForceFieldParameters(smiles, style=style, ncpu=2)
    #smiles[0] = '[Cl-]'
    print(params)
    params0 = params.params_init.copy()
    ase_with_ff = params.get_ase_charmm(params0)
    ase_with_ff.write_charmmfiles(base='pyxtal')#, style=style)
    ff_dic = params.evaluate_ff_single(xtal.to_ase(resort=False), xtal.numMols); print(ff_dic)