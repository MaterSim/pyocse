import xml.etree.ElementTree as ET
from xml.dom import minidom
import ast
import os, time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from math import ceil
import matplotlib.pyplot as plt

from ase import Atoms
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry

from pyxtal import pyxtal
from pyxtal.util import ase2pymatgen
from pymatgen.core import Structure

from ost.utils import reset_lammps_cell
from ost.forcefield import forcefield
from ost.lmp import LAMMPSCalculator
from ost.interfaces.parmed import ParmEdStructure
from ost.charmm import CHARMMStructure

from mace.calculators import mace_mp
from lammps import PyLammps  # , get_thermo_data

#import multiprocessing as mp

def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        t = end_time - start_time
        print(f"{method.__name__} took {t} seconds to execute.")
        #if t > 2: import sys; sys.exit()
        return result
    return timed


def string_to_array(s):
    """Converts a formatted string back into a 1D or 2D NumPy array."""
    # Split the string into lines
    lines = s.strip().split('\n')

    # Check if it's a 1D or 2D array based on the number of lines
    if len(lines) == 1:
        # Treat as 1D array if there's only one line
        array = np.fromstring(lines[0][1:-1], sep=',')
        #print(lines); print(lines[0][1:-1]); print(array); import sys; sys.exit()
    else:
        # Treat as 2D array if there are multiple lines
        array = [np.fromstring(line, sep=' ') for line in lines if line]
        array = np.array(array)

    return array

def array_to_string(arr):
    """Converts a 2D NumPy array to a string format with three numbers per line."""
    lines = []
    for row in arr:
        for i in range(0, len(row), 3):
            line_segment = ' '.join(map(str, row[i:i+3]))
            lines.append(line_segment)
    return '\n' + '\n'.join(lines) + '\n'

def xml_to_dict_list(filename):
    # Parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    data = []
    for item in root.findall('structure'):
        item_dict = {}
        for child in item:
            key = child.tag
            text = child.text.strip()
            # Check if the field should be converted back to an array
            if key in ['lattice', 'position', 'forces', 'numbers', 'stress',
                       'bond', 'angle', 'proper', 'vdW', 'charge', 'offset',
                       'rmse_values', 'r2_values']:

                if text != 'None':
                    #print(text)
                    value = string_to_array(text)
                    #if key == 'position': print(value, value.shape)
                    #if key == 'stress': print(value, value.shape)
                else:
                    value = None
            elif key in ['options']:
                value = ast.literal_eval(text) #print(value)
            else:
                # Attempt to convert numeric values back to float/int
                try:
                    value = float(text)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    if text == 'None':
                        value = None
                    else:
                        value = text

            item_dict[key] = value
        data.append(item_dict)

    return data


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def compute_r2(y_true, y_pred):
    """
    Compute the R-squared coefficient for the actual and predicted values.

    :param y_true: The actual values.
    :param y_pred: The predicted values by the regression model.
    :return: The R-squared value.
    """
    if len(y_true) > 0:
        # Calculate the mean of actual values
        mean_y_true = sum(y_true) / len(y_true)

        # Total sum of squares (SST)
        sst = sum((y_i - mean_y_true) ** 2 for y_i in y_true)

        # Residual sum of squares (SSE)
        sse = sum((y_true_i - y_pred_i) ** 2 for y_true_i, y_pred_i in zip(y_true, y_pred))

        # R-squared
        r2 = 1 - (sse / sst)
    else:
        r2 = 0

    return r2

def get_lmp_efs(lmp_struc, lmp_in, lmp_dat):
    if not hasattr(lmp_struc, 'ewald_error_tolerance'):
        lmp_struc.complete()
    #print('get_lmp_efs', len(dir(lmp_struc)), hasattr(lmp_struc, 'ewald_error_tolerance'))
    calc = LAMMPSCalculator(lmp_struc, lmp_in=lmp_in, lmp_dat=lmp_dat)
    return calc.express_evaluation()

def evaluate_ff_par(ref_dics, lmp_strucs, lmp_dats, lmp_in, e_offset, E_only,
        natoms_per_unit, f_coef, s_coef, dir_name, obj):
    """
    parallel version
    """
    #print("parallel version", E_only)
    pwd = os.getcwd()
    total_mse = 0.0
    eng_arr, force_arr, stress_arr = [[], []], [[], []], [[], []]
    os.chdir(dir_name)

    for ref_dic, lmp_struc, lmp_dat in zip(ref_dics, lmp_strucs, lmp_dats):
        options = ref_dic['options']
        efs = evaluate_structure(ref_dic['structure'],
                                 lmp_struc,
                                 lmp_dat,
                                 lmp_in,
                                 natoms_per_unit)
        result = obj_from_efs(efs, ref_dic, e_offset, E_only, f_coef, s_coef, obj)
        if obj == 'MSE':
            total_mse += result
        else:
            engs, forces, stresses = result
            eng_arr[0].append(engs[0])
            eng_arr[1].append(engs[1])
            if len(forces) > 0:
                force_arr[0].extend(forces[0])
                force_arr[1].extend(forces[1])
            if len(stresses) > 0:
                stress_arr[0].extend(stresses[0])
                stress_arr[1].extend(stresses[1])
    #print(len(eng_arr[0]), len(force_arr[1]), len(stress_arr[0]), '++++++++++++++')

    os.chdir(pwd)
    if obj == 'MSE':
        return total_mse
    else:
        return (eng_arr, force_arr, stress_arr)

def evaluate_ff_error_par(ref_dics, lmp_strucs, lmp_dats, lmp_in, e_offset,
        natoms_per_unit, f_coef, s_coef, dir_name, max_dE=1.25, max_E=1000.0):
    """
    parallel version
    """
    pwd = os.getcwd()
    os.chdir(dir_name)

    result = 0.0

    ff_eng, ff_force, ff_stress = [], [], []
    ref_eng, ref_force, ref_stress = [], [], []

    for ref_dic, lmp_struc, lmp_dat in zip(ref_dics, lmp_strucs, lmp_dats):
        options = ref_dic['options']
        replicate = ref_dic['replicate']
        eng, force, stress = evaluate_structure(ref_dic['structure'],
                                                lmp_struc,
                                                lmp_dat,
                                                lmp_in,
                                                natoms_per_unit)
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

def evaluate_structure(structure, lmp_struc, lmp_dat, lmp_in, natoms_per_unit):
    replicate = len(structure)/natoms_per_unit
    lmp_struc.box = structure.cell.cellpar()
    lmp_struc.coordinates = structure.get_positions()
    return get_lmp_efs(lmp_struc, lmp_in, lmp_dat)

def obj_from_efs(efs, ref_dic, e_offset, E_only, f_coef, s_coef, obj):
    """
    Compute the objective from a single ff_dic.
    If obj is MSE, return mse value
    If obj is r2, return (eng, force, stress) array
    """
    mse = 0
    (eng, force, stress) = efs
    eng_arr, force_arr, stress_arr = [], [[], []], [[], []]

    if ref_dic['options'][0]:
        e1 = eng / ref_dic['replicate'] + e_offset
        e2 = ref_dic['energy'] / ref_dic['replicate']
        mse += (e1-e2) ** 2
        eng_arr = [e1, e2]

    if not E_only:
        if ref_dic['options'][1]:
            f1 = force.flatten()
            f2 = ref_dic['forces'].flatten()
            f_diff = f1 - f2
            mse += f_coef * np.sum(f_diff ** 2)
            force_arr[0] = f1
            force_arr[1] = f2

        if ref_dic['options'][2]:
            s1 = stress.flatten()
            s2 = ref_dic['stress'].flatten()
            s_diff = s1 - s2
            mse += s_coef * np.sum(s_diff ** 2)
            stress_arr[0] = s1
            stress_arr[1] = s2

    if obj == 'MSE':
        return mse
    else:
        return (eng_arr, force_arr, stress_arr)

def evaluate_ref_par(structures, calculator, natoms_per_unit,
                        options=[True, True, True]):
    """
    evaluate the reference structure with the ref_evaluator
    """
    ref_dics = []
    for struc in structures:
        ref_dics.append(evaluate_ref_single(struc,
                                      calculator,
                                      natoms_per_unit,
                                      options))
    return ref_dics

def evaluate_ref_single(structure, calculator, natoms_per_unit,
                        options=[True, True, True], relax=False):
    """
    evaluate the reference structure with the ref_evaluator
    """
    structure = reset_lammps_cell(structure)
    ref_dic = {'structure': structure,
               'energy': None,
               'forces': None,
               'stress': None,
               'replicate': len(structure)/natoms_per_unit,
               'options': options,
               'tag': 'CSP',
              }
    structure.set_calculator(calculator)
    if relax:
        structure.set_constraint(FixSymmetry(structure))
        ecf = ExpCellFilter(structure)
        dyn = FIRE(ecf, a=0.1, logfile='-')
        dyn.run(fmax=0.1, steps=150)
        structure.set_constraint()

    if options[0]: # Energy
        ref_dic['energy'] = structure.get_potential_energy()
    if options[1]: # forces
        ref_dic['forces'] = structure.get_forces()
    if options[2]:
        ref_dic['stress'] = structure.get_stress()
    structure.set_calculator() # reset calculator to None

    return ref_dic

def augment_ref_par(strucs, calculator, steps, N_vibs, n_atoms_per_unit, folder, logfile='-', fmax=0.1):
    """
    parallel version
    """
    #coefs_stress = [0.85, 0.92, 1.08, 1.18, 1.25]
    #dxs = [0.01, 0.02, 0.03]

    pwd = os.getcwd()
    os.chdir(folder)
    ref_dics = []

    for ref_structure in strucs:
        #print(ref_structure)
        ref_dics.extend(augment_ref_single(ref_structure,
                                           calculator,
                                           steps,
                                           N_vibs,
                                           n_atoms_per_unit,
                                           logfile,
                                           fmax))

    os.chdir(pwd)
    return ref_dics

def augment_ref_single(ref_structure, calculator, steps, N_vibs, n_atoms_per_unit, logfile='-',
        fmax=0.1, max_E=1000, min_dE=5.0):
    """
    parallel version
    Add max_E and min_dE to prevent adding the high-E structures
    """

    #coefs_stress = [0.90, 0.95, 1.08, 1.15, 1.20]
    #dxs = [0.025, 0.050, 0.075]
    coefs_stress = [0.85, 0.92, 1.10, 1.25]
    dxs = [0.01, 0.02, 0.03]


    ref_dics = []
    print('# Relaxation to get the ground state: 1')
    ref_structure.set_calculator(calculator)
    ref_structure.set_constraint(FixSymmetry(ref_structure))
    ecf = ExpCellFilter(ref_structure)
    dyn = FIRE(ecf, a=0.1, logfile=logfile)
    dyn.run(fmax=fmax, steps=steps)
    ref_structure.set_constraint()

    # reset_lammps_cell and make supercell (QZ......)
    cell = ref_structure.get_cell_lengths_and_angles()[:3]
    supercell = [1, 1, 1]
    for ax in range(3):
        supercell[ax] = int(ceil(6.5/cell[ax])) # to save some time?
    ref_structure *= supercell
    ref_structure = reset_lammps_cell(ref_structure)

    ref_dic = evaluate_ref_single(ref_structure,
                                  calculator,
                                  n_atoms_per_unit,
                                  [True, True, True])
    ref_dic['tag'] = 'minimum'
    ref_eng = ref_dic['energy']/ref_dic['replicate']
    if ref_eng < max_E:
        ref_dics.append(ref_dic)

        print('# Get elastic configurations: 3 * {:d}'.format(len(coefs_stress)))
        cell0 = ref_structure.cell.array
        for ax in range(3):
            for coef in coefs_stress:
                structure = ref_structure.copy()
                cell = cell0.copy()
                cell[ax, ax] *= coef
                structure.set_cell(cell, scale_atoms=True)
                # Add relaxation to improve the energy
                structure.set_calculator(calculator)
                dyn = FIRE(structure, a=0.1, logfile=logfile)
                dyn.run(fmax=fmax, steps=20)
                ref_dic = evaluate_ref_single(structure,
                                          calculator,
                                          n_atoms_per_unit,
                                          [True, False, True])
                if ref_eng - min_dE < ref_dic['energy']/ref_dic['replicate'] < ref_eng + min_dE:
                    ref_dic['tag'] = 'elastic'
                    ref_dics.append(ref_dic)

        print('# Get purturbation: {:d} * {:d}'.format(N_vibs, len(dxs)))
        pos0 = ref_structure.get_positions()
        for dx in dxs:
            for i in range(N_vibs):
                structure = ref_structure.copy()
                pos = pos0.copy()
                pos += np.random.uniform(-dx, dx, size=pos0.shape)
                structure.set_positions(pos)
                ref_dic = evaluate_ref_single(structure,
                                          calculator,
                                          n_atoms_per_unit,
                                          [True, True, False])
                if ref_eng - min_dE < ref_dic['energy']/ref_dic['replicate'] < ref_eng + min_dE:
                    ref_dic['tag'] = 'vibration'
                    ref_dics.append(ref_dic)
    print('# Finalized data augmentation')

    return ref_dics

def add_strucs_par(strs, smiles):
    strucs = []
    for _str in strs:
        try:
            pmg = Structure.from_str(_str, fmt='cif')
            c0 = pyxtal(molecular=True)
            c0.from_seed(pmg, molecules=smiles)
            strucs.append(c0.to_ase(resort=False))
        except:
            print("Skip a structure due to reading error")
            print(_str)
    return strucs



"""
A class to handle the optimization of force field parameters
for molecular simulation.
"""
class ForceFieldParameters:


    def __init__(self,
                 smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O'],
                 style = 'gaff',
                 chargemethod = 'am1bcc',
                 ff_evaluator = 'lammps',
                 ref_evaluator = 'mace',
                 f_coef = 0.1,
                 s_coef = 1.0,
                 ncpu = 1,
                 verbose = True,
                 device = 'cpu'):
        """
        Initialize the parameters

        Args:
            smiles (list): list of smiles strings
            style (str): 'gaff' or 'openff'
            chargemethod (str): 'mmff94', 'am1bcc', 'am1-mulliken', 'gasteiger'
            ff_evaluator (str): 'lammps' or 'charmm'
            ref_evaluator (str): 'mace' or 'trochani'
            f_coef (float): coefficients for forces
            s_coef (float): coefficients for stress
        """
        self.smiles = smiles
        self.ff_style = style
        self.ff = forcefield(smiles, style, chargemethod)
        # only works for 1:1 ratio cocrystal for now
        self.natoms_per_unit = sum([len(mol.atoms) for mol in self.ff.molecules])
        params_init, constraints, bounds = self.get_default_ff_parameters()
        self.params_init = params_init
        self.constraints = constraints
        self.bounds = bounds

        self.parameters_current = []
        self.reference_data = []
        self.ff_evaluator = ff_evaluator
        self.ref_evaluator = ref_evaluator
        if ref_evaluator == 'mace':
            self.calculator = mace_mp(model = "small",
                                      dispersion = True,
                                      default_dtype = "float64",
                                      device = device)
        elif ref_evaluator == 'ani':
            from torchani import models
            self.calculator = models.ANI2x().ase()

        if ff_evaluator == 'lammps':
            # Using one lmp instance may cause long time delay at the end
            #cmdargs = ["-screen", "none", "-log", "lmp.log", "-nocite"]
            #self.lmp = PyLammps(name=None, cmdargs=cmdargs)
            # set up the lammps template
            self.ase_templates = {}
            self.lmp_dat = {}
        self.f_coef = f_coef
        self.s_coef = s_coef
        self.terms = ['bond', 'angle', 'proper', 'vdW', 'charge', 'offset']
        self.ncpu = ncpu
        self.verbose = verbose

    def get_default_ff_parameters(self, coefs=[0.5, 1.5], deltas=[-0.2, 0.2]):
        """
        Get the initial FF parameters/bounds/constraints
        # Loop over molecule
        # Loop over LJ, Bond, Angle, Torsion, Improper
        # Loop over molecule
        # Loop over charges
        """

        params = []
        bounds = []
        constraints = []
        N_bond, N_angle, N_proper, N_improper, N_vdW, N_charges = 0, 0, 0, 0, 0, 0

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
        for molecule in self.ff.molecules:
            for dihedral_type in molecule.dihedral_types:
                    phi_k = dihedral_type.phi_k
                    params.append(phi_k)
                    bounds.append((phi_k * coefs[0], phi_k * coefs[1]))
                    N_proper += 1

        # Improper (phi_k) #  per=2, phase=180.000,  scee=1.200, scnb=2.000>
        #for molecule in self.ff.molecules:
        #for improper_type in ps.improper_periodic_types.keys():

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
            N_charges += len(molecule.atoms)

        # N_LJ, N_bond, N_angle, N_proper, N_improper, N_charges
        self.N_bond = N_bond
        self.N_angle = N_angle
        self.N_proper = N_proper
        self.N_improper = N_improper
        self.N_vdW = N_vdW
        self.N_charges = N_charges
        # This is for the offset
        params.append(0)

        #self._params_init = np.array(params)
        #self._constraints = constraints
        #self._bounds = bounds
        return params, constraints, bounds

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

    def get_sub_parameters(self, parameters, terms):
        """
        Get the subparameters/bonds/constraints for optimization

        Args:
            parameters (list): input complete parameters
            termss (list): selected terms ['vdW', 'bond', .etc]

        Returns:
            sub_paras (list)
            sub_bounds (list)
            sub_constraints (list)
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
                id2 = id1 + self.N_charges
                do_charge = True
            elif term == 'offset':
                id1 = self.N_bond + self.N_angle + self.N_proper + self.N_vdW + self.N_charges
                id2 = id1 + 1

            #if term != 'offset':
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
                id2 = id1 + self.N_charges
                sub_constraints = self.constraints
            elif term == 'offset':
                id1 = self.N_bond + self.N_angle + self.N_proper + self.N_vdW + self.N_charges
                id2 = id1 + 1
            parameters[id1:id2] = sub_para

        return parameters

    #@timeit
    def update_ff_parameters(self, parameters, check=True):
        """
        Update FF parameters in self.ff.molecules
        # Loop over molecule
        # Loop over Bond, Angle, Torsion, Improper, vdW, charges

        """
        assert(len(parameters) == len(self.params_init))
        #if check: parameters = self.check_validity(parameters)
        #parameters = parameters.copy()
        self.ff.update_parameters(parameters)
        # reset the ase_lammps to empty
        self.ase_templates = {}
        self.lmp_dat = {}

    def __str__(self):
        s = "\n------Force Field Parameters------\n"
        s += "Bond:        {:3d}\n".format(self.N_bond)
        s += "Angle:       {:3d}\n".format(self.N_angle)
        s += "Proper:      {:3d}\n".format(self.N_proper)
        s += "Improper:    {:3d}\n".format(self.N_improper)
        s += "vdW:         {:3d}\n".format(self.N_vdW)
        s += "Charges:     {:3d}\n".format(self.N_charges)
        s += "Total:       {:3d}\n".format(len(self.params_init))
        s += "Constraints: {:3d}\n".format(len(self.constraints))
        s += "FF_code:    {:s}\n".format(self.ff_evaluator)
        s += "Ref_code:   {:s}\n".format(self.ref_evaluator)
        s += "N_CPU:       {:3d}\n".format(self.ncpu)
        s += "F_coef:      {:.3f}\n".format(self.f_coef)
        s += "S_coef:      {:.3f}\n".format(self.s_coef)
        return s

    def __repr__(self):
        return str(self)


    def augment_reference(self, ref_structure, fmax=0.1, steps=250, N_vibs=10, logfile='-'):
        """
        Generate more reference data based on input structure, including
        1. Fully optimized structue
        2. elastic strain
        3. atomic perturbation (e.g. 0.2 A)

        Args:
            - ref_structure
            - fmax
            - steps (int)
            - N_vibs (int)

        Returns:
        A list of ref_dics that store the structure/energy/force/stress
        """

        #ref_structure = self.ff.reset_lammps_cell(ref_structure)
        return augment_ref_single(ref_structure,
                                  self.calculator,
                                  steps,
                                  N_vibs,
                                  self.natoms_per_unit,
                                  logfile,
                                  fmax)

    #@timeit
    def evaluate_ref_single(self, structure, options=[True, True, True], relax=False):
        """
        evaluate the reference structure with the ref_evaluator
        """
        return evaluate_ref_single(structure,
                                   self.calculator,
                                   self.natoms_per_unit,
                                   options,
                                   relax)


    def get_lmp_inputs_from_ref_dics(self, ref_dics):
        lmp_strucs, lmp_dats = [], []
        for ref_dic in ref_dics:
            structure = ref_dic['structure']
            lmp_struc, lmp_dat = self.get_lmp_input_from_structure(structure)
            lmp_strucs.append(lmp_struc)
            lmp_dats.append(lmp_dat)
        #print('Final'); print(lmp_strucs[0].box); print(lmp_strucs[-1].box); import sys; sys.exit()
        return lmp_strucs, lmp_dats

    def get_lmp_input_from_structure(self, structure, set_template=True):

        replicate = len(structure)/self.natoms_per_unit
        if replicate in self.ase_templates.keys():
            lmp_struc = self.ase_templates[replicate]
            lmp_dat = self.lmp_dat[replicate]
        else:
            lmp_struc = self.ff.get_ase_lammps(structure)
            dat_head = lmp_struc._write_dat_head()
            dat_prm = lmp_struc._write_dat_parameters()
            dat_connect, _, _, _ = lmp_struc._write_dat_connects()
            lmp_dat = [dat_head, dat_prm, dat_connect]
            if set_template:
                self.lmp_dat[replicate] = lmp_dat
                self.ase_templates[replicate] = lmp_struc
        return lmp_struc, lmp_dat

    #@timeit
    def evaluate_ff_single(self, lmp_struc, options=[True]*3,
                           lmp_dat=None,
                           lmp_in=None,
                           box=None,
                           positions=None,
                           parameters=None):
        """
        evaluate the reference structure with the ff_evaluatort

        Args:
            lmp_struc: ase structure
            options (list): [energy, forces, stress]
            lmp_dat:
            lmp_in:
            box:
            positions:
            parameters:
        """
        if parameters is not None:
            self.update_ff_parameters(parameters)

        if type(lmp_struc) == Atoms:
            self.ase_templates = {}
            self.lmp_dat = {}
            lmp_struc, lmp_dat = self.get_lmp_input_from_structure(lmp_struc)
        if box is not None: lmp_struc.box = box
        if positions is not None: lmp_struc.coordinates = positions

        #structure = lmp_struc.to_ase()
        replicate = len(lmp_struc.atoms)/self.natoms_per_unit
        ff_dic = {#'structure': lmp_struc.to_ase(),
                  'energy': None,
                  'forces': None,
                  'stress': None,
                  'replicate': replicate,
                  'options': options,
                  }

        eng, force, stress = get_lmp_efs(lmp_struc, lmp_in, lmp_dat)
        if options[0]: # Energy
            ff_dic['energy'] = eng
        if options[1]: # forces
            ff_dic['forces'] = force
        if options[2]:
            ff_dic['stress'] = stress
        #print(eng); import sys; sys.exit()
        return ff_dic

    def same_lmp(self, struc1, struc2):
        """
        quick comparison for two lmp structures
        """
        for i in range(len(struc1.dihedrals)):
            d1 = struc1.dihedrals[i]
            d2 = struc2.dihedrals[i]
            id1 = [d1.atom1.idx, d1.atom2.idx, d1.atom3.idx, d1.atom4.idx]
            id2 = [d2.atom1.idx, d2.atom2.idx, d2.atom3.idx, d2.atom4.idx]
            if id1 != id2:
                print("Different structures were found")
                struc1.to_ase('1.xyz', format='1.xyz')
                struc2.to_ase('2.xyz', format='2.xyz')
                return False
        return True


    #@timeit
    def get_objective(self, ref_dics, e_offset, E_only=False, lmp_in=None, obj='MSE'):
        """
        Compute the objective mismatch for the give ref_dics
        Todo, Enable the parallel option

        Args:
            ref_dics:
            e_offset:
            E_only:
            lmp_in:
            obj:
        """

        total_obj = 0
        eng_arr, force_arr, stress_arr = [[], []], [[], []], [[], []]

        lmp_strucs, lmp_dats = self.get_lmp_inputs_from_ref_dics(ref_dics)

        if self.ncpu == 1:
            for i, ref_dic in enumerate(ref_dics):
                options = ref_dic['options']
                ff_dic = self.evaluate_ff_single(lmp_strucs[i], options,
                                                 lmp_dats[i], lmp_in,
                                                 )
                #total_obj += self.obj_from_ffdic(ff_dic, ref_dic, e_offset, E_only)
                efs = (ff_dic['energy'], ff_dic['forces'], ff_dic['stress'])
                res = obj_from_efs(efs,
                                   ref_dic,
                                   e_offset,
                                   E_only,
                                   self.f_coef,
                                   self.s_coef,
                                   obj,
                                  )
                if obj == 'MSE':
                    total_obj += res
                else:
                    ([e1, e2], [f1, f2], [s1, s2]) = res
                    eng_arr[0].append(e1)
                    eng_arr[1].append(e2)
                    force_arr[0].extend(f1)
                    force_arr[1].extend(f2)
                    stress_arr[0].extend(s1)
                    stress_arr[1].extend(s1)
        else:
            #parallel process
            N_cycle = int(np.ceil(len(ref_dics)/self.ncpu))
            args_list = []
            for i in range(self.ncpu):
                folder = self.get_label(i)
                id1 = i*N_cycle
                id2 = min([id1+N_cycle, len(ref_dics)])
                #print(i, id1, id2, len(ref_dics))
                os.makedirs(folder, exist_ok=True)
                args_list.append((ref_dics[id1:id2],
                                  lmp_strucs[id1:id2],
                                  lmp_dats[id1:id2],
                                  lmp_in,
                                  e_offset,
                                  E_only,
                                  self.natoms_per_unit,
                                  self.f_coef,
                                  self.s_coef,
                                  folder,
                                  obj,
                                  ))

            with ProcessPoolExecutor(max_workers=self.ncpu) as executor:
                results = [executor.submit(evaluate_ff_par, *p) for p in args_list]
                for result in results:
                    if obj == 'MSE':
                        total_obj += result.result()
                        #print(result.result())
                    else:
                        #([e1, e2], [f1, f2], [s1, s2]) = result.result()
                        (engs, forces, stresses) = result.result()
                        eng_arr[0].extend(engs[0])
                        eng_arr[1].extend(engs[1])
                        force_arr[0].extend(forces[0])
                        force_arr[1].extend(forces[1])
                        stress_arr[0].extend(stresses[0])
                        stress_arr[1].extend(stresses[1])
            if obj == 'R2':
                #print(eng_arr[0])
                total_obj -= compute_r2(eng_arr[0], eng_arr[1])
                total_obj -= self.f_coef * compute_r2(force_arr[0], force_arr[1])
                total_obj -= self.s_coef * compute_r2(stress_arr[0], stress_arr[1])
                #print('BBBBBBBBBBBb', self.f_coef, compute_r2(force_arr[0], force_arr[1]))

        return total_obj


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

    #@timeit
    def optimize_init(self, ref_dics, opt_dict, parameters0=None, obj='MSE'):

        if parameters0 is None:
            #parameters0 = self.params_init.copy()
            offset, parameters0 = self.optimize_offset(ref_dics)
            #print(parameters0); import sys; sys.exit()
        else:
            assert(len(parameters0) == len(self.params_init))
        self.update_ff_parameters(parameters0)
        #self.ff.set_lammps_in('lmp.in')

        terms = list(opt_dict.keys())
        # Move the charge term to the end
        if 'charge' in terms:
            terms.pop(terms.index('charge'))
            terms.append('charge')
            charges = opt_dict['charge']
        else:
            charges = None

        # Set up the input for optimization, including x, bounds, args
        x = []
        ids = []

        e_offset = parameters0[-1]
        for term in terms:
            if len(ids) > 0:
                if term != 'charge':
                    x.extend(opt_dict[term])
                    ids.append(ids[-1] + len(opt_dict[term]))
                else:
                    x.extend([1.0]) #TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
            else:
                if term != 'charge':
                    x.extend(opt_dict[term])
                    ids.append(len(opt_dict[term]))
                else:
                    x.extend([1.0]) #TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

        #print("Starting", x)
        #x = [item for sublist in values for item in sublist]
        _, sub_bounds, _ = self.get_sub_parameters(parameters0, terms)
        bounds = [item for sublist in sub_bounds for item in sublist]

        def obj_fun(x, ref_dics, parameters0, e_offset, ids, obj, charges=None):
            """
            Split the x into list
            """
            values = []
            for i in range(len(ids)):
                if i == 0:
                    id1 = 0
                else:
                    id1 = ids[i-1]
                values.append(x[id1:ids[i]])

            # The last x value is the ratio of charge
            #print(charges, x[-1])
            if charges is not None:
                values.append(x[-1] * charges)
            #print(terms, values)
            parameters = self.set_sub_parameters(values, terms, parameters0)
            self.update_ff_parameters(parameters)
            # Reset the lmp.in file
            lmp_in = self.ff.get_lammps_in()
            objective = self.get_objective(ref_dics, e_offset, lmp_in=lmp_in, obj=obj)
            #print("Debugging", values[0][:5], objective)
            return objective

        # set call back function for debugging
        def objective_function_wrapper(x, ref_dics, parameters0, e_offset, obj, ids, charges):
            global last_function_value
            last_function_value = obj_fun(x, ref_dics, parameters0, e_offset, obj, ids, charges)
            return last_function_value

        arg_lists = (ref_dics, parameters0, e_offset, ids, obj, charges)
        # Actual optimization
        print("Init obj", objective_function_wrapper(x, *arg_lists))#; import sys; sys.exit()

        return x, bounds, objective_function_wrapper, arg_lists

    def optimize_post(self, x, ids, charges):
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
            values.append(x[-1]*charges)

        return values


    def optimize_global(self, ref_dics, opt_dict, parameters0=None, steps=100, obj='MSE', t0=100, alpha=0.99):
        """
        FF parameters' optimization using the simulated annealing algorithm
        Todo, test new interface, add temp scheduling

        Args:
            ref_dics (dict): reference data dictionary
            opt_dict (dict): optimization terms and values
            parameters0 (array): initial full parameters
            steps (int): optimization steps
            obj (str): 'MSE' or 'R2'
            t0 (float): initial temp
            alpha (float): cooling rate
        Returns:
            The optimized values
        """
        x, bounds, obj_fun, fun_args = self.optimize_init(ref_dics, opt_dict, parameters0, obj)
        t = t0
        current_x = x
        current_fun = obj_fun(current_x, *fun_args)
        best_x, best_fun = current_x, current_fun

        for i in range(steps):
            # Generate a candidate solution
            candidate_x = current_x.copy()
            for j in range(len(bounds)):
                lb, ub = bounds[j][0], bounds[j][1]
                candidate_x[j] += 0.02 * np.random.uniform(-1, 1) * (lb-ub)
                if candidate_x[j] > ub:
                    candidate_x[j] = ub
                elif candidate_x[j] < lb:
                    candidate_x[j] = lb
            candidate_fun = obj_fun(candidate_x, *fun_args)

            # Update best fun if necessary
            if candidate_fun < best_fun:
                best_x, best_fun = candidate_x.copy(), deepcopy(candidate_fun)

            # Accept the solution with probability
            if np.random.random() < np.exp((current_fun - candidate_fun)/t):
                current_x, current_fun = candidate_x, candidate_fun

            t *= alpha
            if self.verbose and i % 10 == 0:
                print("Step {:4d} {:5.2f} {:.4f} {:.4f}".format(i, t, candidate_fun, current_fun))#, current_x)
        print("Best results after {:d} steps: {:.4f}".format(steps, best_fun))
        #print("Best fun", obj_fun(best_x, *fun_args))#; import sys; sys.exit()

        values = self.optimize_post(best_x, fun_args[-3], fun_args[-1])
        #print(values)

        return best_x, best_fun, values, steps


    def optimize_local(self, ref_dics, opt_dict, parameters0=None, steps=100, obj='MSE'):
        """
        FF parameters' local optimization using the Nelder-Mead algorithm

        Args:
            ref_dics (dict): reference data dictionary
            opt_dict (dict): optimization terms and values
            parameters0 (array): initial full parameters
            steps (int): optimization steps
        Returns:
            The optimized values
        """
        x, bounds, obj_fun, fun_args = self.optimize_init(ref_dics, opt_dict, parameters0, obj)

        #def my_callback(xk):
        #    print(f"Solution: {xk[:2]}, Objective: {last_function_value}")
        class CallbackFunction:
            def __init__(self):
                self.iteration = 0  # Initialize iteration count

            def callback(self, xk):
                self.iteration += 1  # Increment iteration count
                if self.iteration % 10 == 0:  # Check if it's a multiple of 10
                    print("Step {:4d} {:.4f}".format(self.iteration, last_function_value))

        callback = CallbackFunction() if self.verbose else None

        res = minimize(#obj_fun,
                       obj_fun, #objective_function_wrapper,
                       x,
                       method = 'Nelder-Mead',
                       args = fun_args, #(ref_dics, parameters0, e_offset, ids, charges),
                       options = {'maxiter': steps, 'disp': True},
                       bounds = bounds,
                       callback = callback.callback,
                       )

        # Rearrange the optimized parameters to the list of values
        values = self.optimize_post(res.x, fun_args[-3], fun_args[-1])
        print("Final Obj", res.fun)
        return res.x, res.fun, values, res.nfev

    def optimize_offset(self, ref_dics, parameters0=None, steps=50):
        """
        Approximate the offset energy between FF and Reference evaluators
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

        results = self.evaluate_multi_references(ref_dics, parameters0, max_E=1000, max_dE=1000)
        (ff_values, ref_values, _, _) = results
        (ff_eng, _, _) = ff_values
        (ref_eng, _, _) = ref_values

        x = parameters0[-1]
        if abs(x) < 1e-5:
            x = np.mean(ref_eng - ff_eng)
            print("Initial guess of offset", x)
            #print("ref_eng", ref_eng)
            #print("ff_eng", ff_eng)

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
        return res.x[0], parameters0


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
        if parameters is None:
            parameters = self.params_init.copy()
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
        for i, ref_dic in enumerate(ref_dics):
            self.ase_templates = {}
            self.lmp_dat = {}
            ff_dic = self.evaluate_ff_single(ref_dic['structure'])
            e1 = ff_dic['energy']/ff_dic['replicate'] + parameters[-1]
            e2 = ref_dic['energy']/ff_dic['replicate']
            if abs(e1-e2) < dE:
                add = True
                if ref_dic['options'][1]:
                    f1 = ff_dic['forces'].flatten()
                    f2 = ref_dic['forces'].flatten()
                    rmse = np.sum((f1-f2)**2)/len(f2)
                    if rmse > FMSE:
                        add = False
                if add and ref_dic['options'][2]:
                    s1 = ff_dic['stress']
                    s2 = ref_dic['stress']
                    rmse = np.sum((s1-s2)**2)/len(s2)
                    if rmse > SMSE:
                        add = False
                if add:
                    _ref_dics.append(ref_dic)

        print("Removed {:d} entries by error".format(len(ref_dics)-len(_ref_dics)))
        return _ref_dics

    def cut_references(self, ref_dics, cutoff):
        """
        Cut the list of references by energy
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


    def load_references(self, filename, reset_cell=False):
        """
        Load the reference information

        Args:
            - filename (str): path of reference file
            - reset_cell (bool): whether or not reset the cell

        Returns:
            the list of reference dictionaries
        """
        ref_dics = []
        if filename.endswith(('.xml', '.db')):
            # Load reference data from file
            if filename.endswith('.xml'):
                dics = xml_to_dict_list(filename)
                for dic in dics:
                    structure = Atoms(numbers = dic['numbers'],
                                      positions = dic['position'],
                                      cell = dic['lattice'],
                                      pbc = [1, 1, 1])
                    if reset_cell:
                        structure = reset_lammps_cell(structure)
                    dic0 = {
                            'structure': structure,
                            'energy': dic['energy'],
                            'forces': dic['forces'],
                            'stress': dic['stress'],
                            'replicate': dic['replicate'],
                            'options': dic['options'],
                            'tag': dic['tag'],
                           }
                    ref_dics.append(dic0)
            else:
                pass
        else:
            raise ValueError("Unsupported file format")

        return ref_dics

    def export_references(self, ref_dics, filename='reference.xml'):
        """
        export the reference configurations to xml or ase.db

        Args:
            - ref_dics: list of reference configuration in dict format
            - filename: filename
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

    def get_label(self, i):
        if i < 10:
            folder = f"cpu00{i}"
        elif i < 100:
            folder = f"cpu0{i}"
        else:
            folder = f"cpu0{i}"
        return folder

    def evaluate_single_reference(self, ref_dic, parameters):

        f_mse, f_r2, s_mse, s_r2 = 0, 0, 0, 0
        self.update_ff_parameters(parameters)
        offset_opt = parameters[-1]
        structure, options = ref_dic['structure'], ref_dic['options']

        ff_dic = self.evaluate_ff_single(structure, options, None)
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

        if self.ncpu == 1:
            for i, ref_dic in enumerate(ref_dics):
                structure, options = ref_dic['structure'], ref_dic['options']
                #print(lmp_strucs[i].box)
                structure = reset_lammps_cell(structure)
                box = structure.cell.cellpar()
                coordinates = structure.get_positions()

                ff_dic = self.evaluate_ff_single(lmp_strucs[i], options, lmp_dats[i], None, box, coordinates)
                e1 = ff_dic['energy']/ff_dic['replicate']
                e2 = ref_dic['energy']/ff_dic['replicate']
                de = abs(e1 + offset_opt - e2)
                if e1 < max_E and de < max_dE:
                    ff_eng.append(e1 + offset_opt)
                    ref_eng.append(e2)
                    if ref_dic['options'][1]:
                        ff_force.extend(ff_dic['forces'].tolist())
                        ref_force.extend(ref_dic['forces'].tolist())
                    if ref_dic['options'][2]:
                        ff_stress.extend(ff_dic['stress'].tolist())
                        ref_stress.extend(ref_dic['stress'].tolist())
        else:
            #parallel process
            N_cycle = int(np.ceil(len(ref_dics)/self.ncpu))
            #for cycle in range(N_cycle):
            args_list = []
            for i in range(self.ncpu):
                folder = self.get_label(i)
                id1 = i * N_cycle
                id2 = min([id1+N_cycle, len(ref_dics)])
                #print(i, id1, id2, len(ref_dics))
                os.makedirs(folder, exist_ok=True)
                args_list.append((ref_dics[id1:id2],
                                  lmp_strucs[id1:id2],
                                  lmp_dats[id1:id2],
                                  lmp_in,
                                  offset_opt,
                                  self.natoms_per_unit,
                                  self.f_coef,
                                  self.s_coef,
                                  folder,
                                  max_E,
                                  max_dE))

            with ProcessPoolExecutor(max_workers=self.ncpu) as executor:
                results = [executor.submit(evaluate_ff_error_par, *p) for p in args_list]
                for result in results:
                    res = result.result()
                    ff_eng.extend(res[0])
                    if len(res[1]) > 0: ff_force.extend(res[1])
                    if len(res[2]) > 0: ff_stress.extend(res[2])
                    ref_eng.extend(res[3])
                    if len(res[4]) > 0: ref_force.extend(res[4])
                    if len(res[5]) > 0: ref_stress.extend(res[5])

        ff_eng = np.array(ff_eng).flatten()
        ff_force = np.array(ff_force).flatten()
        ff_stress = np.array(ff_stress)

        ref_eng = np.array(ref_eng).flatten()
        ref_force = np.array(ref_force).flatten()
        ref_stress = np.array(ref_stress)

        mse_eng = np.sqrt(np.mean((ff_eng-ref_eng)**2))
        mse_for = np.sqrt(np.mean((ff_force-ref_force)**2))
        mse_str = np.sqrt(np.mean((ff_stress-ref_stress)**2))
        #print(ff_eng, ref_eng)
        r2_eng = compute_r2(ff_eng, ref_eng)
        r2_for = compute_r2(ff_force, ref_force)
        r2_str = compute_r2(ff_stress, ref_stress)

        ff_values = (ff_eng, ff_force, ff_stress)
        ref_values = (ref_eng, ref_force, ref_stress)
        rmse_values = (mse_eng, mse_for, mse_str)
        r2_values = (r2_eng, r2_for, r2_str)

        return ff_values, ref_values, rmse_values, r2_values


    def add_multi_references(self, strucs, augment=True, steps=120, N_vibs=3, logfile='-'):
        """
        Add multiple references to training

        Args:
            - strucs (list): list of ase strucs with the desired atomic orders
            - augment (bool):
            - steps (int):
            - N_vibs (int):

        Returns:
            list of reference dics
        """
        ref_dics = []

        if not augment:
            if self.ncpu == 1:
                for struc in strucs:
                    ref_structure = reset_lammps_cell(struc)
                    ref_dic = evaluate_ref_single(ref_structure,
                                              self.calculator,
                                              self.natoms_per_unit,
                                              [True, True, True])
                    ref_dic['tag'] = 'CSP'
                    ref_dics.append(ref_dic)
            else:
                N_cycle = int(np.ceil(len(strucs)/self.ncpu))
                args_list = []
                for i in range(self.ncpu):
                    folder = self.get_label(i)
                    id1 = i*N_cycle
                    id2 = min([id1+N_cycle, len(strucs)])
                    os.makedirs(folder, exist_ok=True)
                    print("# parallel process", N_cycle, id1, id2)
                    args_list.append((strucs[id1:id2],
                                      self.calculator,
                                      self.natoms_per_unit,
                                      [True, True, True]))

                with ProcessPoolExecutor(max_workers=self.ncpu) as executor:
                    results = [executor.submit(evaluate_ref_par, *p) for p in args_list]
                    for result in results:
                        res = result.result()
                        ref_dics.extend(res)

        # augment structures is more expensive
        else:
            if self.ncpu == 1:
                for struc in strucs:
                    dics = self.augment_reference(struc,
                                                  steps=steps,
                                                  N_vibs=N_vibs,
                                                  logfile=logfile)
                    ref_dics.extend(dics)

            else:
                N_cycle = int(np.ceil(len(strucs)/self.ncpu))
                args_list = []
                for i in range(self.ncpu):
                    folder = self.get_label(i)
                    id1 = i*N_cycle
                    id2 = min([id1+N_cycle, len(strucs)])
                    os.makedirs(folder, exist_ok=True)
                    print("# parallel process", N_cycle, id1, id2)
                    args_list.append((strucs[id1:id2],
                                      self.calculator,
                                      steps,
                                      N_vibs,
                                      self.natoms_per_unit,
                                      folder,
                                      logfile))

                with ProcessPoolExecutor(max_workers=self.ncpu) as executor:
                    results = [executor.submit(augment_ref_par, *p) for p in args_list]
                    for result in results:
                        res = result.result()
                        ref_dics.extend(res)

        return ref_dics


    def add_multi_references_from_cif(self, cif, N_max=10, augment=True, steps=120, N_vibs=3):
        """
        Add multiple references to training

        Args:
            - cif (str): cif file containg mutliple structures
            - N_max (int):
            - augment (bool):
            - steps (int):
            - N_vibs (int):

        Returns:
            list of reference dics
        """
        from pyxtal.util import parse_cif
        strs, engs = parse_cif(cif, eng=True)
        N_max = min([N_max, len(strs)])
        ids = np.argsort(engs)[:N_max]
        strs = [strs[id] for id in ids if engs[id] < 1000] # sort by eng
        smiles = [smi+'.smi' for smi in self.ff.smiles]
        strucs = []

        if self.ncpu == 1:
            for i, id in enumerate(ids):
                pmg = Structure.from_str(strs[id], fmt='cif')
                c0 = pyxtal(molecular=True)
                c0.from_seed(pmg, molecules=smiles)
                strucs.append(c0.to_ase(resort=False))
        else:
            N_cycle = int(np.ceil(len(strs)/self.ncpu))
            args_list = []
            for i in range(self.ncpu):
                id1 = i*N_cycle
                id2 = min([id1+N_cycle, len(strs)])
                print("# parallel process", N_cycle, id1, id2)
                args_list.append((strs[id1:id2], smiles))

            with ProcessPoolExecutor(max_workers=self.ncpu) as executor:
                results = [executor.submit(add_strucs_par, *p) for p in args_list]
                for result in results:
                    res = result.result()
                    if len(res) > 0:
                        strucs.extend(res)

        return self.add_multi_references(strucs, augment, steps, N_vibs)


    def _plot_ff_parameters(self, ax, params, term='bond-1', width=0.35):
        """
        plot the individual parameters in bar plot style

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
        plot the whole FF parameters

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

    def plot_ff_results(self, figname, ref_dics, params, labels=None,
            max_E=1000, max_dE=1000):
        """
        plot the ff performance results

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
            return [err_dic]
        else:
            if labels is None: labels = ['FF'+str(i) for i in range(len(params))]
            fig, axes = plt.subplots(len(params), 3, figsize=(16, 4*len(params)))
            err_dics = []
            for i, param in enumerate(params):
                _, err_dic = self._plot_ff_results(axes[i], param, ref_dics, labels[i], max_E, max_dE)
                err_dics.append(err_dic)
            plt.savefig(figname)
            return err_dics


    def _plot_ff_results(self, axes, parameters, ref_dics, label,
            max_E=1000, max_dE=1000, size=None):
        """
        Plot the results of FF prediction as compared to the references in
        terms of Energy, Force and Stress values.
        Args:
            axes (list): list of matplotlib axes
            parameters (1D array): array of full FF parameters
            ref_dics (dict): reference data
            offset_opt (float): offset values for energy prediction
            label (str):
        """

        # Set up the ff engine
        self.update_ff_parameters(parameters)

        results = self.evaluate_multi_references(ref_dics, parameters, max_E, max_dE)
        (ff_values, ref_values, rmse_values, r2_values) = results
        (ff_eng, ff_force, ff_stress) = ff_values
        (ref_eng, ref_force, ref_stress) = ref_values
        (mse_eng, mse_for, mse_str) = rmse_values
        (r2_eng, r2_for, r2_str) = r2_values
        print(r2_values)

        label1 = '{:s}. Energy ({:d})\n'.format(label, len(ff_eng))
        label1 += 'Unit: [eV/mole]\n'
        label1 += 'RMSE: {:.4f}\n'.format(mse_eng)
        label1 += 'R2:   {:.4f}'.format(r2_eng)

        label2 = '{:s}. Forces ({:d})\n'.format(label, len(ff_force))
        label2 += 'Unit: [eV/A]\n'
        label2 += 'RMSE: {:.4f}\n'.format(mse_for)
        label2 += 'R2:   {:.4f}'.format(r2_for)

        label3 = '{:s}. Stress ({:d})\n'.format(label, len(ff_stress))
        label3 += 'Unit: [GPa]\n'
        label3 += 'RMSE: {:.4f}\n'.format(mse_str)
        label3 += 'R2:   {:.4f}'.format(r2_str)

        print('\n', label1)
        print('\n', label2)
        print('\n', label3)
        print('\nMin_values: {:.4f} {:.4f}'.format(ff_eng.min(),
            ref_eng.min()))
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


    def generate_report(self, ref_dics, parameters):
        """
        run quick report about the performance of each reference structure

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

            ff_dic = self.evaluate_ff_single(ref_dic['structure'])
            e1 = ff_dic['energy']/ff_dic['replicate'] + parameters[-1]
            e2 = ref_dic['energy']/ff_dic['replicate']
            print('\nStructure {:3d}'.format(i))
            print('Energy_ff_ref: {:8.3f} {:8.3f} {:8.3f}'.format(e1, e2, e1-e2))

            if ref_dic['options'][1]:
                f1 = ff_dic['forces'].flatten()
                f2 = ref_dic['forces'].flatten()
                rmse = np.sum((f1-f2)**2)/len(f2)
                r2 = compute_r2(f1, f2)
                print('Forces-R2-MSE: {:8.3f} {:8.3f}'.format(r2, rmse))

            if ref_dic['options'][2]:
                s1 = ff_dic['stress']
                s2 = ref_dic['stress']
                rmse = np.sum((s1-s2)**2)/len(s2)
                r2 = compute_r2(s1, s2)
                print('Stress_ff    : {:8.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}'.format(*s1))
                print('Stress_ref   : {:8.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}{:9.3f}'.format(*s2))
                print('Stress-R2-MSE: {:8.5f} {:8.5f}'.format(r2, rmse))


    def get_ase_charmm(self, params):
        """
        prepare the charmm input files with the updated params.

        Args:
            params: FF parameters array

        Returns:
            ase_atoms object with the charmm ff information
        """

        self.ff.update_parameters(params)
        n_mols = [1] * len(self.ff.smiles)
        if sum(n_mols) == 1:
            pd_struc = self.ff.molecules[0].copy(cls=ParmEdStructure)
        else:
            from functools import reduce
            from operator import add
            mols = []
            for i, m in enumerate(n_mols):
                mols += [ff.molecules[i] * m]
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
        mols = [smi+'.smi' for smi in self.smiles]
        _ref_dics = []
        for i, ref_dic in enumerate(ref_dics):
            c = pyxtal(molecular=True)
            pmg = ase2pymatgen(ref_dic['structure'])
            try:
                c.from_seed(pmg, molecules=mols)
                if c.check_short_distances_by_dict(criteria) == 0:
                    _ref_dics.append(ref_dic)
            except:
                print(i, "Unable to convert to pyxtal", ref_dic['tag'])
        print("Removed {:d} entries by geometry".format(len(ref_dics)-len(_ref_dics)))
        return _ref_dics

if __name__ == "__main__":
    from pyxtal.db import database

    # db = database('../HT-OCSP/benchmarks/Si.db')
    db = database("../HT-OCSP/benchmarks/test.db")
    xtal = db.get_pyxtal("ACSALA")
    smiles = [mol.smile for mol in xtal.molecules]
    assert smiles[0] is not None
    params = ForceFieldParameters(smiles)
    print(params)
    params0 = params.params_init.copy()
    ase_with_ff = params.get_ase_charmm(params0)
    ase_with_ff.write_charmmfiles(base='pyxtal')
    #params.evaluate_ff_single(xtal.to_ase(resort=False))
