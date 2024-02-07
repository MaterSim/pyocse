import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from scipy.optimize import minimize

from ase import Atoms
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry

from ost.forcefield import forcefield
from ost.lmp import LAMMPSCalculator
from mace.calculators import mace_mp
from lammps import PyLammps  # , get_thermo_data

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
            if key in ['lattice', 'position', 'forces', 'numbers',
                       'bond', 'angle', 'proper', 'vdW', 'charge']:

                if text != 'None':
                    #print(text)
                    value = string_to_array(text)
                    #if key == 'position': print(value, value.shape)
                    if key == 'numbers': print(value, value.shape)
                else:
                    value = None
            else:
                # Attempt to convert numeric values back to float/int, fall back to string
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

        self.ff = forcefield(smiles, style, chargemethod)
        # only works for 1:1 ratio cocrystal for now
        self.natoms_per_unit = sum([len(mol.atoms) for mol in self.ff.molecules])
        params_init, constraints, bounds = self.get_default_ff_parameters()
        self._params_init = params_init
        self._constraints = constraints
        self._bounds = bounds

        self.parameters_current = []
        self.reference_data = []
        self.ff_evaluator = ff_evaluator
        self.ref_evaluator = ref_evaluator
        if ref_evaluator == 'mace':
            self.calculator = mace_mp(model = "small",
                                      dispersion = True,
                                      default_dtype = "float64",
                                      device = device)
        if ff_evaluator == 'lammps':
            cmdargs = ["-screen", "none", "-log", "lmp.log", "-nocite"]
            self.lmp = PyLammps(name=None, cmdargs=cmdargs)
        self.f_coef = f_coef
        self.s_coef = s_coef
        self.terms = ['bond', 'angle', 'proper', 'vdW', 'charge']

    def get_default_ff_parameters(self, coefs=[0.8, 1.2], deltas=[-0.2, 0.2]):
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

        #self._params_init = np.array(params)
        #self._constraints = constraints
        #self._bounds = bounds
        return params, constraints, bounds

    @property
    def params_init(self):
        return self._params_init

    @property
    def constraints(self):
        return self._constraints

    @property
    def bounds(self):
        return self._bounds

    def check_validity(self, parameters):
        """
        Check if the input FF parameters are within the bound
        and satisfy the constaint
        """
        for i, _parameter in enumerate(parameters):
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
            parameters[id1:id2] = sub_para

        return parameters


    def update_ff_parameters(self, parameters, check=True):
        """
        Update FF parameters in self.ff.molecules
        # Loop over molecule
        # Loop over Bond, Angle, Torsion, Improper, vdW, charges

        """
        assert(len(parameters) == len(self.params_init))
        if check: parameters = self.check_validity(parameters)
        parameters = parameters.copy()
        count = 0
        # Bond (k, req)
        for molecule in self.ff.molecules:
            for bond_type in molecule.bond_types: #.keys():
                k = parameters[count]
                req = parameters[count + 1]
                bond_type.k = k
                bond_type.req = req
                count += 2
        # Angle (k, theteq)
        for molecule in self.ff.molecules:
            for angle_type in molecule.angle_types: #.keys():
                k = parameters[count]
                theteq = parameters[count + 1]
                angle_type.k = k
                angle_type.theteq = theteq
                count += 2

        # Proper (phi_k) # per=2, phase=180.000,  scee=1.200, scnb=2.000
        for molecule in self.ff.molecules:
            for dihedral_type in molecule.dihedral_types:
                phi_k = parameters[count]
                dihedral_type.phi_k = phi_k
                count += 1

        # nonbond vdW parameters (rmin, epsilon)
        for molecule in self.ff.molecules:
            ps = molecule.get_parameterset_with_resname_as_prefix()
            for key in ps.atom_types.keys():
                rmin = parameters[count]
                epsilon = parameters[count + 1]
                count += 2
                # https://parmed.github.io/ParmEd/html/_modules/parmed/parameters.html#ParameterSet.from_structure
                for at in molecule.atoms:
                    label = at.residue.name + at.type
                    if label == key:
                        at.atom_type.rmin = rmin
                        at.atom_type.rmin_14 = rmin
                        at.atom_type.epsilon = epsilon
                        at.atom_type.epsilon_14 = epsilon
                        break
                #at.epsilon = epsilon
                #at.rmin = rmin

        # nonbond charges
        #print('updating', parameters[count:])
        for molecule in self.ff.molecules:
            for at in molecule.atoms:
                chg = parameters[count]
                at.charge = chg
                count += 1
        sub_params, _, _ = self.get_sub_parameters(self.params_init, ['vdW'])
        #if sub_params[0][0] != 1.9080000000152688:
        #    print('Debug', sub_params); import sys; sys.exit()

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
        return s

    def __repr__(self):
        return str(self)

    def augment_ref_configurations(self, ref_structure, fmax=0.1, steps=250, N_vibration=10):
        """
        Generate more reference data based on input structure, including
        1. Fully optimized structue
        2. elastic strain
        3. atomic perturbation (e.g. 0.2 A)

        Args:
        - ref_structure

        Returns:
        A list of ref_dics that store the structure/energy/force/stress
        """

        ref_dics = []
        coefs_stress = [0.85, 0.92, 1.08, 1.15, 1.20]
        dxs = [0.05, 0.1, 0.15]

        print('# Relaxation to get the ground state: 1')
        ref_structure.set_calculator(self.calculator)
        ref_structure.set_constraint(FixSymmetry(ref_structure))
        ecf = ExpCellFilter(ref_structure)
        dyn = FIRE(ecf, a=0.1)
        dyn.run(fmax=fmax, steps=steps)
        ref_structure.set_constraint()
        ref_dic = self.evaluate_ref_single(ref_structure, options=[True, True, True])
        ref_dic['tag'] = 'minimum'
        ref_dics.append(ref_dic)

        print('# Get the elastic configurations: 3 * {:d}'.format(len(coefs_stress)))
        cell0 = ref_structure.cell.array
        for ax in range(3):
            for coef in coefs_stress:
                structure = ref_structure.copy()
                cell = cell0.copy()
                cell[ax, ax] *= coef
                structure.set_cell(cell, scale_atoms=True)
                ref_dic = self.evaluate_ref_single(structure, options=[True, False, True])
                ref_dic['tag'] = 'elastic'
                ref_dics.append(ref_dic)

        print('# Get the atomic purturbation: {:d} * {:d}'.format(N_vibration, len(dxs)))
        pos0 = ref_structure.get_positions()
        for dx in dxs:
            for i in range(N_vibration):
                structure = ref_structure.copy()
                pos = pos0.copy()
                pos += np.random.uniform(-dx, dx, size=pos0.shape)
                structure.set_positions(pos)
                ref_dic = self.evaluate_ref_single(structure, options=[True, True, False])
                ref_dic['tag'] = 'vibration'
                ref_dics.append(ref_dic)
        return ref_dics

    def evaluate_ref_single(self, structure, options=[True, True, True]):
        """
        evaluate the reference structure with the ref_evaluator
        """
        ref_dic = {'structure': structure,
                   'energy': None,
                   'forces': None,
                   'stress': None,
                   'replicate': len(structure)/self.natoms_per_unit,
                   'options': options,
                  }
        structure.set_calculator(self.calculator)
        if options[0]: # Energy
            ref_dic['energy'] = structure.get_potential_energy()
        if options[1]: # forces
            ref_dic['forces'] = structure.get_forces()
        if options[2]:
            ref_dic['stress'] = structure.get_stress()
        structure.set_calculator() # reset calculator to None
        return ref_dic

    def evaluate_ff_single(self, structure, options=[True, True, True]):
        """
        evaluate the reference structure with the ff_evaluatort

        Args:
        - struc: ase structure
        - options (list): [energy, forces, stress]
        """
        ff_dic = {'structure': structure,
                  'energy': None,
                  'forces': None,
                  'stress': None,
                  'replicate': len(structure)/self.natoms_per_unit,
                  'options': options,
                  }
        if self.ff_evaluator == 'lammps':
            lmp_struc = self.ff.get_ase_lammps(structure)
            calc = LAMMPSCalculator(lmp_struc, lmp_instance=self.lmp)
            eng, force, stress = calc.express_evaluation()
            if options[0]: # Energy
                ff_dic['energy'] = eng
            if options[1]: # forces
                ff_dic['forces'] = force
            if options[2]:
                ff_dic['stress'] = stress
        return ff_dic


    def get_objective(self, ref_dics, e_offset=0, E_only=False):
        """
        Compute the objective mismatch for the give ref_dics
        """
        total_obj = 0
        for ref_dic in ref_dics:
            struc, options = ref_dic['structure'], ref_dic['options']
            ff_dic = self.evaluate_ff_single(struc, options)
            if options[0]:
                e_diff = ff_dic['energy'] - ref_dic['energy']
                e_diff += ref_dic['replicate'] * e_offset
                total_obj += e_diff ** 2
            if not E_only and options[1]:
                f_diff = ff_dic['forces'] - ref_dic['forces']
                total_obj += self.f_coef * np.sum(f_diff ** 2)
            if not E_only and options[2]:
                s_diff = ff_dic['stress'] - ref_dic['stress']
                total_obj += self.s_coef * np.sum(s_diff ** 2)
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


    def optimize_offset(self, ref_dics, e_offset=0, steps=100):
        """
        Approximate the offset of energy between FF and Reference evaluators

        Args:
            ref_dics (dict): reference data dictionary
            e_offset (float): energy offset
            steps (int): optimization steps

        Returns:
            The optimized e_offset value
        """
        x = [e_offset]
        def fun(x, ref_dics):
            return self.get_objective(ref_dics, x[0], E_only=True)
        res = minimize(fun, x, args=(ref_dics), options={'maxiter': steps})
        print("Optimized offset", res.x[0])
        return res.x[0]

    def optimize(self, ref_dics, e_offset, opt_dict, parameters0=None, steps=100, debug=False):
        """
        Approximate the offset of energy between FF and Reference evaluators

        Args:
            ref_dics (dict): reference data dictionary
            e_offset (float): energy offset
            steps (int): optimization steps
            opt_dict (dict): optimization terms and values
            parameters0 (array): initial full parameters
            debug (Boolean): to show the progress or not
        Returns:
            The optimized charge values
        """
        if parameters0 is None:
            parameters0 = self.params_init.copy()
        else:
            assert(len(parameters0) == len(self.params_init))
        self.update_ff_parameters(parameters0)

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

        for term in terms:
            if len(ids) > 0:
                if term != 'charge':
                    x.extend(opt_dict[term])
                    ids.append(ids[-1] + len(opt_dict[term]))
                else:
                    x.extend([1.0])
            else:
                if term != 'charge':
                    x.extend(opt_dict[term])
                    ids.append(len(opt_dict[term]))
                else:
                    x.extend([1.0])

        #print("Starting", x)
        #x = [item for sublist in values for item in sublist]
        _, sub_bounds, _ = self.get_sub_parameters(parameters0, terms)
        bounds = [item for sublist in sub_bounds for item in sublist]

        def obj_fun(x, ref_dics, parameters0, e_offset, ids, charges=None):
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
            objective = self.get_objective(ref_dics, e_offset)
            #print("Debugging", values[0][:5], objective)
            return objective

        # set call back function for debugging
        def objective_function_wrapper(x, ref_dics, parameters0, e_offset, ids, charges):
            global last_function_value
            last_function_value = obj_fun(x, ref_dics, parameters0, e_offset, ids, charges)
            return last_function_value

        def my_callback(xk):
            print(f"Solution: {xk[:2]}, Objective: {last_function_value}")
        callback = my_callback if debug else None

        # Actual optimization
        res = minimize(#obj_fun,
                       objective_function_wrapper,
                       x,
                       method = 'Nelder-Mead',
                       args = (ref_dics, parameters0, e_offset, ids, charges),
                       options = {'maxiter': steps, 'disp': True},
                       bounds = bounds,
                       callback = callback,
                       )

        # Rearrange the optimized parameters to the list of values
        values = []
        for i in range(len(ids)):
            if i == 0:
                id1 = 0
            else:
                id1 = ids[i-1]
            values.append(res.x[id1:ids[i]])

        # The last x value is the ratio of charge
        if charges is not None:
            values.append(res.x[-1]*charges)

        return res.x, res.fun, values


    def load_parameters(self, filename):
        """
        Load the parameters from a given xml file

        Args:
            filename: xml file to store the parameters information
        """
        if filename.endswith('.xml'):
            dics = xml_to_dict_list(filename)[0]
            parameters = []
            for term in self.terms:
                parameters.extend(dics[term].tolist())
            return np.array(parameters)
        else:
            raise ValueError("Unsupported file format")


    def export_parameters(self, filename='parameters.xml', parameters=None):
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
        if filename.endswith('.xml'):
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

            # Use prettify to get a pretty-printed XML string
            pretty_xml = prettify(root)

            # Write the pretty-printed XML to a file
            with open(filename, 'w') as f:
                f.write(pretty_xml)


    def load_references(self, filename):
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

    def get_score_on_single_reference(self, ref_structure, reference_criteria):
        # Calculate score for a single reference structure
        pass

    def get_scores_on_multi_references(self, ref_structures, ref_criteria):
        # Calculate scores for multiple reference structures
        pass

    def get_report_single_reference(self, ref_structure):
        # Compute and return report for a single reference structure
        pass

    def get_report_on_multi_references(self):
        # Compute and return report on multiple references
        pass



if __name__ == "__main__":
    from pyxtal.db import database

    # db = database('../HT-OCSP/benchmarks/Si.db')
    db = database("../HT-OCSP/benchmarks/test.db")
    xtal = db.get_pyxtal("ACSALA")
    smiles = [mol.smile for mol in xtal.molecules]
    assert smiles[0] is not None
    params = ForceFieldParameters(smiles)
    print(params)
    print(params.constraints)
    params.evaluate_ff_single(xtal.to_ase(resort=False))
