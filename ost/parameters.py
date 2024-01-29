import json
import xml.etree.ElementTree as ET
import numpy as np

from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry

from ost.forcefield import forcefield
from ost.lmp import LAMMPSCalculator
from mace.calculators import mace_mp

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
                 device = 'cpu'):
        """
        Initialize the parameters

        Args:
            smiles (list): list of smiles strings
            style (str): 'gaff' or 'openff'
            chargemethod (str): 'mmff94', 'am1bcc', 'am1-mulliken', 'gasteiger'
            ff_evaluator (str): 'lammps' or 'charmm'
            ref_evaluator (str): 'mace' or 'trochani'
        """

        self.ff = forcefield(smiles, style, chargemethod)
        self.get_default_ff_parameters()
        self.parameters_current = []
        self.reference_data = []
        self.ff_evaluator = ff_evaluator
        self.ref_evaluator = ref_evaluator
        if ref_evaluator == 'mace':
            self.calculator = mace_mp(model = "small",
                                      dispersion = True,
                                      default_dtype = "float64",
                                      device = device)

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
        N_LJ, N_bond, N_angle, N_proper, N_improper, N_charges = 0, 0, 0, 0, 0, 0
        for molecule in self.ff.molecules:
            ps = molecule.get_parameterset_with_resname_as_prefix()

            # LJ parameters (rmin, epsilon)
            for at_type in ps.atom_types.keys():
                rmin = ps.atom_types[at_type].rmin
                epsilon = ps.atom_types[at_type].epsilon
                params.append(rmin)
                params.append(epsilon)
                bounds.append((rmin + deltas[0], rmin + deltas[1]))
                bounds.append((epsilon * coefs[0], epsilon * coefs[1]))
                N_LJ += 2

            # Bond (k, req)
            for bond_type in ps.bond_types.keys():
                k = ps.bond_types[bond_type].k
                req = ps.bond_types[bond_type].req
                params.append(k)
                params.append(req)
                bounds.append((k * coefs[0], k * coefs[1]))
                bounds.append((req + deltas[0], req + deltas[1]))
                N_bond += 2

            # Angle (k, theteq)
            for angle_type in ps.angle_types.keys():
                k = ps.angle_types[angle_type].k
                theteq = ps.angle_types[angle_type].theteq
                params.append(k)
                params.append(theteq)
                bounds.append((k * coefs[0], k * coefs[1]))
                bounds.append((theteq + deltas[0], theteq + deltas[1]))
                N_angle += 2

            # Proper (phi_k) # per=2, phase=180.000,  scee=1.200, scnb=2.000
            for dihedral_type in ps.dihedral_types.keys():
                for proper in ps.dihedral_types[dihedral_type]:
                    phi_k = proper.phi_k
                    per = proper.per
                    phase = proper.phase
                    params.append(phi_k)
                    params.append(per)
                    params.append(phase)
                    bounds.append((phi_k * coefs[0], phi_k * coefs[1]))
                    bounds.append((per, per))
                    bounds.append((phase, phase))
                    N_proper += 1

            # Improper (phi_k) #  per=2, phase=180.000,  scee=1.200, scnb=2.000>
            for improper_type in ps.improper_periodic_types.keys():
                #for improper in ps.improper_periodic_types[improper_type]:
                improper = ps.improper_periodic_types[improper_type]
                phi_k = improper.phi_k
                per = improper.per
                phase = improper.phase
                params.append(phi_k)
                params.append(per)
                params.append(phase)
                bounds.append((phi_k * coefs[0], phi_k * coefs[1]))
                bounds.append((per, per))
                bounds.append((phase, phase))
                params.append(phi_k)
                bounds.append((phi_k * coefs[0], phi_k * coefs[1]))
                N_improper += 1

        for molecule in self.ff.molecules:
            for at in molecule.atoms:
                params.append(at.charge)
                bounds.append((at.charge + deltas[0], at.charge + deltas[1]))
            id1 = len(params) - len(molecule.atoms)
            id2 = len(params)
            sum_chg = sum(params[id1:id2])
            constraints.append((id1, id2))
            N_charges += len(molecule.atoms)

        # N_LJ, N_bond, N_angle, N_proper, N_improper, N_charges
        self.N_LJ = N_LJ
        self.N_bond = N_bond
        self.N_angle = N_angle
        self.N_proper = N_proper
        self.N_improper = N_improper
        self.N_charges = N_charges

        self.params_init = params
        self.constraints = constraints
        self.bounds = bounds

    def __str__(self):
        s = "\n------Force Field Parameters------\n"
        s += "LJ:          {:3d}\n".format(self.N_LJ)
        s += "Bond:        {:3d}\n".format(self.N_bond)
        s += "Angle:       {:3d}\n".format(self.N_angle)
        s += "Proper:      {:3d}\n".format(self.N_proper)
        s += "Improper:    {:3d}\n".format(self.N_improper)
        s += "Charges:     {:3d}\n".format(self.N_charges)
        s += "Total:       {:3d}\n".format(len(self.params_init))
        s += "Constraints: {:3d}\n".format(len(self.constraints))
        s += "FF_code:    {:s}\n".format(self.ff_evaluator)
        s += "Ref_code:   {:s}\n".format(self.ref_evaluator)
        return s

    def __repr__(self):
        return str(self)

    def augment_ref_configurations(self, ref_structure, fmax=0.1, steps=250):
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

        print('# Get the elastic configurations: 3*4')
        cell0 = ref_structure.cell.array
        for ax in range(3):
            for coef in [0.85, 0.92, 1.08, 1.15]:
                structure = ref_structure.copy()
                cell = cell0.copy()
                cell[ax, ax] *= coef
                structure.set_cell(cell, scale_atoms=True)
                ref_dic = self.evaluate_ref_single(structure, options=[True, False, True])
                ref_dic['tag'] = 'elastic'
                ref_dics.append(ref_dic)

        print('# Get the atomic purturbation: 3*10')
        pos0 = ref_structure.get_positions()
        for dx in [0.05, 0.1, 0.2]:
            for i in range(10):
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
                   'stress': None}
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
                  'stress': None}
        if self.ff_evaluator == 'lammps':
            lmp_struc = self.ff.get_ase_lammps(structure)
            calc = LAMMPSCalculator(lmp_struc)
            eng, force, stress = calc.express_evaluation()
            if options[0]: # Energy
                ff_dic['energy'] = eng
            if options[1]: # forces
                ff_dic['forces'] = force
            if options[2]:
                ff_dic['stress'] = stress
        return ff_dic

    def optimization(self, ref_structures, method):
        """
        Counting the offset in total energy
        """
        if method in ['SA', 'Nelder-Mead', 'GA']:
            # Perform optimization
            pass
        else:
            raise ValueError("Unsupported optimization method")


    def load_parameters(self, filename):
        if filename.endswith('.json'):
            with open(filename, 'r') as file:
                self.parameters_current = json.load(file)
        elif filename.endswith('.xml'):
            tree = ET.parse(filename)
            root = tree.getroot()
            # Parse XML to parameters_current
        else:
            raise ValueError("Unsupported file format")
        self.parameters_init = self.parameters_current.copy()


    def export_parameters(self, filename):
        """
        Export the parameters
        """
        dic = {}
        if filename.endswith('.json'):
            with open(filename, 'w') as file:
                json.dump(self.parameters_current, file)
        elif filename.endswith('.xml'):
            # Convert parameters_current to XML and save
            pass
        else:
            raise ValueError("Unsupported file format")


    def load_references(self, filename):
        if filename.endswith(('.json', '.xml', '.db')):
            # Load reference data from file
            pass
        else:
            raise ValueError("Unsupported file format")

    def export_references(self, filename):
        if filename.endswith(('.json', '.xml', '.db')):
            # Export reference data to file
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
