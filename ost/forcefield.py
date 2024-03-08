#!/usr/bin/env python
from openff.interchange import Interchange
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from ost.interchange_parmed import _to_parmed
from ost.interfaces.parmed import ParmEdStructure, ommffs_to_paramedstruc
from ost.interfaces.rdkit import smiles_to_ase_and_pmg
from ost.lmp import LAMMPSStructure
import numpy as np

class forcefield:
    """
    Generate force field for atomistic simulations. Essentially, it supports the
    following functions:
        - 1. prepare the structure xyz and force fields for the given molecule
        - 2. generate the structures with multiple molecules and force fields
    """

    def __init__(self, smiles=None, style="gaff", chargemethod="am1bcc"):
        """
        Args:
            smiles (list): molecular smiles
            style (str): 'gaff' or 'openff'
            chargemethod (str): 'mmff94', 'am1bcc', 'am1-mulliken', 'gasteiger'
        """
        self.dics = []
        self.smiles = smiles
        self.chargemethod = chargemethod
        self.set_partial_charges()

        # setup converter
        converter = get_gaff if style == "gaff" else get_openff

        self.molecules = []
        for i, smi in enumerate(smiles):
            residuename = "U{:02d}".format(i)
            ffdic = converter(smi, chargemethod).ffdic
            # print(ffdic, ffdic.keys())
            # Pass the partial charge
            molecule = ommffs_to_paramedstruc(
                ffdic["omm_forcefield"], ffdic["mol2"], cls=ParmEdStructure
            )
            molecule.ffdic = ffdic
            molecule.change_residue_name(residuename)
            molecule.set_charges(self.partial_charges[i].m)
            self.molecules.append(molecule)

    def set_partial_charges(self):
        """
        Set the partial charge via openff-toolkit
        May do antechamber directly later
        """
        self.partial_charges = []
        for smi in self.smiles:
            molecule = Molecule.from_smiles(smi)
            molecule.assign_partial_charges(self.chargemethod)
            self.partial_charges.append(molecule.partial_charges)
        # print(self.partial_charges)

    def get_ase_lammps(self, atoms):
        """
        Add the lammps ff information into the give atoms object

        Args:
            Atoms: the ase atoms following the atomic order in self.molecules

        Return:
            Atoms with lammps ff information
        """
        # first adjust the cell into lammps format
        pbc = atoms.pbc
        atoms = self.reset_lammps_cell(atoms)
        if len(self.molecules) == 1:
            pd_struc = self.molecules[0].copy(cls=ParmEdStructure)
            pd_struc.update(atoms)
            atoms = LAMMPSStructure.from_structure(pd_struc)
        else:
            from functools import reduce
            from operator import add
            mols = []
            for i, m in enumerate(self.xtal_n_mols):
                mols += [self.molecules[i]*m]
            pd_struc = reduce(add, mols) #self.molecules)
            pd_struc.update(atoms)
            atoms = LAMMPSStructure.from_structure(pd_struc)
            #struc.restore_ffdic()
        atoms.set_pbc(pbc)
        atoms.title = '.'.join(self.smiles)
        return atoms

    def get_lammps_in(self, lmp_dat='lmp.dat'):
        """
        Add the lammps ff information into the give atoms object

        Args:
            Atoms: the ase atoms following the atomic order in self.molecules

        Return:
            Atoms with lammps ff information
        """
        # first adjust the cell into lammps format
        pbc = [True, True, True]
        if len(self.molecules) == 1:
            atoms = self.molecules[0].to_ase()
            pd_struc = self.molecules[0].copy(cls=ParmEdStructure)
            pd_struc.update(atoms)
            atoms = LAMMPSStructure.from_structure(pd_struc)
        atoms.set_pbc(pbc)
        atoms.title = '.'.join(self.smiles)

        return atoms._write_input(lmp_dat)

    def reset_lammps_cell(self, atoms0):
        """
        set the cell into lammps format
        """
        from ost.utils import reset_lammps_cell
        return reset_lammps_cell(atoms0)

    def update_parameters(self, parameters):
        """
        update the forcefield parameters:

        Args:
            parameters: a 1D array of FF parameters
        """
        count = 0
        # Bond (k, req)
        for molecule in self.molecules:
            for bond_type in molecule.bond_types: #.keys():
                k = parameters[count]
                req = parameters[count + 1]
                bond_type.k = k
                bond_type.req = req
                count += 2
        # Angle (k, theteq)
        for molecule in self.molecules:
            for angle_type in molecule.angle_types: #.keys():
                k = parameters[count]
                theteq = parameters[count + 1]
                angle_type.k = k
                angle_type.theteq = theteq
                count += 2

        # Proper (phi_k) # per=2, phase=180.000,  scee=1.200, scnb=2.000
        for molecule in self.molecules:
            for dihedral_type in molecule.dihedral_types:
                phi_k = parameters[count]
                dihedral_type.phi_k = phi_k
                count += 1

        # nonbond vdW parameters (rmin, epsilon)
        for molecule in self.molecules:
            ps = molecule.get_parameterset_with_resname_as_prefix()
            for key in ps.atom_types.keys():
                rmin = parameters[count]
                epsilon = parameters[count + 1]
                count += 2
                for at in molecule.atoms:
                    label = at.residue.name + at.type
                    if label == key:
                        at.atom_type.rmin = rmin
                        at.atom_type.rmin_14 = rmin
                        at.atom_type.epsilon = epsilon
                        at.atom_type.epsilon_14 = epsilon
                        #break

        # nonbond charges
        for molecule in self.molecules:
            for at in molecule.atoms:
                chg = parameters[count]
                at.charge = chg
                count += 1


def get_openff(smiles, chargemethod, ff_name="openff-2.0.0.offxml"):
    """
    Get Openff parameters from smiles
    """
    molecule = Molecule.from_smiles(smiles)
    molecule.assign_partial_charges(chargemethod)
    topology = Topology.from_molecules(molecule)
    forcefield = get_openff_with_silicon(ff_name)
    out = Interchange.from_smirnoff(
        forcefield, topology, charge_from_molecules=[molecule]
    )
    struc = ParmEdStructure.from_structure(_to_parmed(out))
    struc.box = None
    return struc


def get_gaff(smiles, chargemethod="gas", base="ff"):
    """
    Get gaff parameters from smiles
    """
    from pathlib import Path

    from ost.interfaces.ambertools import run_antechamber
    from ost.interfaces.parmed import amber_to_pdstruc
    from ost.utils import temporary_directory_change

    # with temporary_directory_change(cleanup=False, prefix='tmp'):
    with temporary_directory_change(cleanup=True, prefix="tmp"):
        ase_atoms, pmgmol, charge, spin, _ = smiles_to_ase_and_pmg(smiles, "test")
        path = Path(f"{base}_init.mol2")
        pmgmol.to(filename=str(path), fmt="mol2")

        # Don't run charge analysis for 1-atom residue
        #if len(ase_atoms) == 1:
        chargemethod = None
        amber_files = run_antechamber(
            "test",
            path,
            charge,
            spin,
            resname="UNK",
            atomtyping="gaff",
            chargemethod=chargemethod,
            base=base,
        )
        struc = amber_to_pdstruc(amber_files["prmtop"], amber_files["inpcrd"], base)
    return struc


def get_openff_with_silicon(xml="openff-2.0.0.offxml"):
    ff2 = ForceField(xml)
    # Dreiding Si3   0.3100 4.2700
    # Dreiding P_3   0.3200 4.1500
    # Reference P
    # smirks="[#15:1]
    # epsilon="0.2 * mole ** -1 * kilocalorie ** 1
    # rmin_half="2.1 * angstrom ** 1"></Atom>

    # Si LJ-vdW
    smirks = "[#14:1]"
    epsilon = 0.21 * unit.mole**-1 * unit.kilocalorie
    rmin_half = 2.1 * unit.angstrom
    ff2.get_parameter_handler("vdW").add_parameter(
        {"smirks": smirks, "epsilon": epsilon, "rmin_half": rmin_half, "id": "n23"}
    )
    # Si-O
    smirks = "[#14:1]-[#8:2]"
    length = 1.650 * unit.angstrom
    # length = 1.600 * unit.angstrom
    k = 500 * unit.angstrom**-2 * unit.mole**-1 * unit.kilocalorie
    # k = 700 * unit.angstrom**-2 * unit.mole**-1 * unit.kilocalorie
    ff2.get_parameter_handler("Bonds").add_parameter(
        {"smirks": smirks, "length": length, "k": k, "id": "b89"}
    )
    # To add Si-C/Si-C-Si
    # Si-C
    smirks = "[#14:1]-[#6:2]"
    length = 1.840 * unit.angstrom
    # length = 1.697 * unit.angstrom
    k = 700 * unit.angstrom**-2 * unit.mole**-1 * unit.kilocalorie
    ff2.get_parameter_handler("Bonds").add_parameter(
        {"smirks": smirks, "length": length, "k": k, "id": "b90"}
    )

    # The Si–O–Si angle is
    # 144° in α-quartz,
    # 155° in β-quartz,
    # 147° in α-cristobalite
    # 153±20)° in vitreous silica.
    # Si-O-Si
    # <Angle smirks="[*:1]-[#8:2]-[*:3]" angle="110.3538806181 * degree ** 1" k="130.181232192 * mole ** -1 * radian ** -2 * kilocalorie ** 1" id="a28"></Angle>
    # <Angle smirks="[*:1]-[#8X2+1:2]=[*:3]" angle="115.0964372837 * degree ** 1" k="71.2688479385 * mole ** -1 * radian ** -2 * kilocalorie ** 1" id="a30"></Angle>
    smirks = "[#14:1]-[#8:2]-[#14:3]"
    angle = 1.4947e02 * unit.degree
    k = 2.29000e02 * unit.mole**-1 * unit.radian**-2 * unit.kilocalorie
    param = {"smirks": smirks, "angle": angle, "k": k, "id": "a40"}
    ff2.get_parameter_handler("Angles").add_parameter(
        param
    )  # , before="[*:1]-[#8:2]-[*:3]")

    # Dreiding: Si3      O_3          700.0    1.5870
    # Reference P-O single bond
    # length="1.644080332096 * angstrom ** 1"
    # k="543.1128482396""

    # O-Si-O
    smirks = "[#8:1]-[#14:2]-[#8:3]"
    angle = 1.0947e02 * unit.degree
    k = 2.50e02 * unit.mole**-1 * unit.radian**-2 * unit.kilocalorie
    # k = 2.2974e02 * unit.mole**-1 * unit.radian**-2 * unit.kilocalorie
    ff2.get_parameter_handler("Angles").add_parameter(
        {"smirks": smirks, "angle": angle, "k": k, "id": "a41"}
    )

    # C-Si-Si
    smirks = "[#6:1]-[#14:2]-[*:3]"
    angle = 1.0947e02 * unit.degree
    k = 2.2974e02 * unit.mole**-1 * unit.radian**-2 * unit.kilocalorie
    ff2.get_parameter_handler("Angles").add_parameter(
        {"smirks": smirks, "angle": angle, "k": k, "id": "a42"}
    )

    # Reference *-P-O-*
    # [*:1]-[#8X2:2]-[#15:3]~[*:4]
    # periodicity1="3"
    # phase1="0.0 * degree ** 1"
    # id="t159"
    # k1="0.5946925269495 * mole ** -1 * kilocalorie ** 1"
    # idivf1="1.0"></Proper>
    # Dreiding: X     Si3   O_3   X     1.0    -3 180.00

    # *-Si-O-*
    smirks = "[*:1]-[#14:2]-[#8:3]-[*:4]"
    periodicity1 = 3
    phase1 = 180.0 * unit.degree
    k1 = 0.2 * unit.mole**-1 * unit.kilocalorie
    # k1 = 1.5 * unit.mole**-1 * unit.kilocalorie
    ff2.get_parameter_handler("ProperTorsions").add_parameter(
        {
            "smirks": smirks,
            "periodicity1": periodicity1,
            "phase1": phase1,
            "k1": k1,
            "idivf1": 1.0,
            "id": "t160",
        }
    )

    # *-Si-C-*
    smirks = "[*:1]-[#14:2]-[#6:3]-[*:4]"
    periodicity1 = 3
    phase1 = 180.0 * unit.degree
    k1 = 0.2 * unit.mole**-1 * unit.kilocalorie
    # k1 = 1.5 * unit.mole**-1 * unit.kilocalorie
    ff2.get_parameter_handler("ProperTorsions").add_parameter(
        {
            "smirks": smirks,
            "periodicity1": periodicity1,
            "phase1": phase1,
            "k1": k1,
            "idivf1": 1.0,
            "id": "t161",
        }
    )

    # Referenec *-P(*)-*
    # smirks="[*:1]~[#7X3$(*~[#15,#16](!-[*])):2](~[*:3])~[*:4]
    # periodicity1="2"
    # phase1= 180.0 * degree ** 1
    # k1= 1.1 * mole ** -1 * kilocalorie ** 1
    # id="i3"></Improper>

    # Improper *-Si(*)-*
    smirks = "[*:1]~[#14:2](=[*:3])~[*:4]"
    periodicity1 = 2
    phase1 = 180.0 * unit.degree
    k1 = 0.2 * unit.mole**-1 * unit.kilocalorie
    # k1 = 1.5 * unit.mole**-1 * unit.kilocalorie
    ff2.get_parameter_handler("ImproperTorsions").add_parameter(
        {
            "smirks": smirks,
            "periodicity1": periodicity1,
            "phase1": phase1,
            "k1": k1,
            "id": "i5",
        }
    )
    return ff2


if __name__ == "__main__":
    from pyxtal.db import database

    # db = database('../HT-OCSP/benchmarks/Si.db')
    db = database("../HT-OCSP/benchmarks/test.db")
    xtal = db.get_pyxtal("ACSALA")
    smiles = [mol.smile for mol in xtal.molecules]
    assert smiles[0] is not None
    for style in ["gaff", "openff"]:
        for charge in ["mmff94", "am1bcc", "am1-mulliken", "gasteiger"]:
            print("\ntest", style, charge)
            ff = forcefield(smiles, style=style, chargemethod=charge)
            print(ff.partial_charges)
