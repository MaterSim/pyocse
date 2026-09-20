#!/usr/bin/env python
from openff.interchange import Interchange
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from pyocse.interchange_parmed import _to_parmed
from pyocse.interfaces.parmed import ParmEdStructure, ommffs_to_paramedstruc
from pyocse.interfaces.rdkit import smiles_to_ase_and_pmg
from pyocse.lmp import LAMMPSStructure
from pyxtal.constants import single_smiles
import os
import re

# SMIRNOFF force field selection for style="openff".
#
# Sage 2.0.0 has no proper torsion for the sp3 C-N-N(=O)=O nitramine pattern
# (RDX, HMX, CL-20, tetryl fail with "ProperTorsionHandler was not able to find
# parameters"); Sage 2.1 covers it. Short names map to the first offxml the
# installed openff-toolkit / openff-forcefields can load:
SAGE_VERSIONS = {
    "1.3": ["openff-1.3.1.offxml", "openff-1.3.0.offxml"],
    "2.0": ["openff-2.0.0.offxml"],
    "2.1": ["openff-2.1.0.offxml", "openff-2.1.0-rc.1.offxml"],
    "2.2": ["openff-2.2.1.offxml", "openff-2.2.0.offxml"],
}
DEFAULT_SAGE_VERSION = "2.1"


def available_openff_files():
    """offxml files the installed OpenFF stack can load (empty set if unknown)."""
    try:
        from openff.toolkit.typing.engines.smirnoff import get_available_force_fields
        return set(get_available_force_fields())
    except Exception:
        return set()


def resolve_openff(name=None):
    """Turn a Sage version or offxml name into a loadable offxml file name.

    Accepts "2.0", "2.1", "2.0.0", "sage-2.1", "openff-2.1", "Sage 2.0",
    a full file name such as "openff-2.0.0.offxml", or None (-> the default
    version). Raises ValueError for versions this installation cannot load.
    """
    if name is None or str(name).strip() == "":
        name = DEFAULT_SAGE_VERSION
    name = str(name).strip()
    if name.endswith(".offxml"):
        return name
    key = re.sub(r"^(openff|sage)[-_ ]*", "", name, flags=re.IGNORECASE).strip()
    key = re.sub(r"\.0$", "", key) if re.fullmatch(r"\d+\.\d+\.0", key) else key  # 2.0.0 -> 2.0
    if key not in SAGE_VERSIONS:
        raise ValueError(
            f"unknown OpenFF/Sage version {name!r}; use one of {sorted(SAGE_VERSIONS)} "
            "or a full *.offxml file name"
        )
    avail = available_openff_files()
    for cand in SAGE_VERSIONS[key]:
        if not avail or cand in avail:
            return cand
    raise ValueError(
        f"Sage {key} is not installed ({SAGE_VERSIONS[key]} not found); "
        "install conda-forge openff-forcefields or pick another version"
    )


def _split_openff_style(style):
    """'openff-2.0' -> ('openff', '2.0'); 'openff' / 'gaff' -> (style, None)."""
    if isinstance(style, str) and style.lower().startswith("openff") and len(style) > 6:
        return "openff", style[6:].lstrip("-_ ")
    return style, None


# Process-wide default: PYOCSE_OPENFF may hold a version ("2.0") or a file name.
# Multiprocessing workers inherit it. Explicit ff_name= / openff_version=
# arguments to forcefield() / ForceFieldParameters() win.
DEFAULT_OPENFF = resolve_openff(os.environ.get("PYOCSE_OPENFF"))


class forcefield:
    """
    Generate force field for atomistic simulations. Essentially, it supports the
    following functions:
        - 1. prepare the structure xyz and force fields for the given molecule
        - 2. generate the structures with multiple molecules and force fields
    """

    def __init__(self, smiles, style="gaff", chargemethod="am1bcc", workdir=".",
                 ff_name=None, openff_version=None):
        """
        Args:
            smiles (list): molecular smiles
            style (str): 'gaff', 'openff', or 'openff-2.0' / 'openff-2.1' (version shorthand)
            chargemethod (str): 'mmff94', 'am1bcc', 'am1-mulliken', 'gasteiger'
            workdir (str): '.'
            ff_name (str): SMIRNOFF offxml for style='openff' (full file name)
            openff_version (str): Sage version for style='openff', e.g. '2.0' or '2.1'
                           (default: PYOCSE_OPENFF env or Sage 2.1); ignored for 'gaff'
        """
        self.dics = []
        self.smiles = smiles
        style, style_version = _split_openff_style(style)
        self.style = style
        if ff_name is not None:
            self.ff_name = resolve_openff(ff_name)
        elif openff_version is not None or style_version:
            self.ff_name = resolve_openff(openff_version or style_version)
        else:
            self.ff_name = DEFAULT_OPENFF
        self.chargemethod = chargemethod
        if max([len(s) for s in smiles]) > 180:
            self.chargemethod = "gasteiger" #chargemethod
            print("Use gasteiger chargemethod to save time for large molecule")
        self.workdir = workdir
        self.set_partial_charges()

        # setup converter
        if style == "gaff":
            converter = get_gaff
        else:
            converter = lambda smi, chg: get_openff(smi, chg, ff_name=self.ff_name)

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
            if smi in single_smiles:
                molecule.set_charges(self.partial_charges[i])
            else:
                molecule.set_charges(self.partial_charges[i].m)
            if style == 'openff':
                for at in molecule.atoms:
                    at.atom_type.number = i
            self.molecules.append(molecule)

    def set_partial_charges(self):
        """
        Set the partial charge via openff-toolkit
        May do antechamber directly later
        """
        self.partial_charges = []
        for smi in self.smiles:
            if smi in single_smiles:
                pattern = r'([A-Za-z]+)([+\-]?\d*)'
                matches = re.search(pattern, smi)
                if matches:
                    charge_str = matches.group(2)
                    if charge_str == '+':
                        charge = 1.0
                    elif charge_str == '-':
                        charge = -1.0
                    else:
                        charge = float(charge_str)
                else:
                    raise ValueError("smiles cannot be analyzed", smi)
                self.partial_charges.append([charge])

            else:
                molecule = Molecule.from_smiles(smi, allow_undefined_stereo=True)
                molecule.assign_partial_charges(self.chargemethod)
                self.partial_charges.append(molecule.partial_charges)
        # print(self.partial_charges)

    def get_ase_lammps(self, atoms, numMols):
        """
        Add the lammps ff information into the give atoms object

        Args:
            Atoms: the ase atoms following the atomic order in self.molecules
            numMols: the list of number of molecules

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
            for i, m in enumerate(numMols):
                mols += [self.molecules[i]*m]
            pd_struc = reduce(add, mols)#; print(pd_struc); print(numMols)
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
        else:
            from functools import reduce
            from operator import add
            pd_struc = reduce(add, self.molecules)
        atoms = LAMMPSStructure.from_structure(pd_struc)
        atoms.set_pbc(pbc)
        atoms.title = '.'.join(self.smiles)

        return atoms._write_input(self.workdir + '/' + lmp_dat)

    def reset_lammps_cell(self, atoms0):
        """
        set the cell into lammps format
        """
        from pyocse.utils import reset_lammps_cell
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


def get_openff(smiles, chargemethod, ff_name=DEFAULT_OPENFF):
    """
    Get Openff parameters from smiles
    """
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
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
    from pyocse.interfaces.ambertools import run_antechamber
    from pyocse.interfaces.parmed import amber_to_pdstruc
    from pyocse.utils import temporary_directory_change

    # with temporary_directory_change(cleanup=False, prefix='tmp'):
    with temporary_directory_change(cleanup=True, prefix="tmp"):
        _, pmgmol, charge, spin, _ = smiles_to_ase_and_pmg(smiles, "test")
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


def get_openff_with_silicon(xml=DEFAULT_OPENFF):
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
