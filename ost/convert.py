#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

from ost.interfaces.ambertools import run_antechamber
from ost.interfaces.parmed import amber_to_pdstruc, ommxml_to_pdstruc, ParmEdStructure
from ost.interfaces.rdkit import smiles_to_ase_and_pmg, RDKIT
from ost.utils import dump_toml, temporary_directory_change

from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange
from openff.units import unit


def convert_gaff(
    smiles: str,
    molname: str,
    atomtyping: Optional[str] = "gaff",
    chargemethod: Optional[str] = "gas",
    base: Optional[str] = "ff",
    cleanup: Optional[bool] = True,
    savetoml: Optional[str] = None,
) -> dict:
    """
    Do ambertools to generate gaff parameter of target smiles.
    inputs:
        smiles: str, target smiles
        molname: str, name of the molecule
        atomtyping: str, atomtyping method, default is gaff
        chargemethod: str, charge method, default is gas
        base: str, base name of the output files, default is ff
        cleanup: bool, whether to clean up the temporary directory, default is True
        savetoml: output toml file, default is None
    outputs:
        dict
    """

    with temporary_directory_change(cleanup=cleanup, prefix="_".join(["tmp", molname, atomtyping, chargemethod]) + "_"):
        ase_atoms, pmgmol, charge, spin_multiplicity, _rdkit_mol = smiles_to_ase_and_pmg(smiles, molname)
        path = Path(f"{base}_init.mol2")
        pmgmol.to(filename=str(path), fmt="mol2")
        # Don't run charge analysis for 1-atom residue
        if len(ase_atoms) == 1:
            chargemethod = None
            print(pmgmol.to(fmt="mol2"))

        amber_files = run_antechamber(
            molname, path, charge, spin_multiplicity, resname="UNK", atomtyping=atomtyping, chargemethod=chargemethod, base=base
            )
        struc = amber_to_pdstruc(amber_files["prmtop"], amber_files["inpcrd"], base)

    dic = {
        "mol_name": molname,
        "mol_smi": smiles,
        "mol_formula": ase_atoms.get_chemical_formula(),
        "mol_weight": float(sum(ase_atoms.get_masses())),
        "mol_charge": charge,
        "mol_spin_multiplicity": spin_multiplicity,
        "data": {"omm_info": struc.ffdic},
    }
    if savetoml:
        dump_toml(dic, savetoml)

    return dic


def convert_openff(
    smiles: str,
    molname: str,
    forcefield_name: Optional[str] = "openff-2.0.0.offxml",
    base: Optional[str] = "ff",
    cleanup: Optional[bool] = True,
    savetoml: Optional[str] = None,
) -> dict:
    """
    Do openff-toolkit to generate openff parameter of target smiles.
    inputs:
        smiles: str, target smiles
        molname: str, name of the molecule
        ffname: str, force field name
        base: str, base name of the output files, default is ff
        cleanup: bool, whether to clean up the temporary directory, default is True
        savetoml: output toml file, default is None
    outputs:
        dict
    """

    with temporary_directory_change(cleanup=cleanup, prefix="_".join(["tmp", molname, "openff"]) + "_"):
        ase_atoms, pmgmol, charge, spin_multiplicity, rdkit_mol = smiles_to_ase_and_pmg(smiles, molname)
        molecule = Molecule.from_rdkit(rdkit_mol)
        topology = Topology.from_molecules(molecule)
        forcefield = ForceField(forcefield_name)
        print(forcefield_name); print(molecule)
        out = Interchange.from_smirnoff(force_field=forcefield, topology=topology)
        print(out._to_parmed())
        struc = ParmEdStructure.from_structure(out._to_parmed())
        struc.box = None

    dic = {
        "mol_name": molname,
        "mol_smi": smiles,
        "mol_formula": ase_atoms.get_chemical_formula(),
        "mol_weight": float(sum(ase_atoms.get_masses())),
        "mol_charge": charge,
        "mol_spin_multiplicity": spin_multiplicity,
        "data": {"omm_info": struc.ffdic},
    }
    for k in struc.ffdic['omm_forcefield'][0]['ForceField'].keys(): 
        print(struc.ffdic['omm_forcefield'][0]['ForceField'][k])
    if savetoml:
        dump_toml(dic, savetoml)

    return dic


def amber2toml(molname, dirname, tomlpath, base="ff"):
    prmtop = Path(dirname) / f"{base}.prmtop"
    inpcrd = Path(dirname) / f"{base}.inpcrd"
    struc = amber_to_pdstruc(prmtop, inpcrd, base)
    ase_atoms = struc.to_ase()
    smiles = open(Path(dirname) / f"{molname}.smi").read()
    cands = open(Path(dirname) / f"{molname}.charge_spin").read()
    charge, spin_multiplicity = [int(x) for x in cands.split()]

    dic = {
        "mol_name": molname,
        "mol_smi": smiles,
        "mol_formula": ase_atoms.get_chemical_formula(),
        "mol_weight": float(sum(ase_atoms.get_masses())),
        "mol_charge": charge,
        "mol_spin_multiplicity": spin_multiplicity,
        "data": {"omm_info": struc.ffdic},
    }
    dump_toml(dic, Path(tomlpath) / f"{molname}.toml")


def convert_ffxml(smiles: str,
                  molname: str,
                  ffxml: str,
                  tomlpath: str,
                  cleanup=True) -> dict:
    """
    """
    with temporary_directory_change(cleanup=cleanup, prefix="_".join(["tmp", molname, "ffxml"]) + "_"):
        ase_atoms, _pmgmol, charge, spin_multiplicity, rdkit_mol = smiles_to_ase_and_pmg(smiles, molname)
        RDKIT.to_pdb(rdkit_mol=rdkit_mol, fname=molname+".pdb")
        struc = ommxml_to_pdstruc(ffxml, molname+".pdb")
        ase_atoms = struc.to_ase()

        dic = {
            "mol_name": molname,
            "mol_smi": smiles,
            "mol_formula": ase_atoms.get_chemical_formula(),
            "mol_weight": float(sum(ase_atoms.get_masses())),
            "mol_charge": charge,
            "mol_spin_multiplicity": spin_multiplicity,
            "data": {"omm_info": struc.ffdic},
        }
        dump_toml(dic, Path(tomlpath) / f"{molname}.toml")

    return dic


if __name__ == "__main__":
    inputs = [
        ["CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin", "gaff", "gas"],
    ]
    for smiles, molname, at, cm in inputs:
        dic = convert_gaff(smiles, molname, atomtyping=at, chargemethod=cm)
        tomlname = f"{molname}_{cm}.toml"
        dump_toml(dic, tomlname)
