#!/usr/bin/env python3
import copy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from pymatgen.core import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from molff.utils import temporary_directory_change


class RDKIT:
    """
    rdkit utilities
    """

    @staticmethod
    def file_to_rdkit_mol(path: Path or str) -> Chem.Mol:
        """
        Load a PDB file to RDKit :class:`Mol` object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError
        elif path.suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        elif path.suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
        else:
            raise ValueError(f"{path.suffix} is not a supported file type")

        Chem.SanitizeMol(
            mol,
            (Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY ^ Chem.SANITIZE_ADJUSTHS),
        )
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
        Chem.AssignStereochemistryFrom3D(mol)
        mol.SetProp("?_Name", path.stem)
        return mol

    @staticmethod
    def smiles_to_rdkit_mol(smiles: str, name: str = None) -> Chem.Mol:
        """
        smiles_string: str
        """
        mol = AllChem.MolFromSmiles(smiles)
        if type(name) is str:
            mol.SetProp("_Name", name)
        mol_hydrogens = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol_hydrogens, randomSeed=1)
        AllChem.SanitizeMol(mol_hydrogens)
        return mol_hydrogens

    @staticmethod
    def get_smiles(
        rdkit_mol: Chem.Mol,
        isomeric: bool = True,
        explicit_hydrogens: bool = True,
        mapped: bool = False,
        canonical: bool = True,
    ) -> str:
        cp_mol = copy.deepcopy(rdkit_mol)
        if mapped:
            explicit_hydrogens = True
            for atom in cp_mol.GetAtoms():
                # mapping starts from 1 as 0 means no mapping in rdkit
                atom.SetAtomMapNum(atom.GetIdx() + 1)
        if not explicit_hydrogens:
            cp_mol = Chem.RemoveHs(cp_mol)
        return Chem.MolToSmiles(cp_mol, isomericSmiles=isomeric, allHsExplicit=explicit_hydrogens, canonical=canonical)

    @staticmethod
    def get_smirks_matches(rdkit_mol: Chem.Mol, smirks: str) -> List[Tuple[int, ...]]:
        original = copy.deepcopy(rdkit_mol)
        keymol = Chem.MolFromSmarts(smirks)
        mapping = {}
        for atom in keymol.GetAtoms():
            kid = atom.GetAtomMapNum()
            if kid != 0:
                mapping[kid + 1] = atom.GetIdx()
        all_matches = set()
        for match in original.GetSubstructMatches(keymol, uniquify=True, useChirality=True):
            tmp = [match[atom] for atom in mapping.values()]
            all_matches.add(tuple(tmp))
        return list(all_matches)

    @staticmethod
    def get_smartrs(rdkit_mol: Chem.Mol) -> str:
        return Chem.MolToSmarts(rdkit_mol)

    @staticmethod
    def generate_conformers(
        rdkit_mol: Chem.Mol,
        numConfs: Optional[int] = 350,
        maxIters: Optional[int] = 1000,
        rmsTh: Optional[float] = 1.0,
        optimize: bool = True,
    ) -> List[Atoms]:
        """
        numConf : number of conformers
        maxIters : max iteration
        """
        AllChem.EmbedMultipleConfs(
            rdkit_mol,
            numConfs=numConfs,
            randomSeed=1,
            clearConfs=False,
            useBasicKnowledge=True,
            pruneRmsThresh=rmsTh,
            enforceChirality=True,
        )
        if optimize:
            rmslist = []
            AllChem.AlignMolConformers(rdkit_mol, RMSlist=rmslist)
            AllChem.MMFFOptimizeMolecule(rdkit_mol, maxIters=maxIters)
        positions = rdkit_mol.GetConformers()

        return [RDKIT.to_ase(rdkit_mol, conformer.GetPositions()) for conformer in positions]

    @staticmethod
    def get_elements(rdkit_mol: Chem.Mol) -> List[str]:
        ret = []
        for atom in rdkit_mol.GetAtoms():
            ret.append(atom.GetSymbol())
        return ret

    @staticmethod
    def to_ase(rdkit_mol: Chem.Mol, positions: Optional[np.array] = None) -> Atoms:
        if positions is None:
            RDKIT.generate_conformers(rdkit_mol, numConfs=1)
            positions = rdkit_mol.GetConformer().GetPositions()
        return Atoms(RDKIT.get_elements(rdkit_mol), positions=positions)

    @staticmethod
    def get_netcharge_and_spin_multiplicity(rdkit_mol: Chem.Mol) -> Tuple[int, int]:
        net_charge = Chem.GetFormalCharge(rdkit_mol)
        S = 0.5 * Descriptors.NumRadicalElectrons(rdkit_mol)
        spin_multiplicity = int(2 * S + 1)
        return net_charge, spin_multiplicity

    @staticmethod
    def to_pdb(rdkit_mol: Chem.Mol, fname: Path) -> None:
        Chem.rdmolfiles.MolToPDBFile(rdkit_mol, str(fname), flavor=0b001100)

    @staticmethod
    def to_sdf(rdkit_mol: Chem.Mol, fname: Path) -> None:
        Chem.rdmolfiles.MolToMolFile(rdkit_mol, str(fname))

    @staticmethod
    def to_file(rdkit_mol: Chem.Mol, fname: Path or str) -> None:
        fname = Path(fname)
        if fname.suffix == ".pdb":
            RDKIT.to_pdb(rdkit_mol, fname)
        elif fname.suffix == ".sdf":
            RDKIT.to_sdf(rdkit_mol, fname)

    @staticmethod
    def to_pdbstr(rdkit_mol: Chem.Mol) -> str:
        return Chem.rdmolfiles.MolToPDBBlock(rdkit_mol, flavor=0b001100)


def smiles_to_ase_and_pmg(smiles: str, molname: str) -> Tuple[Atoms, Molecule, int, int, Chem.Mol]:
    """
    Convert smiles to ase and pymatgen using rdkit
    input:
        smiles: str
        molname: str
    output:
        ase_atoms: Atoms
        pmg_mol: Molecule
        charge: int
        spin_multiplicity: int
        rdkit_mol: Chem.Mol
    """

    rdkit_mol = RDKIT.smiles_to_rdkit_mol(smiles, molname)
    charge, spin_multiplicity = RDKIT.get_netcharge_and_spin_multiplicity(rdkit_mol)
    ase_atoms = RDKIT.generate_conformers(rdkit_mol)[0]
    charges = np.zeros(len(ase_atoms))
    magmoms = np.zeros(len(ase_atoms))
    charges[0] = charge
    magmoms[0] = spin_multiplicity - 1
    ase_atoms.set_initial_charges(charges)
    ase_atoms.set_initial_magnetic_moments(magmoms)
    pmg_mol = AseAtomsAdaptor.get_molecule(ase_atoms, Molecule)
    pmg_mol.set_charge_and_spin(charge, spin_multiplicity)
    return ase_atoms, pmg_mol, charge, spin_multiplicity, rdkit_mol
