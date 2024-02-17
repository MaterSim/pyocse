#!/usr/bin/env python

from io import StringIO
from readline import parse_and_bind

import numpy as np
from openmm import unit as u
from openmm.app import Modeller
from openmm.app.forcefield import ForceField
from parmed import ParameterSet, Structure
from parmed.amber import AmberParm
from parmed.modeller.residue import ResidueTemplateContainer
from parmed.openmm import OpenMMParameterSet, energy_decomposition_system, load_topology
from xmltodict import parse

from ost.utils import dict_to_xmlstr, temporary_directory_change


class ParmEdStructure(Structure):

    _progname = "ParmEd"
    LARGEBOX = [150, 150, 150, 90, 90, 90]
    DEFAULT_EWALD_ERROR_TOLERANCE = 5e-4
    DEFAULT_CUTOFF_SKIN = 2.0
    DEFAULT_CUTOFF_COUL = 9.0
    DEFAULT_CUTOFF_LJIN = 7.0  # use charmm swithing function
    DEFAULT_CUTOFF_LJOUT = 9.0  # use charmm switching function

    @property
    def cutoff(self):
        return max(self.cutoff_coul, self.cutoff_ljout) + self.cutoff_skin

    def set_cutoffs(self, cutoff_skin, cutoff_coul, cutoff_ljin, cutoff_ljout):
        self.cutoff_skin = cutoff_skin
        self.cutoff_coul = cutoff_coul
        self.cutoff_ljin = cutoff_ljin
        self.cutoff_ljout = cutoff_ljout

    #@property
    def gewald(self):
        # from openmm definition
        # http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html
        return (-np.log(self.ewald_error_tolerance * 2.0)) ** 0.5 / self.cutoff_coul

    #@property
    def fftgrid(self):
        """
        determine the grid used for
        """

        grid = np.array(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                8,
                9,
                10,
                12,
                15,
                16,
                18,
                20,
                24,
                25,
                27,
                30,
                32,
                36,
                40,
                45,
                48,
                50,
                54,
                60,
                72,
                75,
                80,
                81,
                90,
                96,
                100,
                108,
                120,
                125,
                135,
                144,
                150,
                160,
                162,
                180,
                200,
                216,
                225,
                240,
                243,
                250,
                270,
                288,
                300,
                324,
                360,
                375,
                400,
            ]
        )
        fftxyz = [0, 0, 0]
        if self.box is not None:
            box = self.box
        else:
            box = self.LARGEBOX

        for i, l in enumerate(box[0:3]):
            tmp = grid[grid > l]
            if len(tmp) == 0:
                # Ensure 1 grid point per Angstrom.
                fftxyz[i] = int(l)+1
            else:
                fftxyz[i] = max([32, tmp[0]])
        return fftxyz

    @property
    def progname(self):
        return self._progname

    @classmethod
    def from_structure(cls, structure):
        return structure.copy(cls)
        #return inst

    @property
    def parameterset(self):
        return ParameterSet.from_structure(self, allow_unequal_duplicates=True)

    def get_parameterset_with_resname_as_prefix(self):
        parameterset = self._do_with_resname(ParameterSet.from_structure, [self])
        return parameterset

    def _do_with_resname(self, func, fargs):

        # set residuename
        bak = []
        for atom in self:
            resname = atom.residue.name
            n = len(resname)
            at = atom.atom_type
            if len(at.name) < n or at.name[:n] != resname:
                at.name = resname + at.name
                bak.append((at, n))
            atom.name = resname + atom.name
            atom.type = resname + atom.type

        ret = func(*fargs)

        for (at, n) in bak:
            at.name = at.name[n:]
        for atom in self:
            n = len(atom.residue.name)
            atom.name = atom.name[n:]
            atom.type = atom.type[n:]

        return ret

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff_skin = self.DEFAULT_CUTOFF_SKIN
        self.cutoff_coul = self.DEFAULT_CUTOFF_COUL
        self.cutoff_ljin = self.DEFAULT_CUTOFF_LJIN
        self.cutoff_ljout = self.DEFAULT_CUTOFF_LJOUT
        self.ewald_error_tolerance = self.DEFAULT_EWALD_ERROR_TOLERANCE
        self._ffdic = None

    def complete(self):
        self.cutoff_skin = self.DEFAULT_CUTOFF_SKIN
        self.cutoff_coul = self.DEFAULT_CUTOFF_COUL
        self.cutoff_ljin = self.DEFAULT_CUTOFF_LJIN
        self.cutoff_ljout = self.DEFAULT_CUTOFF_LJOUT
        self.ewald_error_tolerance = self.DEFAULT_EWALD_ERROR_TOLERANCE
        

    # def __iadd__(self, other):
    #    new = super().__iadd__(other)
    #    return new

    # def copy(self, cls):
    #    copied = super().copy(cls)
    #    return copied

    @property
    def ffdic(self) -> dict:
        if self._ffdic is None:
            self.restore_ffdic()
        return self._ffdic

    @ffdic.setter
    def ffdic(self, dic: dict):
        self._ffdic = dic

    def restore_ffdic(self):
        self._ffdic = parmedstrc_to_ffdic(self, writemol2=False, writexml=False)

    ################################################################################
    @staticmethod
    def _order_indexes(keys):
        ln = len(keys)
        if ln == 2:
            i, j = keys
            x = min((i, j), (j, i))
        elif ln == 3:
            i, j, k = keys
            x = min((i, j, k), (k, j, i))
        elif ln == 4:
            i, j, k, m = keys
            x = min((i, j, k, m), (m, k, j, i))
        return x

    @staticmethod
    def _get_keys_from_parmed_topology(prop):
        attrs = dir(prop)
        keys = []
        if ("atom1" not in attrs) or ("atom2" not in attrs):
            raise ValueError("bad topology")
        if "atom3" not in attrs:
            keys = [prop.atom1.type, prop.atom2.type]
        elif "atom4" not in attrs:
            keys = [prop.atom1.type, prop.atom2.type, prop.atom3.type]
        else:
            keys = [prop.atom1.type, prop.atom2.type, prop.atom3.type, prop.atom4.type]
        keys = [k for k in keys]
        keys = ParmEdStructure._order_indexes(keys)
        return keys

    @staticmethod
    def get_keys_and_types(props):
        from collections import OrderedDict

        from parmed import DihedralType

        typedict = OrderedDict()
        for prop in props:
            orderedkey = ParmEdStructure._get_keys_from_parmed_topology(prop)
            if orderedkey not in typedict:
                if type(prop.type) == DihedralType:
                    per = prop.type.per
                    imp = prop.improper
                    orderedkey = (orderedkey, (per, imp))
                typedict[orderedkey] = prop.type
            else:
                td = typedict[orderedkey]
                if td != prop.type:
                    print("Duplicated paramter exists!", prop.type, td)
                    raise ValueError
        return typedict

    def get_unique_residue(self):
        resnames = []
        resorgs = {}
        for a in self.atoms:
            if a.residue.name not in resnames:
                resnames.append(a.residue.name)
                resorgs[a.residue.number] = a.residue.name
        return resorgs, resnames

    def each_atoms_only_unique_residue(self, with_unique_type=False):
        resorgs, _resnames = self.get_unique_residue()
        for atom in self.atoms:
            if atom.residue.number in resorgs.keys():
                if with_unique_type:
                    unique_type = atom.residue.name + atom.type
                    yield atom, unique_type
                else:
                    yield atom

    @staticmethod
    def get_str_with_writer(writer, *args, **kwargs):
        tmp = StringIO()
        writer(of=tmp, *args, **kwargs)
        return tmp.getvalue()

    def change_residue_name(self, name):
        for residue in self.residues:
            residue.name = name

    @classmethod
    def from_xml_mol2(cls, fxml, fmol2):
        from parmed.formats.mol2 import Mol2File

        mol2 = Mol2File.parse(fmol2, structure=True)
        for atom in mol2.atoms:
            atom.atomic_number = cls.guess_atomic_number(atom.name)
        for r in mol2.topology.residues():
            for tatom, matom in zip(r.atoms(), mol2.atoms):
                tatom.type = matom.type
        return cls.from_xml_top_pos(fxml, mol2.topology, mol2.positions)

    @classmethod
    def from_xml_pdb(cls, fxml, fpdb):
        from parmed.formats.pdb import PDBFile

        pdb = PDBFile.parse(fpdb)
        return cls.from_xml_top_pos(fxml, pdb.topology, pdb.positions)

    @classmethod
    def from_xml_top_pos(cls, fxml, topology, positions):

        ff = ForceField(fxml)  # , 'amber14/tip3pfb.xml')

        modeller = Modeller(topology, positions)
        unmatched_residues = ff.getUnmatchedResidues(modeller.topology)
        if len(unmatched_residues) > 0:
            # for virtual site
            modeller.addExtraParticles(ff)
        system = ff.createSystem(modeller.topology)
        parmed_structure = load_topology(modeller.topology, system, xyz=modeller.positions, condense_atom_types=False)

        # update atomtype by orignalname
        for res in parmed_structure.residues:
            resname = res.name
            ffresname = [x for x in ff._templates.keys()][0]
            if len(ff._templates) and resname != ffresname:
                print("Warning, resname is different between xml and topology file")
                resname = ffresname
            for (
                atom,
                ffatom,
            ) in zip(res.atoms, ff._templates[resname].atoms):
                assert atom.name.lower() == ffatom.name.lower()
                atom.atom_type.name = ffatom.type
                atom.type = ffatom.type

        # manually resiet 1-4 pair scaling from 1.0(parmed default) to amber
        for dt in parmed_structure.dihedral_types:
            dt.scee = 1.2
            dt.scnb = 2.0
        inst = cls.from_structure(parmed_structure)

        return inst

    def to_ase(self, charge=0, spin_multiplicity=1, nopbc=False):
        from ase import Atoms

        elements = self.guess_elements()
        if nopbc:
            atoms = Atoms(
                symbols=elements,
                positions=self.coordinates,
                pbc=[0, 0, 0],
            )
        else:
            if self.box is None:
                box = self.LARGEBOX
            else:
                box = self.box
            atoms = Atoms(
                symbols=elements,
                positions=self.coordinates,
                pbc=[1, 1, 1],
                cell=box,
            )
        charges = np.zeros(atoms.get_number_of_atoms())
        charges[0] = charge
        atoms.set_initial_charges(charges)
        mmos = np.zeros(atoms.get_number_of_atoms())
        mmos[0] = spin_multiplicity - 1
        atoms.set_initial_magnetic_moments(mmos)
        return atoms

    def guess_elements(self):
        from parmed.periodic_table import Element

        return [Element[self.guess_atomic_number(atom.name)] for atom in self.atoms]

    @staticmethod
    def generate_energy_dict(energy0):

        energy = {}
        e_tot = 0.0
        for x in energy0:
            tag, value = x
            e_tot += value
            for key in ["Nonbonded", "Bond", "Angle", "Torsion"]:
                if key in tag:
                    energy[key.lower()] = value
        energy["total"] = e_tot
        return energy

    @staticmethod
    def guess_atomic_number(atomname):
        from parmed.periodic_table import AtomicNum

        sym = "".join([i for i in atomname if not i.isdigit()])
        sym.upper()
        if len(sym) > 1:
            sym = "%s%s" % (sym[0], sym[1].lower())
        return AtomicNum[sym]

    def set_charges(self, charges):
        if len(self.atoms) != len(charges):
            print("Error bad number of charge", len(self.atoms), len(charges))
            raise
        for atom, q in zip(self.atoms, charges):
            atom.charge = q  # in prmtop, q is multiplied by 18.2223 to convert kcal/mol

    def neutralize_charge(self):
        qs = [atom.charge for atom in self.atoms]
        delta = sum(qs) / len(qs)
        for atom in self.atoms:
            atom.charge -= delta

    @staticmethod
    def gen_type2id(typedict):
        dct = {}
        for i, (_k, v) in enumerate(typedict.items(), 1):
            if type(v) == list:
                dct[tuple(v)] = i
            else:
                dct[v] = i
        return dct

    def update_multi_mol(self, atoms):
        raise

    def update(self, atoms, boxsize_check=False):
        """
        generate upscaled structure for ex. super cell

        Args:
            atoms: supercell in Ase Atoms object
            boxsize_check: check box or not
        """

        natom = len(self.atoms)
        natom_new = len(atoms.get_positions())
        mod = natom_new % natom
        if mod != 0:
            raise ValueError(f"bad number of atoms, old:{natom}, new{natom_new}")

        #if len(self.residues) > 1 and natom != natom_new:
        #    print("not support multi residue and diffrent number of atom case")
        #    raise ValueError

        mul = int(natom_new / natom)
        symbols = self.to_ase().get_chemical_symbols()
        newsymbol = atoms.get_chemical_symbols()[:natom]

        if symbols != newsymbol:
            raise ValueError(f"bad element order, old:{symbols}, new: {newsymbol}")
            #update the sequence
            #For a supercell

        if mul > 1:
            self *= mul

        self.coordinates = atoms.get_positions()
        newbox = atoms.cell.cellpar()
        if boxsize_check:
            self.boxsize_check(newbox)
        self.box = newbox

    def boxsize_check(self, box):
        """
        check box by cutoff
        input:
            box: [a,b,c,alpha,beta,gamma]
        """
        a, b, c, *_ = box
        if a < self.cutoff * 2 or b < self.cutoff * 2 or c < self.cutoff * 2:
            minl = min([a, b, c])
            print(f"Error: box size is too small, cutoff {self.cutoff}*2 < min box size {minl}")
            raise ValueError

    def coordinates_shift(self, shift):
        tmp = self.get_coordinates()
        tmp += shift
        self.coordinates = tmp

    def write_aseio(self, *args, **kwargs):
        self.to_ase().write(*args, **kwargs)


def amber_to_pdstruc(prmtop, inpcrd, base="ff", cls=ParmEdStructure):
    """
    Convert AMBER prmtop and inpcrd files to dictionary of forcefield parameters as openmm format.
    """
    parmedstruc = cls.from_structure(AmberParm(str(prmtop), str(inpcrd)))
    return parmedstruc


def ommxml_to_pdstruc(fxml, fpdb, base="ff", cls=ParmEdStructure):
    parmedstruc = cls.from_xml_pdb(str(fxml), str(fpdb))
    return parmedstruc


def parmedstrc_to_ffdic(parmedstruc, base="ff", writemol2=True, writexml=True):
    ffs = []
    mol2s = []
    for i, (s, _items) in enumerate(parmedstruc.split()):
        if writemol2:
            s.save(base + str(i) + ".mol2", format="mol2", mol3=True, overwrite=True)

        rescon = ResidueTemplateContainer.from_structure(s)
        ommps = OpenMMParameterSet.from_structure(s)
        for res in rescon:
            ommps.residues[res.name] = res
        if writexml:
            ommps.write(base + str(i) + ".xml", provenance={"Info": "OMM XML"}, improper_dihedrals_ordering="amber")
        sio = StringIO()
        ommps.write(sio, provenance={"Info": "OMM XML"}, improper_dihedrals_ordering="amber")
        ffs.append(parse(sio.getvalue()))

        sio = StringIO()
        s.save(sio, format="mol2", mol3=True, overwrite=True)
        mol2s.append(sio.getvalue())

    ffdic = {}
    ffdic["omm_forcefield"] = ffs
    ffdic["mol2"] = mol2s

    return ffdic


def ommffs_to_paramedstruc(ommffs, mol2strs, cls=ParmEdStructure, cleanup=True):

    pdstruc = None
    with temporary_directory_change(cleanup=cleanup) as _:
        if type(mol2strs) is list:
            for i, (mol2str, ommff) in enumerate(zip(mol2strs, ommffs)):
                base = f"tmp_{i}"
                open(f"{base}.mol2", "w").write(mol2str)
                xmlstr = dict_to_xmlstr(ommff)
                open(f"{base}.xml", "w").write(xmlstr)
                s = cls.from_xml_mol2(f"{base}.xml", f"{base}.mol2")
                if pdstruc is None:
                    pdstruc = s
                else:
                    pdstruc += s
        else:
            open("tmp.mol2", "w").write(mol2strs)
            fxmls = []
            for i, ommff in enumerate(ommffs):
                fxmls.append(f"tmp_{i}.xml")
                xmlstr = dict_to_xmlstr(ommff)
                open(fxmls[-1], "w").write(xmlstr)
            pdstruc = cls.from_xml_mol2(fxmls, f"{base}.mol2")
    return pdstruc
