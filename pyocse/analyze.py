from ase.io import read
from scipy.spatial.transform import Rotation
import networkx as nx
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms
from pyxtal.molecule import pyxtal_molecule, make_graph
from pymatgen.core.structure import Molecule

def display_molecules(molecules, labels=None, size=(400,300)):
    """
    display the molecules in Pymatgen object.

    Args:
        molecules: a list of pymatgen molecules
        labels: dictionary of labels and location
        size: (width, height) in tuple

    Returns:
        py3Dmol object

    """
    import py3Dmol

    (width, height) = size
    view = py3Dmol.view(height=height, width=width)
    mol_strs = ""
    for mol in molecules:
        mol_strs += mol.to(fmt='xyz') + '\n'
    view.addModels(mol_strs, 'xyz')
    view.setStyle({'stick':{'colorscheme':'greenCarbon'}})
    for key in labels.keys():
        text, pos = key, labels[key]
        view.addLabel(text, {"position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                             "fontColor":"black",
                             "backgroundColor": "white",
                             "fontsize": 12,
                             "backgroundOpacity": "0.1",
                            })

    return view.zoomTo()


def in_list(d1, ds, numbers):
    for d in ds:
        match = True
        for i in numbers:
            if d[i] != d1[i]:
                match = False
                #print('quit', d[i], d1[i])
                break
        if match:
            return True
    return False

class lmp_mol:
    """
    Class for handling molecular crystals in lammps_dump file

    Args:
        dump_file: lammps dump file
        dat_file: lammps dat file
        smi: smiles string
    """

    def __init__(self, dump_file, dat_file):
        self.dat_file = dat_file
        self.dump_file = dump_file
        self.rotation_file = self.dump_file + '.rotation'

        #parse dat
        self.dicts, smile = self.parse_dat()
        if smile is not None: self.smile = smile #; import sys; sys.exit()
        self.p_mol = pyxtal_molecule(self.smile+'.smi')

        #parse dump
        self.struc = self.parse_dump()
        self.mol_ids = list(set(self.struc.arrays['mol_id']))
        self.n_molecules = len(self.mol_ids)
        self.n_atom_per_mol = len(self.p_mol.mol)
        self.cell = self.struc.cell[:]

    def parse_dump(self):
        """
        Read the structure from lammps dump_file
        """
        struc = read(self.dump_file, format='lammps-dump-text')
        data = np.loadtxt(self.dump_file, skiprows=9)
        ids = np.array(data[:, 0], dtype=int)
        seq = np.argsort(ids)
        mol_ids = np.array(data[:, 1], dtype=int)[seq]
        type_ids = np.array(data[:, 2], dtype=int)[seq]
        numbers = [self.dicts[type_id] for type_id in type_ids]
        struc.set_atomic_numbers(numbers)
        struc.new_array('mol_id', mol_ids)
        struc.new_array('type_id', type_ids)
        #struc.set_tags(mol_ids)
        return struc

    def new_dump(self, filename=None, multi=False):
        """
        Write the lammps dump_file with rotation information
        """
        if filename is None:
            filename = self.dump_file + '.new'
        angs = np.loadtxt(self.rotation_file)[:, 1:4]
        data = np.loadtxt(self.dump_file, skiprows=9)
        if multi:
            angs1 = np.loadtxt(self.rotation_file+'-multi')[:, 1:4]
            angs = np.append(angs, angs1, 1)

        headers = []
        with open(self.dump_file, 'r') as f:
            for i in range(8):
                tmp = f.readline()
                headers.append(tmp)
        #print('HHHH\n', headers); import sys; sys.exit()

        with open(filename, 'w') as f:
            # Write the head
            for h in headers:
                f.write(h)
            # Write the atomic information
            if multi:
                f.write('ITEM: ATOMS id mol type x y z alpha beta gamma r1 r2 r3\n')
            else:
                f.write('ITEM: ATOMS id mol type x y z alpha beta gamma\n')
            count = 0
            for i, mol_id in enumerate(self.mol_ids):
                atom = self.get_molecule(mol_id)
                for j, pos in enumerate(atom.positions):
                    f.write("{:8d} {:6d} {:2d} ".format(count+j, mol_id, j+1))
                    f.write("{:9.3f} {:9.3f} {:9.3f}".format(*pos))
                    if multi:
                        f.write("{:7.1f}{:7.1f}{:7.1f}{:7.1f}{:7.1f}{:7.1f}\n".format(*angs[i]))
                    else:
                        f.write("{:7.1f}{:7.1f}{:7.1f}\n".format(*angs[i]))
                count += len(atom)
        print("Write new dump file to {:s}".format(filename))

    def parse_dat(self):
        """
        Read the atomic type from lammps dat file
        """
        dicts = {}
        smile = None
        with open(self.dat_file) as f:
            lines = f.readlines()
            if lines[0].find('smile') > -1:
                smile = lines[0].split(':')[-1]
                smile.replace(' ', '')

            for i, l in enumerate(lines):
                if l.find('atom types') > -1:
                    # 38 atom types
                    N_types = int(l.split()[0])
                elif l.find('Mass') > -1:
                    for linenumber in range(i+2, i+2+N_types):
                        #1  12.0107800 #C1
                        #1  12.0107800 #U00c3
                        tmp = lines[linenumber].split()
                        if tmp[-1].startswith('#U00'):
                            tmp[-1] = tmp[-1][4:]
                        else:
                            tmp[-1] = tmp[-1][1:]
                        #if len(tmp[-1])>1 and tmp[-1][1].isalpha():
                        #    symbol = tmp[-1][:2]
                        #else:
                        symbol = tmp[-1][0]
                        symbol = symbol.replace(symbol[0], symbol[0].upper(), 1)
                        index = int(tmp[0])
                        #print(index, symbol)
                        dicts[index] = atomic_numbers[symbol]
                    break

        return dicts, smile

    def get_molecule(self, mol_id=1):
        """
        get the atoms object by molecular id

        Args:
            mol_id: int
        """
        ids = self.struc.arrays['mol_id'] == mol_id
        types = self.struc.arrays['type_id'][ids]
        scaled_pos = self.struc.get_scaled_positions()[ids]
        numbers = self.struc.get_atomic_numbers()[ids]
        dist = scaled_pos - scaled_pos[0]
        shift = np.round(dist)
        scaled_pos -= shift

        #resort it by type_id
        seq = np.argsort(types)
        struc = Atoms(numbers=numbers[seq],
                      scaled_positions=scaled_pos[seq],
                      cell=self.cell,
                      pbc=[1, 1, 0])
        return struc

    def get_molecules(self, mol_ids=[1]):
        """
        get the atoms object by molecular ids

        Args:
            mol_id: list of int
        """
        N_per_mol = self.n_atom_per_mol
        all_numbers = np.zeros(N_per_mol*len(mol_ids), dtype=int)
        all_scaled_positions = np.zeros([N_per_mol*len(mol_ids), 3])
        for i, mol_id in enumerate(mol_ids):
            ids = self.struc.arrays['mol_id'] == mol_id
            types = self.struc.arrays['type_id'][ids]
            scaled_pos = self.struc.get_scaled_positions()[ids]
            numbers = self.struc.get_atomic_numbers()[ids]
            dist = scaled_pos - scaled_pos[0]
            shift = np.round(dist)
            scaled_pos -= shift
            start, end = i*N_per_mol, (i+1)*N_per_mol
            #resort it by type_id
            seq = np.argsort(types)
            #print(types, start, end)
            all_scaled_positions[start:end, :] += scaled_pos[seq]
            all_numbers[start:end] += numbers[seq]
        struc = Atoms(numbers=all_numbers,
                      scaled_positions=all_scaled_positions,
                      cell=self.cell,
                      pbc=[1, 1, 0])
        return struc

    def get_orientation_between(self, mol_id1, mol_id2, opt=False):
        """
        Get Orientation between two molecules
        """
        mol0 = self.p_mol.mol
        N = len(mol0)
        atom1 = self.get_molecule(mol_id1)
        atom2 = self.get_molecule(mol_id2)
        if opt:
            mol1 = Molecule(atom1.numbers, atom1.positions)
            mol2 = Molecule(atom2.numbers, atom2.positions)
            G0 = make_graph(mol0)
            G1 = make_graph(mol1)
            G2 = make_graph(mol2)
            fun = lambda n1, n2: n1['name'] == n2['name']
            GM1 = nx.isomorphism.GraphMatcher(G0, G1, node_match=fun)
            GM2 = nx.isomorphism.GraphMatcher(G0, G2, node_match=fun)
            numbers = [i for i in mol0.atomic_numbers if i>1]
            orders1 = []
            orders2 = []
            for G in GM1.isomorphisms_iter():
                if not in_list(G, orders1, numbers):
                    orders1.append(G)
            for G in GM2.isomorphisms_iter():
                if not in_list(G, orders2, numbers):
                    orders2.append(G)
                    break

            rmsds = []
            _trans = []
            for order1 in orders1:
                for order2 in orders2:
                    o1 = [order1[at] for at in range(N)]
                    o2 = [order2[at] for at in range(N)]
                    xyz1 = mol1.cart_coords[o1]
                    xyz2 = mol2.cart_coords[o2]
                    rmsd, trans = self.p_mol.get_rmsd2(xyz1, xyz2)
                    rmsds.append(rmsd)
                    _trans.append(trans[:3, :3].T)
            rmsds = np.array(rmsds)
            r = Rotation.from_matrix(trans[np.argmin(rmsds)])
            return r.as_euler('zxy', degrees=True), rmsd
        else:
            xyz1 = atom1.positions
            xyz2 = atom2.positions
            rmsd, trans = self.p_mol.get_rmsd2(xyz1, xyz2)
            r = Rotation.from_matrix(trans[:3, :3].T)
            return r.as_euler('zxy', degrees=True), rmsd

    def get_orientation(self, mol_id, ref=None):
        """
        For the given xyz, compute the orientation

        Args:
            mol_id: index of the molecule
            ref: reference molecular positions in np.array

        Returns:
            angles: euler angles
            rmsd: root mean square deviation
        """
        if ref is None:
            xyz1 = self.p_mol.mol.cart_coords
        else:
            xyz1 = ref
        atoms = self.get_molecule(mol_id)
        xyz2 = atoms.positions
        rmsd, trans = self.p_mol.get_rmsd2(xyz1, xyz2)
        r = Rotation.from_matrix(trans[:3, :3].T)
        return r.as_euler('zxy', degrees=True), rmsd

    def show_molecules(self, mol_ids, size=(800, 600)):
        mols = []
        labels = {}
        for id in mol_ids:
            if id in self.mol_ids:
                a = self.get_molecule(id)
                mols.append(Molecule(a.numbers, a.positions))
                labels[id] = np.mean(a.positions, axis=0)
        return display_molecules(mols, labels, size=size)

    def plot_rotation(self, figname=None):
        """
        Plot the distribution of rotations
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        d = np.loadtxt(self.rotation_file)
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(nrows=3, ncols=1)
        for row in range(3):
            ax = fig.add_subplot(gs[row, 0])
            ax.hist(d[:, row+1], bins=150, density=False)
            ax.set_xlim([-180, 180])
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

    def compute_rotations(self, ref=None):
        """
        Compute the rotation information

        Args:
            ref: reference molecular xyz
        """
        if ref is None:
            ref = self.get_molecule(self.mol_ids[0]).positions

        angles, rmsds = [], []
        mol_ids = self.mol_ids

        if len(ref) == self.n_molecules * self.n_atom_per_mol:
            multi = True
            rotation_file = self.rotation_file + '-multi'
        else:
            multi = False
            rotation_file = self.rotation_file

        for i, id in enumerate(mol_ids):
            if multi:
                start = self.n_atom_per_mol * i
                end = self.n_atom_per_mol * (i+1)
                xyz = ref[start:end, :]
            else:
                xyz = ref

            angle, rmsd = self.get_orientation(id, xyz)
            angles.append(angle)
            rmsds.append(rmsd)

        print(rotation_file)
        with open(rotation_file, 'w') as f:
            for id, ang, msd in zip(mol_ids, angles, rmsds):
                f.write("{:6d} ".format(id))
                f.write("{:7.2f} {:7.2f} {:7.2f} ".format(*ang))
                f.write("{:6.3f}\n".format(msd))

def process(dat_file, xyzs, dump_file):
    tmp = dump_file.split('.')
    fname = tmp[0] + '.rotation.' + tmp[2]
    print(dump_file, fname)
    xtal = lmp_mol(dump_file, dat_file)
    xtal.compute_rotations()
    xtal.compute_rotations(ref=xyzs)
    xtal.new_dump(fname, multi=True)

if __name__ == '__main__':
    from optparse import OptionParser
    from glob import glob
    from multiprocessing import Pool
    from functools import partial

    parser = OptionParser()
    parser.add_option("-m", "--mol", dest="mol",
                      help="molecule file",
                      metavar="mol")
    parser.add_option("-p", "--pattern", dest="pattern",
                     help="pattern",
                     metavar="pattern")
    parser.add_option("-n", "--ncpu", dest="ncpu", type=int,
                     help="ncpu",
                     metavar="ncpu")

    (options, args) = parser.parse_args()
    dat_file = options.mol
    pattern = options.pattern
    dump_files = glob(pattern+'*')
    dump_files = [d for d in dump_files if d.find('rot')==-1]

    numbers = [int(d.split('.')[-1]) for d in dump_files]
    seq = np.argsort(numbers)
    dump_files = [dump_files[s] for s in seq]
    print("dumpfiles", dump_files)

    xtal0 = lmp_mol(dump_files[0], dat_file)
    xyzs = xtal0.get_molecules(xtal0.mol_ids).positions
    print(xyzs.shape)

    #xtal0.compute_rotations(ref=xyzs)
    pool = Pool(options.ncpu)
    func = partial(process, dat_file, xyzs)
    pool.map(func, dump_files)
    pool.close()
    pool.join()