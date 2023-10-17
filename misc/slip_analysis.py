from collections import deque
from ase.neighborlist import NeighborList
from ase.io.lammpsrun import get_max_index, construct_cell, lammps_data_to_ase_atoms
from pyxtal.descriptor import _qlm
import numpy as np
from ase.atoms import Atoms

def read_xyz(filename, cellname=None):
    fileobj = open(filename)
    lines = fileobj.readlines()
    fileobj.close()
    images = []
    while len(lines) > 0:
        symbols = []
        positions = []
        natoms = int(lines.pop(0))
        lines.pop(0)  # Comment line; ignored
        for _ in range(natoms):
            line = lines.pop(0)
            symbol, x, y, z = line.split()[:4]
            symbol = symbol.lower().capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        images.append(Atoms(symbols=symbols, positions=positions))

    # update the cell vectors
    if cellname is not None:
        fileobj = open(cellname)
        lines = deque(fileobj.readlines())
        fileobj.close()
        cells = []

        while len(lines) > 0:
            line = lines.popleft()
            if 'Lattice vectors (A)' in line:
                celldatarows = [lines.popleft() for _ in range(3)]
                cells.append(np.loadtxt(celldatarows))

        if len(cells) <= len(images):
            images = images[:len(cells)]
            for mat, struc in zip(cells, images):
                struc.set_pbc([1, 1, 1])
                struc.set_cell(mat)
                #print(struc)
        else:
            print(len(cells), len(images))
            raise RuntimeError('Number of structure is inconsistent')

    return images

strucs = read_xyz(f+'geo_final.xyz', f+'md.out')
