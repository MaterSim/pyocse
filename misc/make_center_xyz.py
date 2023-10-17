import os, sys
from collections import deque
import numpy as np
from ase.io.lammpsrun import construct_cell, lammps_data_to_ase_atoms
from ase.atoms import Atoms
from ase.io import write

if len(sys.argv)<2:
    raise RuntimeError('Needs to provide path')

path = sys.argv[1]
os.chdir(path)

with open('center.dat', 'r') as f: lines = deque(f.readlines())
box = np.loadtxt('box_parameters.dat')

images = []
count = 0
line = lines.popleft()#; print(line)
line = lines.popleft()#; print(line)
line = lines.popleft()#; print(line)
n_atoms = 0

while len(lines) > n_atoms:
    line = lines.popleft(); #print(line)
    tmp = line.split(); #print(tmp)
    n_atoms, step = int(tmp[1]), int(tmp[0])
    datarow = ['6 '+ lines.popleft() for _ in range(n_atoms)]
    xyz = np.loadtxt(datarow, dtype=str)[1:]
    [xhi, xlo, yhi, ylo, zhi, zlo, xy, xz, yz] = box[count]
    xlo += min([0, xy, xz, xy+xz])
    xhi += max([0, xy, xz, xy+xz])
    ylo += min([0, yz])
    yhi += max([0, yz])
    diagdisp = [xlo, xhi, ylo, yhi, zlo, zhi]
    offdiag = [xy, xz, yz]
    cell, celldisp = construct_cell(diagdisp, offdiag)
    atoms = lammps_data_to_ase_atoms(
            data=xyz,
            colnames=['type', 'id', 'x', 'y', 'z'],
            cell=cell,
            celldisp=celldisp,
            atomsobj=Atoms,
            pbc=[1, 1, 1],
            )
    print(count, atoms.cell.cellpar())#; import sys; sys.exit()
    images.append(atoms)
    count += 1
#print("Complete", path+'/center.xyz')
write('center.xyz', images=images, format='extxyz')
