from ost.build import Builder
from pyxtal.db import database
import numpy as np
import os

# Set the crystal model
data = [
        ('ANLINB02', [[0,1,0], [0,0,1], [1,0,0]]),
       ]

#style = 'gaff' #'openff'
name = 'twin'
style = 'openff'
db = database('dataset/mech.db')
dim = [100, 20, 50]
vertices = [(0.4, 0), (0.6, 0), (0.8, 1), (1.0, 1)]
angle = 30

for d in data:
    (code, matrix) = d
    matrix = np.array(matrix)
    print(code)
    xtal = db.get_pyxtal(code)
    smiles = [mol.smile for mol in xtal.molecules]
    bu = Builder(smiles=smiles, style=style)

    #for angle in (0, 30, 60, 90):
    # Get the relaxed cell paramters?
    bu.set_xtal(xtal, para_min=10.0)
    
    # Apply the orientation
    print('Unitcell:   ', bu.xtal.get_cell_lengths_and_angles())
    print('Matrix:     ', matrix)

    bu.set_slab(bu.xtal, bu.xtal_mol_list, matrix=matrix, dim=dim, vacuum=50)
    rotation = (np.array([0, 1, 0]), angle)
    atoms = bu.rotate_molecules(bu.ase_slab, bu.ase_slab_mol_list, vertices, rotation, ids=[0, 2])

    # Directory
    folder = name + '-' +code + '-' + style
    if not os.path.exists(folder): os.makedirs(folder)
    cwd = os.getcwd()
    os.chdir(folder)

    # Example 1: tensile, assuming z direction
    task1 = {'type': 'tensile',
             'temperature': 200,
             'pressure': 1.0,
             'max_strain': 0.2,
             'rate': 2e+8,
            }
    print('Supercell:  ', bu.ase_slab.get_cell_lengths_and_angles())
    bu.set_task(task1)
    atoms.write(str(angle)+'.xyz', format='extxyz')
    # Add lammps ff information
    lammps_slab = bu.get_ase_lammps(atoms)
    lammps_slab.write_lammps()

    os.chdir(cwd)
