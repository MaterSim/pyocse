from pyocse.build import Builder
from pyxtal.db import database
import numpy as np
import os

# Set the crystal model
data = [
        #('BIYRIM01', [[1,0,0], [0,1,0], [0,0,1]]),
        ('ANLINB02', [[1,0,0], [0,1,0], [0,0,1]]),
       ]
style = 'openff' #'gaff'
db = database('dataset/mech.db')
dim = [100, 240, 40]


# Prepare the lammps input files
for d in data:
    (code, matrix) = d
    matrix = np.array(matrix)
    print(code)
    xtal = db.get_pyxtal(code)
    smiles = [mol.smile for mol in xtal.molecules]
    bu = Builder(smiles=smiles, style=style)
    bu.set_xtal(xtal, para_min=10.0)

    # Directory
    folder = code + '-' + style
    if not os.path.exists(folder): os.makedirs(folder)
    cwd = os.getcwd()
    os.chdir(folder)

    # Example 1: tensile, assuming z direction
    task1 = {'type': 'tensile',
             'temperature': 300,
             'pressure': 1.0,
             'max_strain': 0.1,
             'rate': 1e+8,
             }

    bu.set_slab(bu.xtal, bu.xtal_mol_list, matrix=matrix, dim=dim)#, reset=False)
    print('Supercell:  ', bu.ase_slab.get_cell_lengths_and_angles())
    bu.set_task(task1)

    bu.lammps_slab.write_lammps()
    bu.ase_slab.write('test.xyz', format='extxyz')
    os.chdir(cwd)
