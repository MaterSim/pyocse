from ost.build import Builder
from pyxtal.db import database
import numpy as np
import os

# Set the crystal model
data = [
        ('BIYRIM01', [[0,0,1], [0,-1,0], [1,0,0]]), # Elastic, 90.124
        #('DAHLOQ', [[0,0,1], [0,-1,0], [1,0,0]]), # Brittle
        #('DAHMUX', [[0,0,1], [0,-1,0], [1,0,0]]), # Plastic, 93.99, ac
        #('YEWYAD', [[0,0,1], [1,0,0], [0,1,0]]), # Brittle, 98.902, ab
       ]

#style = 'gaff' #'openff'
style = 'openff'
db = database('dataset/mech.db')
dim = [300, 60, 60]

for d in data:
    (code, matrix) = d
    matrix = np.array(matrix)
    print(code)
    xtal = db.get_pyxtal(code)
    smiles = [mol.smile for mol in xtal.molecules]
    bu = Builder(smiles=smiles, style=style)

    # Get the relaxed cell paramters?
    bu.set_xtal(xtal, para_min=10.0)
    
    # Apply the orientation
    print('Unitcell:   ', bu.xtal.get_cell_lengths_and_angles())
    print('Matrix:     ', matrix)

    # Directory
    folder = '3pf-'+code+'-'+style
    if not os.path.exists(folder): os.makedirs(folder)
    cwd = os.getcwd()
    os.chdir(folder)

    # 3pf bending
    task = {'type': '3pf',
            'temperature': 300.0,      # K
            'pressure': 1.0,           # atmospheres 
            'indenter_rate': 1e-4,     # A/fs (10 m/s)
            'indenter_radius': 30.0,   # A
            'indenter_distance': 100.0,# A
            'indenter_buffer': 10.0,   # A
            'indenter_k': 1.0,         # eV/^3
            'inderter_t_hold': 300.0,  # ps, timesteps
            'pxatm': 0,                # atm
            'pyatm': 0,                # atm
           }

    bu.set_slab(bu.xtal, bu.xtal_mol_list, matrix=matrix, dim=dim, 
                vacuum = task['indenter_distance'] * 1.2, #make sure the z-axis is long
                separation = task['indenter_radius'] + task['indenter_buffer'], 
                orthogonality=True)
    print('Supercell:  ', bu.ase_slab.get_cell_lengths_and_angles())
    bord_ids, fix_ids = bu.get_molecular_ids(bu.ase_slab, bu.ase_slab_mol_list, width=5.0, axis=0)
    z_max = bu.ase_slab.get_positions()[:,2].max()
    z_min = bu.ase_slab.get_positions()[:,2].min()
    print('border molecules', bord_ids)
    print('fix molecules', fix_ids)

    task['indenter_height'] = z_max + task['indenter_radius']
    task['border_mols'] = bord_ids   # List of [a, b] 
    task['fix_mols'] = fix_ids       # Number of molecules per column 

    bu.set_task(task)
    
    bu.lammps_slab.write_lammps(orthogonality=True)
    bu.dump_slab_centers(bu.ase_slab, bu.ase_slab_mol_list, bord_ids, fix_ids)
    bu.ase_slab.write('test.xyz', format='extxyz')
    os.chdir(cwd)
