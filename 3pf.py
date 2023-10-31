from ost.build import Builder
from pyxtal.db import database
import numpy as np
import os

# Set the crystal model
data = [
        #('BIYRIM01', [[0,0,1], [0,-1,0], [1,0,0]]), # Elastic, 90.124
        #('DAHLOQ', [[0,0,1], [0,-1,0], [1,0,0]]), # Brittle
        ('DAHMUX', [[1,0,0], [0,1,0], [0,0,1]]), # Plastic, 93.99, ac
        #('YEWYAD', [[0,0,1], [1,0,0], [0,1,0]]), # Brittle, 98.902, ab
       ]

#style = 'gaff' #'openff'
style = 'openff'
db = database('dataset/mech.db')
dim = [250, 20, 40]
#dim = [100, 20, 20]

for d in data:
    (code, matrix) = d
    matrix = np.array(matrix)
    print(code)
    xtal = db.get_pyxtal(code)
    smiles = [mol.smile for mol in xtal.molecules]
    bu = Builder(smiles=smiles, style=style)

    # Get the relaxed cell paramters?
    bu.set_xtal(xtal)#, para_min=10.0)
    
    # Apply the orientation
    print('Unitcell:   ', bu.xtal.get_cell_lengths_and_angles())
    print('Matrix:     ', matrix)

    # Directory
    folder = '3pf-TT'+code+'-'+style
    if not os.path.exists(folder): os.makedirs(folder)
    cwd = os.getcwd()
    os.chdir(folder)

    # 3pf bending
    task = {'mode': 'bend',              # uni_xx_bulk, 3pf_xz_slab0
            'pbc_x': False,
            'type': 'single',           #
            'indenter_distance': 100.0, # A
            'inderter_t_hold': 300.0,   # ps, timesteps
            'indenter_rate': 1e-4,      # A/fs (10 m/s)
            'dump_steps': 50,           # 
            'indenter_radius': 30.0,   # Ang
            'indenter_buffer': 10.0,   # Ang
            'indenter_k': 1.0,         # eV/^3
            'timerelax': 1000,       # fs
           }

    bu.set_slab(bu.xtal, bu.xtal_mol_list, matrix=matrix, dim=dim, 
                vacuum = task['indenter_distance'] * 1.2, #make sure the z-axis is long
                separation = task['indenter_radius'] + task['indenter_buffer'], 
                orthogonality=True)
    print('Supercell:  ', bu.ase_slab.get_cell_lengths_and_angles())

    # Get the molecular ids
    bord_ids = bu.get_molecular_bord_ids(bu.ase_slab, bu.ase_slab_mol_list, axis=0)
    fix_ids = bu.get_molecular_fix_ids(bu.ase_slab, bu.ase_slab_mol_list, axis=0)
    print('border molecules', bord_ids)
    print('fix molecules', fix_ids)

    zs = bu.ase_slab.get_positions()[:,2]
    z_max, z_min = zs.max(), zs.min()
    task['indenter_height'] = z_max + task['indenter_radius']
    task['border_mols'] = bord_ids   # List of [a, b] 
    task['fix_mols'] = fix_ids       # Number of molecules per column 
    bu.set_task(task)
    
    # update x_lo/x_hi
    if not task['pbc_x']:
        padding = [100, 0, 0]
    else:
        padding = [0, 0, 0]
    bu.lammps_slab.write_lammps(orthogonality=True, padding=padding)
    bu.dump_slab_centers(bu.ase_slab, bu.ase_slab_mol_list, bord_ids, fix_ids)
    #bu.ase_slab.write('test.xyz', format='extxyz')
    os.chdir(cwd)
