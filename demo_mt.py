from ost.build import Builder
from pyxtal.db import database
import numpy as np
import os, sys
from optparse import OptionParser

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-d", "--db", dest="db",
                      help="database name, required",
                      metavar="db")

    parser.add_option("-c", "--code", dest="code",
                      help="code, required",
                      metavar="code")

    parser.add_option("-p", "--per", dest="per",
                      help="cpu pernodes, default 48",
                      default=48,
                      metavar="per")

    parser.add_option("-n", "--nodes", dest="nodes",
                      help="number of nodes, default: 1",
                      type=int,
                      default=1,
                      metavar="nodes")

    parser.add_option("-l", "--lammps", dest="lammps",
                      help="lammps command",
                      metavar="lammps")


    # A pipeline to perform sceening of Organic crytals with possible Martensitic Transition
    # 1. extract crystal
    # 2. perform one cycle of tension-compress, shear deformation
    # For each cycle:
    # 2.1 choose the direction and create supercell
    # 2.2 generate lammmps file (lmp.dat, lmp.in, deform.in)
    # 2.3 launch lammps simulation
    # 2.4 Generate output files (Stress/Strain curve, Energy/Strain curve)
    # 2.5 Optional, scripts to generate movies
    
    # Set the crystal model
    (options, args) = parser.parse_args()

    matrix = np.eye(3)
    style = 'openff' #'gaff'
    db = database(options.db)
    ncpu = options.per * options.nodes
    code = options.code
    _lmpcmd = options.lammps.split('srun')[-1]

    # in case the lmpcmd has -n 
    if len(_lmpcmd.split('-n')) > 1:
        tmps = _lmpcmd.split('-n')
        _lmpcmd = tmps[0] + ' '.join(tmps[-1].split()[1:])

    lmpcmd = "srun -n " + str(ncpu) + ' ' + _lmpcmd
    print(code, lmpcmd)

    # Extract information
    directions = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
    xtal = db.get_pyxtal(code)
    smiles = [mol.smile for mol in xtal.molecules]
    bu = Builder(smiles=smiles, style=style)
    bu.set_xtal(xtal, para_min=10.0)
    
    # Setup Directory
    compound_folder = 'MT-' + code + '-' + style
    if not os.path.exists(compound_folder): os.makedirs(compound_folder)
    os.chdir(compound_folder)

    # Create the lammps file for the unit cell
    bu.set_slab(bu.xtal, bu.xtal_mol_list, matrix=matrix)
    bu.lammps_slab.write_lammps()
    
    for direction in directions:
        print("direction", direction)
        deform_folder = direction
        if not os.path.exists(deform_folder): os.makedirs(deform_folder)
    
        os.chdir(deform_folder)

        # Get supercell
        dim = bu.get_dim(direction)
        print('Supercell:  ', bu.ase_slab.get_cell_lengths_and_angles())
        [a, b, c] = bu.ase_slab.get_cell_lengths_and_angles()[:3]
        replicate = [int(np.ceil(dim[i]/l)) for i, l in enumerate([a, b, c])]

        # Create lammps master input
        task = {
                'type': 'cycle',
                'direction': direction,
                'temperature': 300,
                'pressure': 1.0,
                'max_strain': 0.3,
                'rate': 2e+8, 
                'dump_steps': 50,
                'replicate': replicate,
                }
        bu.set_task(task)
        os.system('cp ../lmp.in ./')
        os.system('cp ../lmp.dat ./')

        # Execute the calculation
        os.system(lmpcmd)
        try:
            bu.plot(direction)
        except:
            print('Fail to make plot', code, direction)
        os.chdir('../')
    
