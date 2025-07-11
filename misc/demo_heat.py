from pyocse.build import Builder
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
    # A pipeline to perform sceening of Organic crytals with possible Martensitic Transition
    # 1. extract crystal
    # 2. create crystal model in different setting (matrix)
    # 3. generate lammmps file (lmp.dat, lmp.in, deform.in)

    # Set the crystal model
    (options, args) = parser.parse_args()
    matrix = np.array([[1,1,2],[1,-1,0],[4,3,-1]])
    style = 'openff' #'gaff'
    db = database(options.db)
    code = options.code

    #directions = ['xy', 'xz', 'yz']
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
