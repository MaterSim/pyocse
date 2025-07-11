# Python Organic Crystal Simulation Environment (PYOCSE)

This is a public repository that aims to automate the simulation of organic crystals with a primary emphasis on the mechanical properties of organic crystals. Currently, we focus on two components:

1. Automate the geneation of structural model and force field (through `ambertools` or `openff`)
2. Design different workflows to simulate the properties of organic crystals under mechanical loads (powered by `lammps`)

## Python Setup
git clone this repository and then go to the root directory

```
conda env create -n pyocse
conda activate pyocse
pip install .
```

*If you want to update the existing ocse enviroment*

```
conda activate pyocse
conda env update --file environment.yml
```

## LAMMPS Setup:
```
make yes-MOLECULE
make yes-EXTRA-MOLECULE 
make yes-KSPACE 
make mpi -j 12
```

## Examples

1. Add your structure to the database

```python
# Find your structure from https://www.ccdc.cam.ac.uk/structures/Search?
tag = {
       "csd_code": 'ACSALA',
       "ccdc_number": 1101020,
       "smiles": "CC(=O)OC1=CC=CC=C1C(O)=O",
}

# Load the pyxtal structure
from pyxtal import pyxtal
xtal = pyxtal(molecular=True)
xtal.from_seed(str(tag['ccdc_number'])+'.cif',
               molecules=[str(tag['smiles'])+'.smi'])
xtal.tag = tag
print(xtal)

# Deposit your structure
from pyxtal.db import make_entry_from_pyxtal, database
entry = make_entry_from_pyxtal(xtal)
db = database('dataset/mech.db')
db.add(entry)
```

2. 3D periodic boundary condictions (shearing/tensile/compression)


```python
from pyocse.build import Builder
from pyxtal.db import database
import numpy as np
import os

# Set the crystal model
data = [
        ('ACSALA', [[1,0,0], [0,1,0], [0,0,1]]),
       ]
style = 'openff' #'gaff'
db = database('dataset/mech.db')

# Define your desired dimension 
dim = [100, 40, 40]

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

    bu.set_slab(bu.xtal, bu.xtal_mol_list, matrix=matrix, dim=dim)
    print('Supercell:  ', bu.ase_slab.get_cell_lengths_and_angles())
    bu.set_task(task1)

    bu.lammps_slab.write_lammps()
    # Just for a quick view from VESTA
    bu.ase_slab.write('test.xyz', format='extxyz')
    os.chdir(cwd)
```

You can check the details in `misc/uniaxial.py`

## Contacts:

- Qiang Zhu (qzhu8@uncc.edu)
- Shinnosuke Hattori (shinnosuke.hattori@sony.com)
