# Python Organic Crystal Simulation Environment (PYOCSE)

This is a public repository that aims to automate the simulation of organic crystals with a primary emphasis on the mechanical properties of organic crystals. Currently, we focus on two components:

1. Automate the geneation of structural model and force field (through `ambertools` or `openff`)
2. Design different workflows to simulate the properties of organic crystals under mechanical loads (powered by `lammps`)

## Python Setup
git clone this repository and then go to the root directory

```
conda install -c conda-forge mamba
mamba env create -n ocse
conda activate osce
pip install .
```

*If you want to update the existing ocse enviroment*

```
conda activate ocse
mamba env update --file environment.yml
```

## LAMMPS Setup:
```
make yes-MOLECULE
make yes-EXTRA-MOLECULE 
make yes-KSPACE 
make mpi -j 12
```

## Examples

### 3D periodic boundary condictions (shearing/tensile/compression)

check the `uniaxial.py`


### 2D PBC bending

check the `3pf.py`

### no PBC free bending (to add)


## Contacts:

- Qiang Zhu (qzhu8@uncc.edu)
- Shinnosuke Hattori (shinnosuke.hattori@sony.com)
