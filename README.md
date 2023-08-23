# Organic Simulation Toolkit (OST)

This is a public repository that aims to automate the simulation of organic crystals with a primary emphasis on the mechanical properties of organic crystals. Currently, we focus on two components:

- 1. Automate the geneation of structural model and force field (through `ambertools` or `openff`)
- 2. Design different workflows to simulate the properties of organic crystals under mechanical loads (powered by `lammps`)

# Python Setup
go to the directory
```
conda install -c conda-forge mamba
mamba env create -n ost python=3.9
conda activate ost
pip install .
```

## LAMMPS Setup:
```
make yes-MOLECULE
make yes-EXTRA-MOLECULE 
make yes-KSPACE 
make mpi -j 12
```

## Examples

In a recent


## Contacts:

- Qiang Zhu (qzhu8@uncc.edu)
- Shinnosuke Hattori (shinnosuke.hattori@gmail.com)

3D periodic boundary condictions
- shearing
- tensile/compression

2D PBC
- bending

no PBC
- free bending

