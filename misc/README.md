Return to the OST root directory 

# Check the available list of compounds 

$ ase db dataset/mech.db -c csd_code -L 40

# Submit new calculations
$ sbatch -J BIYRIM01 misc/myrun-anvil

# To view the results
$ python misc/figure.py

Will generate a file called `total.png` on the root directory

# To get xyz file for a particular calculation
$ python misc/make_center_xyz.py MT-TRPHAM01-openff/xy

will generate `ceter.xyz` which you can visualize in ovito.
