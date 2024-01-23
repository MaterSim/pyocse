import os
from mace.calculators import mace_mp
from mace.calculators import MACECalculator
from ase import Atoms, units
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ost.lmp.calculator import LAMMPSCalculator
from pymatgen.core import Structure
from pyxtal import pyxtal
from pyxtal.db import database
from pyxtal.util import parse_cif

device="cuda:0"

#calculator = MACECalculator(
#    #model_paths=/path/to/pretrained.model,
#    model_paths="2023-12-10-mace-128-L0_epoch-199.model",
#    dispersion=True,
#    device=device,
#    default_dtype="float32" or "float64",
#)
calculator = mace_mp(model="small", dispersion=True, default_dtype="float32" or "float64", device=device)




# sys.path.insert(0, "/usr/lib/python3/dist-packages")

# Get pyxtal
db = database("../../benchmarks/Si.db")
xtal_init: pyxtal = db.get_pyxtal("WAHJEW")
smi = "C[Si]1(O)O[Si](C)(O)O[Si](C)(O)O[Si](C)(O)O1"
rank_strs, engs = parse_cif("Ranked-pyxtal-WAHJEW.cif", eng=True)


# %%


def get_energy(xtal, dir_name, step=500, fmax=0.1):

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    struc = xtal.to_ase()
    struc.calc = calculator
    energy0 = struc.get_potential_energy()
    struc.set_constraint(FixSymmetry(struc))
    ecf = ExpCellFilter(struc)
    dyn = FIRE(ecf, a=0.1)
    dyn.run(fmax=fmax, steps=step)
    energy = struc.get_potential_energy()

    print(dir_name, energy0, energy)
    os.chdir("..")

    return energy


# %%
maxrank = 50
ret = {}
dir_name = "ref"
ret[dir_name] = get_energy(xtal_init, dir_name)
for i in range(1, maxrank + 1, 1):
    # setup simulation
    dir_name = "rank" + str(i)
    pmg = Structure.from_str(rank_strs[i - 1], fmt="cif")
    xtal = pyxtal(molecular=True)
    xtal.from_seed(pmg, molecules=[smi + ".smi"])
    ret[dir_name] = get_energy(xtal, dir_name)


