# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ost.lmp.calculator import LAMMPSCalculator
from pymatgen.core import Structure
from pyxtal import pyxtal
from pyxtal.db import database
from pyxtal.util import parse_cif

# sys.path.insert(0, "/usr/lib/python3/dist-packages")

# Get pyxtal
db = database("../../benchmarks/Si.db")
xtal_init: pyxtal = db.get_pyxtal("WAHJEW")
smi = "C[Si]1(O)O[Si](C)(O)O[Si](C)(O)O[Si](C)(O)O1"
rank_strs, engs = parse_cif("Ranked-pyxtal-WAHJEW.cif", eng=True)
q_WAHJEW = np.loadtxt("q_WAHJEW.txt")[:, 3]
sioring_id = [1, 3, 4, 7, 8, 11, 12, 15]
print(q_WAHJEW)


# %%


def get_energy(xtal, dir_name, qs=None):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    # print(xtal)  # ; import sys; sys.exit()
    if xtal.has_special_site():
        xtal = xtal.to_subgroup()

    xtal.to_file("init.cif")
    struc = xtal.get_forcefield(code="lammps", chargemethod="gasteiger")
    if qs is not None:
        for j, atom in enumerate(struc.atoms):
            atom.charge = qs[j % len(qs)]
    # manual
    # struc.angle_types[3].theteq = 149.0

    struc.write_lammps()
    lammps_calc = LAMMPSCalculator(struc)
    lammps_calc.minimize()
    energy = lammps_calc.get_energy()
    print(dir_name, energy)
    os.chdir("..")

    return energy


# %%
ret = {}
qfacts = [1.0, 0.8, 0.6, 0.4]
maxrank = 5
for qfact in qfacts:
    qs = [q * qfact if i in sioring_id else q for i, q in enumerate(q_WAHJEW)]
    dir_name = "ref_qf" + str(qfact)
    ret[dir_name] = get_energy(xtal_init, dir_name, qs=qs)
    for i in range(1, maxrank + 1, 1):
        # setup simulation
        dir_name = "rank" + str(i) + "_qf" + str(qfact)
        pmg = Structure.from_str(rank_strs[i - 1], fmt="cif")
        xtal = pyxtal(molecular=True)
        xtal.from_seed(pmg, molecules=[smi + ".smi"])
        ret[dir_name] = get_energy(xtal, dir_name, qs=qs)


# %%
# plot the results for each subplot
key = "coul"
df0 = pd.DataFrame(ret)

df = df0.loc[key]
dplot = {}
for qfact in qfacts:
    for i in range(1, maxrank + 1, 1):
        dplot["rank" + str(i) + "_qf" + str(qfact)] = (
            df["rank" + str(i) + "_qf" + str(qfact)] - df["ref_qf" + str(qfact)]
        )

fig, ax = plt.subplots(figsize=(8, 6))
fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
ax.bar(dplot.keys(), dplot.values(), color="blue")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("Energy - E_ref (kcal/mol)")
ax.set_xlabel("Rank")
ax.set_title(key + " Energy of WAHJEW")
plt.tight_layout()
plt.savefig("WAHJEW.png", dpi=300)
plt.show()


# struc = xtal.get_forcefield(code="lammps", chargemethod="gasteiger")
# calc = CHARMM(xtal, prm="charmm.prm", rtf="charmm.rtf", atom_info=atom_info)
# print(calc.structure.lattice)
# calc.run(clean=False)
# print(calc.structure.energy)
# print(calc.structure.lattice)
# calc.structure.to_file("opt.cif")

# %%
