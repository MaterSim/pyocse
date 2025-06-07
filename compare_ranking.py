from pyocse.parameters import ForceFieldParameters
from pyxtal.db import database
import os
import numpy as np

# Load the reference structure
db = database("dataset/ht.db")
xtal = db.get_pyxtal("ACSALA")
params = ForceFieldParameters(
        smiles=[mol.smile for mol in xtal.molecules],
        style='openff',
        ncpu=1)

if os.path.exists('dataset/references0.xml'):
    ref_dics0 = params.load_references('dataset/references0.xml')
else:
    params.set_ref_evaluator()
    ref_dics0 = params.add_references([xtal])
    params.export_references(ref_dics0, "dataset/references0.xml")

# Load parameters and references
p0, errors = params.load_parameters('dataset/parameters.xml')
ref_dics1 = params.load_references('dataset/references.xml')[:1000]
#params.write_lmp_dat_from_ref_dics(ref_dics)
#params_opt = params.optimize_offset(ref_dics, parameters0=p0)

# Get ff_dics
ff_dics0, ref_dics0 = params.evaluate_ff_references(ref_dics0, p0, update=False)
ff_dics1, ref_dics1 = params.evaluate_ff_references(ref_dics1, p0, update=False)
(ff_vals0, ref_vals0, _, _) = params.get_statistics(ff_dics0, ref_dics0)
(ff_vals1, ref_vals1, _, _) = params.get_statistics(ff_dics1, ref_dics1)
ref_eng, ff_eng = ff_vals0[0], ref_vals0[0]
ref_engs, ff_engs = ff_vals1[0], ref_vals1[0]
N = len(ff_engs)
for i in range(len(ref_eng)):
    ff_rank = np.sum(ff_eng[i] > ff_engs)
    mace_rank = np.sum(ref_eng[i] > ref_engs)
    print(f'Reference structure {i} ff   energy ranking: {ff_rank}/{N}')
    print(f'Reference structure {i} mace energy ranking: {mace_rank}/{N}')

params.plot_ff_results("performance.png",
                [ref_dics1, ref_dics0],
                [p0],
                ff_dics = [ff_dics1, ff_dics0])

