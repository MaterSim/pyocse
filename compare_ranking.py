from pyocse.parameters import ForceFieldParameters
from pyxtal.db import database
import os

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
ref_dics = params.load_references('dataset/references.xml')[:100]
params.write_lmp_dat_from_ref_dics(ref_dics)
params_opt = params.optimize_offset(ref_dics, parameters0=p0)
errs = params.plot_ff_results("performance.png",
                       [ref_dics, ref_dics0],
                       [params_opt])
print(errs)
# Find the ranking of the reference structures
