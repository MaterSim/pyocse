import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pyxtal import pyxtal
from pyxtal.db import database
from pyxtal.util import parse_cif #, new_struc
from ost.parameters import ForceFieldParameters
from htocsp.common import new_struc
from pymatgen.core import Structure


reference_file = 'reference.xml'
parameters_init_file = 'parameters_init.xml'
parameters_opt_file = 'parameters_opt.xml'


db = database("../../../HT-OCSP/benchmarks/Si.db")
xtal = db.get_pyxtal("WAHJEW")
smiles = [mol.smile for mol in xtal.molecules]
assert smiles[0] is not None

# Initialize the param instance
params = ForceFieldParameters(smiles, style='openff', f_coef=0.1); print(params)
if os.path.exists(parameters_init_file):
    parameters0 = params.load_parameters(parameters_init_file)
else:
    parameters0 = params.params_init.copy()
    params.export_parameters(parameters_init_file)

# Get the reference configurations
if os.path.exists(reference_file):
    ref_dics = params.load_references(reference_file)
else:
    smi = smiles[0]
    strs, engs = parse_cif('Ranked-pyxtal-WAHJEW.cif', eng=True)
    ids = np.argsort(engs)
    ref_dics = []
    for i, id in enumerate(ids[:3]): 
        pmg = Structure.from_str(strs[id], fmt='cif') 
        c0 = pyxtal(molecular=True)
        c0.from_seed(pmg, molecules=[smi+'.smi'])
    
        ase = c0.to_ase(resort=False)
        dics = params.augment_ref_configurations(ase, 
                                                 steps=120, 
                                                 N_vibration=5)
        ref_dics.extend(dics)
    params.export_references(ref_dics, filename=reference_file)

# FF parameters optimization
if os.path.exists(parameters_opt_file):
    parameters_opt = params.load_parameters(parameters_opt_file)
else:
    parameters_opt = None

for i in range(3):
    for data in [(['bond'], 250),
                 (['angle'], 250),
                 (['bond', 'angle'], 250),
                 #(['proper'], 150),
                 #(['bond', 'angle', 'proper'], 150),
                 #(['vdW'], 100),
                 #(['proper', 'vdW'], 100),
                 #(['charge'], 100),
                 #(['vdW', 'charge'], 100),
                 #(['proper', 'vdW', 'charge'], 100),
                 #(['bond', 'angle', 'proper', 'vdW', 'charge'], 100),
                ]:
        (terms, steps) = data
        t0 = time()
        # Get the optimized energy offset
        offset_opt = params.optimize_offset(ref_dics)

        # Conduct actual optimization
        opt_dict = params.get_opt_dict(terms, 
                                       parameters=parameters_opt)
        print('Init values\n', opt_dict)
        x, fun, values = params.optimize(ref_dics,
                                         offset_opt,
                                         opt_dict,
                                         parameters_opt,
                                         steps = steps,
                                         debug = True)
        parameters_opt = params.set_sub_parameters(values, 
                                                   terms, 
                                                   parameters_opt)
        print('Opt_obj {:.2f} min '.format((time()-t0)/60), terms, fun)
        print('Opt_solution\n', x)
        print('Opt_values\n', values)
        if os.path.exists('lmp.log'): os.remove('lmp.log')

# Save the Final parameters
params.export_parameters(parameters_opt_file, parameters_opt)


# Results analysis
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
params.plot_ff_results(axes[0], parameters0, ref_dics, label='Init')
params.plot_ff_results(axes[1], parameters_opt, ref_dics, label='Opt')
plt.savefig('Results.png')
print("export results in Results.png\n")

grid_size = (5, 2)
fig = plt.figure(figsize=(10, 16))
for i, term in enumerate(['bond', 'angle', 'proper', 'vdW', 'charge']):
    if term == 'charge':
        ax = plt.subplot2grid(grid_size, (i, 0), colspan=2, fig=fig)
        params.plot_ff_parameters(ax, parameters0, parameters_opt, term)
    else:
        ax1 = plt.subplot2grid(grid_size, (i, 0), fig=fig)
        ax2 = plt.subplot2grid(grid_size, (i, 1), fig=fig)
        params.plot_ff_parameters(ax1, parameters0, parameters_opt, term+'-1')
        params.plot_ff_parameters(ax2, parameters0, parameters_opt, term+'-2')
plt.savefig('parameters.png')
print("export results in parameters.png\n")
