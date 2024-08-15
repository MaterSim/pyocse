import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pyxtal import pyxtal
from pyxtal.db import database
from ost.parameters import ForceFieldParameters

reference_file = 'reference.xml'
parameters_init_file = 'parameters_init.xml'
parameters_opt_file = 'parameters_opt.xml'


db = database("../../../HT-OCSP/benchmarks/Si.db")
xtal = db.get_pyxtal("WAHJEW")
smiles = [mol.smile for mol in xtal.molecules]
assert smiles[0] is not None

# Initialize the param instance
params = ForceFieldParameters(smiles, style='openff', f_coef=1.0, s_coef=1.0, ncpu=8)
print(params)

# Get the reference configurations
if os.path.exists(reference_file):
    ref_dics = params.load_references(reference_file)
else:
    t0 = time()
    ref_dics = params.add_multi_references_from_cif('Ranked-pyxtal-WAHJEW.cif',

                                                    steps=100,
                                                    N_max=8,
                                                    N_vibs=4)
    params.export_references(ref_dics, filename=reference_file)
    print("Time on importing references", time()-t0)

if os.path.exists(parameters_init_file):
    parameters0 = params.load_parameters(parameters_init_file)
else:
    t0 = time()
    parameters0 = params.params_init.copy()
    _, parameters0 = params.optimize_offset(ref_dics, parameters0, guess=True)
    params.export_parameters(parameters_init_file, parameters0)
    print("Time on importing parameters", time()-t0)

# FF parameters optimization
if os.path.exists(parameters_opt_file):
    parameters_opt = params.load_parameters(parameters_opt_file)
else:
    parameters_opt = parameters0.copy()

for i in range(1):
    for data in [
                 (['bond'], 250), 
                 (['angle'], 250),
                 (['bond', 'angle'], 250),
                 (['proper'], 150),
                 (['bond', 'angle', 'proper'], 150),
                 (['vdW'], 100),
                 (['proper', 'vdW'], 100),
                 (['charge'], 100),
                 (['vdW', 'charge'], 100),
                 (['proper', 'vdW', 'charge'], 150),
                 (['bond', 'angle', 'proper', 'vdW', 'charge'], 100),
                ]:
        (terms, steps) = data
        t0 = time()

        # Conduct actual optimization
        opt_dict = params.get_opt_dict(terms, 
                                       None,
                                       parameters_opt)

        print('Init values\n', opt_dict)
        x, fun, values, it = params.optimize_local(
                                         ref_dics,
                                         opt_dict,
                                         parameters_opt,
                                         obj = 'R2',
                                         steps = steps,
                                         )

        parameters_opt = params.set_sub_parameters(values, 
                                                   terms, 
                                                   parameters_opt)
        opt_dict = params.get_opt_dict(terms,
                                       None,
                                       parameters_opt)
        x, fun, values, it = params.optimize_global(
                                            ref_dics,
                                            opt_dict,
                                            parameters_opt,
                                            steps = steps,
                                            obj='R2',
                                            t0=0.15)

        parameters_opt = params.set_sub_parameters(values,
                                                   terms,
                                                   parameters_opt)
        t = (time()-t0)/60
        print('Opt_obj {:.2f} min '.format(t), terms, fun, it)
        print('Opt_solution\n', x)
        print('Opt_values\n', values)

        # Get the optimized energy offset
        _, parameters_opt = params.optimize_offset(ref_dics, parameters_opt)

    # Save the Final parameters
    params.export_parameters(parameters_opt_file, parameters_opt)
    
    # Results analysis
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    params.plot_ff_results(axes[0], parameters0, ref_dics, label='Init')
    params.plot_ff_results(axes[1], parameters_opt, ref_dics, label='Opt')
    plt.savefig('Results.png')
    print("\nexport results in Results.png")
    
    grid_size = (5, 2)
    fig = plt.figure(figsize=(10, 16))
    for i, term in enumerate(['bond', 'angle', 'proper', 'vdW', 'charge']):
        if term in ['charge', 'proper']:
            ax = plt.subplot2grid(grid_size, (i, 0), colspan=2, fig=fig)
            params.plot_ff_parameters(ax, parameters0, parameters_opt, term)
        else:
            ax1 = plt.subplot2grid(grid_size, (i, 0), fig=fig)
            ax2 = plt.subplot2grid(grid_size, (i, 1), fig=fig)
            params.plot_ff_parameters(ax1, parameters0, parameters_opt, term+'-1')
            params.plot_ff_parameters(ax2, parameters0, parameters_opt, term+'-2')
    plt.title(smiles[0])
    plt.savefig('parameters.png')
    print("export results in parameters.png\n")
