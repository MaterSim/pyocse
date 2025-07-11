from pyxtal.db import database
from pyocse.parameters import ForceFieldParameters
import matplotlib.pyplot as plt

# Load the data
db = database("../HT-OCSP/benchmarks/test.db")
xtal = db.get_pyxtal("ACSALA")
smiles = [mol.smile for mol in xtal.molecules]
assert smiles[0] is not None

# Initialize the param instance
params = ForceFieldParameters(smiles, f_coef=0.2); print(params)
params.export_parameters('parameters_0.xml')
#para = params.load_parameters('parameters.xml')#; print(para)

# Get the reference configurations
ase = xtal.to_ase(resort=False)
ff_dic = params.evaluate_ff_single(ase)
ref_dics = params.augment_ref_configurations(ase, steps=12, N_vibration=5)#0)
params.export_references(ref_dics, filename='reference.xml')
#ref_dics = params.load_references('reference.xml')

# FF parameters optimization
parameters0 = params.params_init.copy()
parameters_opt = None
values = None

for i in range(3):
    for terms in [['bond'],
                  ['angle'],
                  #['bond', 'angle'],
                  ['proper'],
                  #['bond', 'angle', 'proper'],
                  ['vdW'],
                  #['proper', 'vdW'],
                  ['charge'],
                  #['vdW', 'charge'],
                  #['proper', 'vdW', 'charge'],
                  #['proper', 'vdW', 'charge'],
                 ]:
        # Get the optimized energy offset
        offset_opt = params.optimize_offset(ref_dics)

        # Conduct actual optimization
        opt_dict = params.get_opt_dict(terms, parameters=parameters_opt)
        print('Initial values\n', opt_dict)
        x, fun, values = params.optimize(ref_dics,
                                         offset_opt,
                                         opt_dict,
                                         parameters_opt,
                                         steps = 50,
                                         debug = True)
        parameters_opt = params.set_sub_parameters(values, terms, parameters_opt)
        print(terms, fun)
        print('Final solution\n', x)
        print('Final values\n', values)

# Save the Final parameters
params.export_parameters('parameters_opt.xml', parameters_opt)

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
