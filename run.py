from pyxtal.db import database
from ost.parameters import ForceFieldParameters

# Load the data
db = database("../HT-OCSP/benchmarks/test.db")
xtal = db.get_pyxtal("ACSALA")
smiles = [mol.smile for mol in xtal.molecules]
assert smiles[0] is not None

# Initialize the param instance
params = ForceFieldParameters(smiles, f_coef=0.2) #; print(params)
params.export_parameters('parameters_0.xml')
#para = params.load_parameters('parameters.xml')#; print(para)

# Get the reference configurations
ase = xtal.to_ase(resort=False)
ff_dic = params.evaluate_ff_single(ase)
ref_dics = params.augment_ref_configurations(ase, steps=12, N_vibration=3)#0)
params.export_references(ref_dics, filename='reference.xml')
#ref_dics = params.load_references('reference.xml')

# FF parameters optimization
parameters0 = params.params_init.copy()
parameters_opt = None
values = None

for i in range(3):
    for terms in [['bond'],
                  #['angle'],
                  #['bond', 'angle'],
                  #['proper'],
                  #['bond', 'angle', 'proper'],
                  #['vdW'],
                  #['proper', 'vdW'],
                  #['charge'],
                  #['vdW', 'charge'],
                  #['proper', 'vdW', 'charge'],
                  #['proper', 'vdW', 'charge'],
                 ]:
        # Get the optimized energy offset
        offset_opt = params.optimize_offset(ref_dics)

        # Conduct actual optimization
        opt_dict = params.get_opt_dict(terms, parameters=parameters_opt)
        print('Initial values\n', opt_dict)
        x, fun, values = params.optimize(ref_dics, offset_opt, opt_dict, parameters_opt, steps=10, debug=True)
        parameters_opt = params.set_sub_parameters(values, terms, parameters_opt)
        print(terms, fun)
        print('Final solution\n', x)
        print('Final values\n', values)

# Save the Final parameters
params.export_parameters('parameters_opt.xml', parameters_opt)

# Results analysis
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
params.plot_ff_results(axes[0], parameters0, ref_dics, label='Init')
params.plot_ff_results(axes[1], parameters_opt, ref_dics, label='Opt')
plt.savefig('Results.png')
print("export results in Results.png\n")

#import numpy as np
#
#mace_eng, mace_force, mace_stress = [], [], []
#ff_eng1, ff_force1, ff_stress1 = [], [], []
#ff_eng2, ff_force2, ff_stress2 = [], [], []
#
#params.update_ff_parameters(parameters_opt)
#for ref_dic in ref_dics:
#    ff_dic = params.evaluate_ff_single(ref_dic['structure'], options=ref_dic['options'])
#    mace_eng.append(ref_dic['energy']/ref_dic['replicate'])
#    ff_eng1.append(ff_dic['energy']/ff_dic['replicate'] + offset_opt)
#    if ref_dic['options'][1]:
#        mace_force.extend(ref_dic['forces'].tolist())
#        ff_force1.extend(ff_dic['forces'].tolist())
#    if ref_dic['options'][2]:
#        mace_stress.extend(ref_dic['stress'].tolist())
#        ff_stress1.extend(ff_dic['stress'].tolist())
#
#params.update_ff_parameters(parameters0)
#for ref_dic in ref_dics:
#    ff_dic = params.evaluate_ff_single(ref_dic['structure'], options=ref_dic['options'])
#    ff_eng2.append(ff_dic['energy']/ff_dic['replicate'] + offset_opt)
#    if ref_dic['options'][1]:
#        ff_force2.extend(ff_dic['forces'].tolist())
#    if ref_dic['options'][2]:
#        ff_stress2.extend(ff_dic['stress'].tolist())
#
#mace_eng = np.array(mace_eng); ff_eng1 = np.array(ff_eng1); ff_eng2 = np.array(ff_eng2)
#mace_force = np.array(mace_force); ff_force1 = np.array(ff_force1); ff_force2 = np.array(ff_force2)
#mace_stress = np.array(mace_stress); ff_stress1 = np.array(ff_stress1); ff_stress2 = np.array(ff_stress2)
#
#fig, axs = plt.subplots(2, 3, figsize=(16, 8))
#axs[0, 0].scatter(mace_eng, ff_eng2, label='Energy (Init: {:.2f})'.format(np.sum((ff_eng2-mace_eng)**2)))
#axs[1, 0].scatter(mace_eng, ff_eng1, label='Energy (Opt: {:.2f})'.format(np.sum((ff_eng1-mace_eng)**2)))
#axs[0, 1].scatter(mace_force, ff_force2, label='Forces (Init: {:.2f})'.format(np.sum( (ff_force2-mace_force)**2 ))   )
#axs[1, 1].scatter(mace_force, ff_force1, label='Forces (Opt: {:.2f})'.format( np.sum((ff_force1-mace_force)**2)))
#axs[0, 2].scatter(mace_stress, ff_stress2, label='Stress (Init: {:.2f})'.format(np.sum((ff_stress2-mace_stress)**2)))
#axs[1, 2].scatter(mace_stress, ff_stress1, label='Stress (Opt: {:.2f})'.format(np.sum((ff_stress1-mace_stress)**2)))
#for ax in axs.flatten():
#    ax.set_xlabel('MACE')
#    ax.set_ylabel('FF')
#    ax.legend()

