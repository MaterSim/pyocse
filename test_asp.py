# disable the lammps output
# /Users/qzhu8/miniconda3/envs/ost/lib/python3.9/site-packages/lammps/pylammps.py

from pyocse.parameters import ForceFieldParameters
from time import time
import os

params = ForceFieldParameters(
    smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],
    f_coef=1.0,
    s_coef=1.0,
    style = 'openff',
    ref_evaluator='mace',
    ncpu=1)

p0, errors = params.load_parameters("dataset/parameters.xml")
ref_dics = params.load_references("dataset/references.xml")[:120]

os.makedirs("ASP", exist_ok=True)
os.chdir("ASP")

t0 = time()
e_offset, params_opt = params.optimize_offset(ref_dics, p0)
params.update_ff_parameters(params_opt)
print(params.get_objective(ref_dics, e_offset, obj="MSE"))
errs = params.plot_ff_results("performance_init.png", ref_dics, [params_opt])

# Stepwise optimization of each term
for data in [
    (["bond", "angle", "proper"], 50),
    (["proper", "vdW", "charge"], 50),
    (["bond", "angle", "proper", "vdW", "charge"], 50),
]:
    (terms, steps) = data
    opt_dict = params.get_opt_dict(terms, None, params_opt)
    x, fun, values, it = params.optimize_global(ref_dics,
                                                opt_dict,
                                                params_opt,
                                                obj="MSE",
                                                t0=0.1,
                                                steps=25)
    params_opt = params.set_sub_parameters(values, terms, params_opt)

    opt_dict = params.get_opt_dict(terms, None, params_opt)
    x, fun, values, it = params.optimize_local(ref_dics,
                                               opt_dict,
                                               params_opt,
                                               obj="MSE",
                                               steps=steps)
    params_opt = params.set_sub_parameters(values, terms, params_opt)
    _, params_opt = params.optimize_offset(ref_dics, params_opt)

    t = (time() - t0) / 60
    print(f"\nFF optimization {t:.2f} min ", data)

#print(params.get_objective(ref_dics, e_offset))
errs = params.plot_ff_results("performance_opt.png", ref_dics, [params_opt])
params.plot_ff_parameters("parameters.png", [params_opt])
params.export_parameters("parameters_opt.xml", params_opt, errs[0])
