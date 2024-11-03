from pyocse.parameters import ForceFieldParameters
from time import time

params = ForceFieldParameters(
    smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],
    f_coef=1.0,
    s_coef=1.0,
    style = 'openff',
    ref_evaluator='mace',
    ncpu=1)

p0, _ = params.load_parameters("dataset/parameters.xml")
ref_dics = params.load_references("dataset/references.xml")[:20]

t0 = time()
_, params_opt = params.optimize_offset(ref_dics, p0)

for data in [
    (["bond", "angle", "proper"], 50),
    (["proper", "vdW", "charge"], 50),
    (["bond", "angle", "proper", "vdW", "charge"], 50),
]:
    (terms, steps) = data

    # Actual optimization
    opt_dict = params.get_opt_dict(terms, None, params_opt)
    x, fun, values, it = params.optimize_global(ref_dics,
                                                opt_dict,
                                                params_opt,
                                                obj="R2",
                                                t0=0.1,
                                                steps=25)

    params_opt = params.set_sub_parameters(values, terms, params_opt)

    opt_dict = params.get_opt_dict(terms, None, params_opt)

    x, fun, values, it = params.optimize_local(ref_dics,
                                               opt_dict,
                                               params_opt,
                                               obj="R2",
                                               steps=steps)

    params_opt = params.set_sub_parameters(values, terms, params_opt)
    _, params_opt = params.optimize_offset(ref_dics, params_opt)

t = (time() - t0) / 60
print(f"FF optimization {t:.2f} min ", fun)
errs = params.plot_ff_results("performance.png", ref_dics, [params_opt])
params.plot_ff_parameters("parameters.png", [params_opt])
params.export_parameters("parameters_opt.xml", params_opt, errs[0])

