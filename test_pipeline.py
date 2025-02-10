"""
Global Optimizer to get the training data
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
import argparse
import os
from time import time
import numpy as np

from pyxtal.optimize import WFS, DFS
from pyxtal.representation import representation
from pyocse.pso import PSO
from pyocse.parameters import ForceFieldParametersBase
from test_psopl import TEMPLATE, obj_function_par

from multiprocessing import set_start_method
set_start_method('spawn', force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gen", dest="gen", type=int, default=10,
                        help="Number of generation, optional")
    parser.add_argument("-p", "--pop", dest="pop", type=int, default=100,
                        help="Population size, optional")
    parser.add_argument("-n", "--ncpu", dest="ncpu", type=int, default=1,
                        help="cpu number, optional")
    parser.add_argument("--ffopt", action='store_true',
                        help="enable ff optimization")
    parser.add_argument("-a", "--algo", dest="algo", default='WFS',
                        help="algorithm, default: WFS")
    parser.add_argument("--dir", default="mtest",
                        help="Output directory, default is test.")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps for PSO, default is 10.")
    parser.add_argument("--pso", type=int, default=10,
                        help="Number of particles for PSO, default is 10.")

    options = parser.parse_args()

    smiles, sg, wdir = "CC(=O)OC1=CC=CC=C1C(=O)O", [14], "aspirin-simple"
    x = "81 11.38  6.48 11.24  96.9 1 0 0.23 0.43 0.03  -44.6   25.0   34.4  -76.6   -5.2  171.5 0"
    style = 'openff'

    # ff parameters
    np.random.seed(1234)
    params = ForceFieldParametersBase(
        smiles=[smiles],
        f_coef=1, #0.1,
        s_coef=1, #0,
        e_coef=1,
        style=style,
        ncpu=options.ncpu,
    )

    # Convert
    rep = representation.from_string(x, [smiles])
    xtal = rep.to_pyxtal()
    pmg = xtal.to_pymatgen()

    # Sampling to get references.xml
    if True:
        fun = globals().get(options.algo)
        go = fun(smiles,
                 wdir,
                 sg,
                 tag = 'aspirin',
                 N_gen = options.gen,
                 N_pop = options.pop,
                 N_cpu = options.ncpu,
                 ff_style = style,
                 ff_opt = True,
                )
        go.run()

    # PSO optimization
    p0, errors = params.load_parameters(f"{wdir}/parameters.xml") # from csp folder
    ref_dics = params.load_references(f"{wdir}/references.xml")   # from csp folder
    
    cwd = os.getcwd()
    os.makedirs(options.dir, exist_ok=True)
    os.chdir(options.dir)

    # Prepare lmp.dat at once
    params.write_lmp_dat_from_ref_dics(ref_dics)

    e_offset, params_opt = params.optimize_offset(ref_dics, p0)
    params.update_ff_parameters(params_opt)
    errs = params.plot_ff_results("performance_init.png", ref_dics, [params_opt])
    t0 = time()
    print("R2 objective", params.get_objective(ref_dics, e_offset, obj="R2"))
    os.system("mv lmp.in lmp0.in")
    t1 = time(); print("computation from params", t1-t0)

    ref_data = params.get_reference_data_and_mask(ref_dics)

    # Stepwise optimization loop
    terms = ["bond", "angle", "proper", "vdW"]

    sub_vals, sub_bounds, _ = params.get_sub_parameters(params_opt, terms)
    vals = np.concatenate(sub_vals)
    bounds = np.concatenate(sub_bounds)

    optimizer = PSO(
            obj_function = obj_function_par,
            obj_args = (TEMPLATE, ref_data, e_offset, options.ncpu),
            bounds = bounds,
            seed = vals.reshape((1, len(vals))),
            num_particles = options.pso,
            dimensions = len(bounds),
            inertia = 0.5,
            cognitive = 0.2,
            social = 0.8,
            max_iter = options.steps,
            ncpu = options.ncpu,
            xml_file = "pso.xml",
    )

    best_position, best_score = optimizer.optimize()
    params_opt = params.set_sub_parameters(best_position, terms, params_opt)
    e_offset, params_opt = params.optimize_offset(ref_dics, params_opt)
    params.update_ff_parameters(params_opt)
    print("e_offset", e_offset)

    t = (time() - t0) / 60
    print(f"\nStepwise optimization for terms {terms} completed in {t:.2f} minutes.")
    print(f"Best Score: {best_score:.4f}")

    # Final evaluation and saving results
    errs = params.plot_ff_results("performance_opt_pso.png", ref_dics, [params_opt])
    params.export_parameters("parameters_opt_pso.xml", params_opt, errs[0])
    print("Optimization completed successfully.")

    optimizer = PSO.load("pso.xml",
                         obj_function_par,
                         (TEMPLATE, ref_data, e_offset, options.ncpu))
    print(optimizer)
    print(optimizer.global_best_id, optimizer.global_best_score)
    os.chdir(cwd)
    os.system(f"cp {options.dir}/parameters_opt_pso.xml {wdir}/parameters.xml")

    if True:
        fun = globals().get(options.algo)
        go = fun(smiles,
                 wdir,
                 sg,
                 tag = 'aspirin',
                 N_gen = options.gen,
                 N_pop = options.pop,
                 N_cpu = options.ncpu,
                 ff_style = style,
                 ff_opt = True,
                 ff_parameters = "parameters.xml",
                )
        go.run()

    print("Done.")