"""
Pipeline to simultaneously optimize the force field parameters and the crystal structure
$ python -W "ignore" test_pipeline.py -p 5 -g 2 --pso 12 --steps 5
"""
import os
import argparse
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
    parser.add_argument("-a", "--algo", dest="algo", default='WFS',
                        help="algorithm, default: WFS")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps for PSO, default is 10.")
    parser.add_argument("--pso", type=int, default=10,
                        help="Number of particles for PSO, default is 10.")
    parser.add_argument("--dir", default="test",
                        help="Output directory, default is test.")

    options = parser.parse_args()

    # Enter the working directory
    os.makedirs(options.dir, exist_ok=True)
    os.chdir(options.dir)

    # Define the molecule and the space group
    np.random.seed(1234)
    smiles, sg, wdir = "CC(=O)OC1=CC=CC=C1C(=O)O", [14], f"Sampling-{options.algo}"
    x = "81 11.38  6.48 11.24  96.9 1 0 0.23 0.43 0.03  -44.6   25.0   34.4  -76.6   -5.2  171.5 0"
    style = 'openff'
    rep = representation.from_string(x, [smiles])
    xtal = rep.to_pyxtal()
    pmg = xtal.to_pymatgen()
    terms = ["bond", "angle", "proper", "vdW"]

    # FF parameters
    params = ForceFieldParametersBase(
        smiles=[smiles],
        style=style,
        ncpu=options.ncpu,
    )

    # Sampling to get the starting references.xml
    fun = globals().get(options.algo)
    go = fun(smiles, wdir, sg,
             tag = 'aspirin',
             N_gen = options.gen,
             N_pop = options.pop,
             N_cpu = options.ncpu,
             ff_style = style,
             ff_opt = True,
             check = False)
    go.run()
    go.save('sampling.xml')

    # Iterative optimization loop
    max_iter = 2
    for i in range(max_iter):
        t0 = time()

        # Prepare PSO inputs
        p0, _ = params.load_parameters(f"{wdir}/parameters.xml")
        ref_dics = params.load_references(f"{wdir}/references.xml")
        ref_data = params.get_reference_data_and_mask(ref_dics)
        params.write_lmp_dat_from_ref_dics(ref_dics)
        e_offset, params_opt = params.optimize_offset(ref_dics, p0)
        obj_args = (TEMPLATE, ref_data, e_offset, options.ncpu)

        # PSO optimization
        print("PSO optimization starts...")
        t0 = time()
        if os.path.exists("pso.xml"):
            optimizer = PSO.load("pso.xml", obj_function_par, obj_args)
        else:
            sub_vals, sub_bounds, _ = params.get_sub_parameters(params_opt, terms)
            vals = np.concatenate(sub_vals)
            bounds = np.concatenate(sub_bounds)
            optimizer = PSO(obj_function = obj_function_par,
                            obj_args = obj_args,
                            bounds = bounds,
                            seed = vals.reshape((1, len(vals))),
                            num_particles = options.pso,
                            dimensions = len(bounds),
                            max_iter = options.steps,
                            ncpu = options.ncpu,
                            xml_file = "pso.xml")
        best_position, best_score = optimizer.optimize()
        t = (time() - t0) / 60
        print(f"\nPSO is completed in {t:.2f} mins with Best Score: {best_score:.4f}")

        # Update the parameters and export the results
        params_opt = params.set_sub_parameters(best_position, terms, params_opt)
        e_offset, params_opt = params.optimize_offset(ref_dics, params_opt)
        params.update_ff_parameters(params_opt)
        errs = params.plot_ff_results(f"performance_pso_{i+1}.png", ref_dics, [params_opt])
        params.export_parameters(f"{wdir}/parameters.xml", params_opt, errs[0])

        # Sampling to get the starting references.xml
        if i < max_iter - 1:
            fun = globals().get(options.algo)
            go = fun.load('sampling.xml')
            go.run()

    print("Done.")
