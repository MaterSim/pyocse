"""
Pipeline to simultaneously optimize the force field parameters and the crystal structure
$ python -W "ignore" test_pipeline.py --steps 5
"""
import os, argparse
from time import time
import numpy as np
from pyxtal.optimize import WFS, DFS
from pyocse.pso import PSO, obj_function_par
from pyocse.parameters import ForceFieldParametersBase
from multiprocessing import set_start_method
set_start_method('spawn', force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gen", dest="gen", type=int, default=2,
                        help="Number of generation, default is 2.")
    parser.add_argument("-p", "--pop", dest="pop", type=int, default=5,
                        help="Population size, dfault is 5.")
    parser.add_argument("-n", "--ncpu", dest="ncpu", type=int, default=1,
                        help="cpu number, default is 1.")
    parser.add_argument("-a", "--algo", dest="algo", default='WFS',
                        help="algorithm, default: WFS")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps for PSO, default is 10.")
    parser.add_argument("--pso", type=int, default=4,
                        help="Number of particles for PSO, default is 4.")
    parser.add_argument("--dir", default="test",
                        help="Output directory, default is test.")
    parser.add_argument("-i", "--iter", type=int, default=5,
                        help="Number of CSP/PSO iterations, default is 5.")

    options = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevents conflicts in parallel execution

    # Enter the working directory
    os.makedirs(options.dir, exist_ok=True)
    os.chdir(options.dir)

    # Define the molecule and the space group
    np.random.seed(1234)
    #smiles, sg, wdir = "CC(=O)OC1=CC=CC=C1C(=O)O", [14], f"Sampling-{options.algo}"
    tag, smiles, sg, wdir = 'coumarin', "C1=CC(=CC(=C1)O)O", [14], f"Sampling-{options.algo}"
    style = 'openff'
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
             tag = tag,
             N_gen = options.gen,
             N_pop = options.pop,
             N_cpu = options.ncpu,
             ff_style = style,
             ff_opt = True,
             check = False)
    go.run()
    go.save('sampling.xml')

    # Iterative optimization loop
    max_iter = options.iter
    for i in range(max_iter):
        t0 = time()

        # Prepare PSO inputs
        p0, _ = params.load_parameters(f"{wdir}/parameters.xml")
        ref_dics = params.load_references(f"{wdir}/references.xml")
        ref_dics = params.cut_references_by_error(ref_dics, p0, dE=2.5, FMSE=2.5)
        ref_data = params.get_reference_data_and_mask(ref_dics)
        params.write_lmp_dat_from_ref_dics(ref_dics)
        params_opt = params.optimize_offset(ref_dics, parameters0=p0)
        TEMPLATE = params.get_lmp_template()
        obj_args = (TEMPLATE, ref_data, params_opt[-1], options.ncpu)

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
        print(f"\nPSO {len(ref_dics)} time usage: {t:.2f} mins with Best Score: {best_score:.4f}")

        # Update the parameters and export the results
        params_opt = params.set_sub_parameters(best_position, terms, params_opt)
        params_opt = params.optimize_offset(ref_dics, parameters0=params_opt)
        params.update_ff_parameters(params_opt)
        errs = params.plot_ff_results(f"performance_pso_{i+1}.png", ref_dics, [params_opt])
        params.export_parameters(f"{wdir}/parameters.xml", params_opt, errs[0])

        # Sampling to get the starting references.xml
        if i < max_iter - 1:
            fun = globals().get(options.algo)
            go = fun.load('sampling.xml')
            go.run()
            params.remove_duplicate_structures(f"{wdir}/references.xml")
    print("Done.")
