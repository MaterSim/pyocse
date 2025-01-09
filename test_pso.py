from pyocse.pso import PSO
from pyocse.parameters import ForceFieldParametersBase
from time import time
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import set_start_method

# Set the start method to 'spawn'
set_start_method('spawn', force=True)

# Global shared arguments for all workers
def worker_init(shared_params, shared_para0, shared_terms, shared_ref_dics):
    global params, para0, terms, ref_dics
    params = shared_params
    para0 = shared_para0
    terms = shared_terms
    ref_dics = shared_ref_dics

def worker(para_values):
    #t0 = time()
    path = mp.current_process().name
    score = obj_function(para_values, params, para0, terms, ref_dics, path)
    #print(f"Worker {mp.current_process().pid} finished in {time()-t0:.2f} s")
    return score

def obj_function_par(para_values_list, params, para0, terms, ref_dics, ncpu):
    """
    Parallel evaluation of objective function for multiple sets of parameters.

    Args:
        para_values_list: List of 1D-arrays, each containing parameter values.
        params: Force field parameter instance (shared among all tasks).
        para0: Base parameters.
        terms: List of force field terms.
        ref_dics: Reference dataset dictionary.
        num_workers: Number of parallel processes.

    Returns:
        List of objective scores.
    """
    t0 = time()

    if ncpu == 1:
        scores = []
        for i, para_value in enumerate(para_values_list):
            score = obj_function(para_value, params, para0, terms, ref_dics, '.')
            scores.append(score)
        print(f"Time for serial computation: {time()-t0:.4f}")
        return scores

    # Initialize the pool once and reuse it for all iterations
    if not hasattr(obj_function_par, 'pool'):
    
        obj_function_par.pool = mp.Pool(
            processes=ncpu,
            initializer=worker_init,
            initargs=(params, para0, terms, ref_dics)
        )
        print(f"Pool initialized, {time()-t0}")

    results = obj_function_par.pool.map(worker, para_values_list)
    print(f"Time for parallel computation: {time()-t0:.4f}")

    return results

def obj_function(para_values, params, para0, terms, ref_dics, path, obj='R2'):
    """
    Objective function for PSOGAOptimizer.

    Args:
        para_values: 1D-Array of parameter values to evaluate.
        params: parameter instance
        para0: Array of all FF parameter as the base
        terms: list of FF terms to optimize
        ref_dics: dictionary of dataset

    Returns:
        Objective score.
    """

    # Split 1D array of para_values to a list grouped by each term
    #sub_values = []
    #count = 0
    #for term in terms:
    #    N = getattr(params, 'N_'+term)
    #    sub_values.append(para_values[count:count+N])
    #    count += N

    #print("debug subvals", sub_values[0][:5], para0[:5])
    #updated_params = params.set_sub_parameters(sub_values, terms, para0)
    updated_params = params.set_sub_parameters(para_values, terms, para0)

    # Update the parameters in the force field with the base parameter
    params.update_ff_parameters(updated_params)

    # Reset the LAMMPS input if necessary
    lmp_in = params.ff.get_lammps_in()

    # Calculate the objective (e.g., MSE)
    objective_score = params.get_objective(
        ref_dics=ref_dics,
        e_offset=para0[-1],
        lmp_in=lmp_in,
        obj=obj,
        path=path,
    )

    return objective_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncpu", type=int, default=5, 
                        help="Number of CPUs to use, default is 5.")
    parser.add_argument("--steps", type=int, default=30, 
                        help="Number of opt steps, default is 30.")
    parser.add_argument("--ref", default="dataset/references.xml", 
                        help="Reference dataset file, default is dataset/references.xml.")
    parser.add_argument("--params", default="dataset/parameters.xml", 
                        help="Initial parameters file, default is dataset/parameters.xml.")
    parser.add_argument("--style", default="openff", 
                        help="Force field style, default is openff.")
    parser.add_argument("--export", default="parameters_opt_pso.xml", 
                        help="Export optimized parameters, default is parameters_opt_pso.xml.")
    parser.add_argument("--dir", default="test", 
                        help="Output directory, default is test.")

    args = parser.parse_args()

    np.random.seed(1234)
    params = ForceFieldParametersBase(
        smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],
        f_coef=1, #0.1,
        s_coef=1, #0,
        e_coef=1,
        style=args.style,
        ncpu=1,
    )
    
    p0, errors = params.load_parameters(args.params)
    ref_dics = params.load_references(args.ref)[:1000]
    
    os.makedirs(args.dir, exist_ok=True)
    os.chdir(args.dir)
    
    t0 = time()
    e_offset, params_opt = params.optimize_offset(ref_dics, p0)
    params.update_ff_parameters(params_opt)
    errs = params.plot_ff_results("performance_init.png", ref_dics, [params_opt])
    print("MSE objective", params.get_objective(ref_dics, e_offset, obj="MSE"))
    print("R2 objective", params.get_objective(ref_dics, e_offset, obj="R2"))
    
    # Stepwise optimization loop
    terms = ["bond", "angle", "proper", "vdW"]
    
    sub_vals, sub_bounds, _ = params.get_sub_parameters(params_opt, terms)
    vals = np.concatenate(sub_vals)
    bounds = np.concatenate(sub_bounds)

    # PSO-GA optimization
    optimizer = PSO(
            obj_function=obj_function_par,
            obj_args=(params, params_opt, terms, ref_dics),
            bounds=bounds,
            seed=vals.reshape((1, len(vals))),
            num_particles=96, 
            dimensions=len(bounds),
            inertia=0.5,
            cognitive=0.2,
            social=0.8,
            mutation_rate=0.3,
            crossover_rate=0.5,
            max_iter=args.steps,
            ncpu=args.ncpu,
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
