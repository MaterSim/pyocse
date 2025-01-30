"""
Global Optimizer to get the training data
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
import argparse
import os
from pyxtal.optimize import WFS, DFS, QRS
from pyxtal.representation import representation

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

    options = parser.parse_args()

    smiles, sg, wdir = "CC(=O)OC1=CC=CC=C1C(=O)O", [14], "aspirin-simple"
    x = "81 11.38  6.48 11.24  96.9 1 0 0.23 0.43 0.03  -44.6   25.0   34.4  -76.6   -5.2  171.5 0"

    # Convert
    rep = representation.from_string(x, [smiles])
    xtal = rep.to_pyxtal()
    pmg = xtal.to_pymatgen()

    # Sampling
    fun = globals().get(options.algo)
    go = fun(smiles,
             wdir,
             sg,
             tag = 'aspirin',
             N_gen = options.gen,
             N_pop = options.pop,
             N_cpu = options.ncpu,
             ff_style = 'gaff',
             #ff_parameters = "parameters_opt_pso_2.xml",
             ff_opt = True,
            )
    go.run() #ref_pmg=pmg)
    #go.print_matches(header='Ref_match')
    #go.plot_results()
