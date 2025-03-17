# disable the lammps output
# /Users/qzhu8/miniconda3/envs/ost/lib/python3.9/site-packages/lammps/pylammps.py

from pyocse.parameters import ForceFieldParameters
from time import time
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os

class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout

with SuppressOutput():
    params = ForceFieldParameters(
        smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],
        f_coef=1.0,
        s_coef=1.0,
        style='openff',
        ref_evaluator='mace',
        ncpu=8,
    )

sys.stdout = sys.__stdout__  # Restore print output

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevents conflicts in parallel execution
    t0 = time()
    ref_dics = params.load_references("dataset/references.xml")
    print(f"loading references takes {(time() - t0) / 60:.2f} min ")

    os.makedirs("ASP", exist_ok=True)
    os.chdir("ASP")

    t0 = time()
    params_opt = params.optimize_offset(ref_dics)
    params.update_ff_parameters(params_opt)
    errs = params.plot_ff_results("performance_init.png", ref_dics, [params_opt])
    t = (time() - t0) / 60
    print(f"optimize and plot takes {t:.2f} min ")