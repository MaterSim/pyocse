from pyocse.pso import PSO
from pyocse.parameters import ForceFieldParametersBase, timeit
from time import time
import os
import numpy as np
from ase import units 

import numpy as np

def r2_score(y_true, y_pred):
    tss = np.sum((y_true - np.mean(y_true))**2)
    rss = np.sum((y_true - y_pred)**2)
    r2 = 1 - (rss / tss)
    return r2

def get_force_arr(input_file):
    # Initialize lists to store data
    forces = []

    # Read the forces_summary.txt file
    with open(input_file, "r") as file:
        for line in file:
            # Skip non-data lines (assuming format: id type fx fy fz)
            if line.strip() and not line.startswith("ITEM"):
                p = line.split()
                if len(p) == 5:  # Ensure the line contains data
                    forces.append(float(p[2]))
                    forces.append(float(p[3]))
                    forces.append(float(p[4]))
    forces = np.array(forces)
    return forces.flatten()

def obj_function_par(para_values, template, ref_data, e_offset, ncpu):
    print(len(para_values))
    scores = []
    if ncpu == 1:
        for i, para_value in enumerate(para_values):
            score = obj_function(para_value, template, ref_data, e_offset)
            print(i, score)
            scores.append(score)
    return scores

@timeit
def obj_function(para_value, template, ref_data, e_offset, path='lmp.in', obj='R2'):
    """
    Objective function for GAOptimizer.

    Args:
        para_value: 1D-Array of parameter values to evaluate.
        template: A dictionary to guide the order of coefficients
        ref_data: (eng, force, stress, numbers, mask_e, mask_f, mask_s)

    Returns:
        Objective score.
    """
    strs = f"variable index loop {len(ref_data[0])}\n"
    strs += """
label loop_start
clear 

units real
atom_style full

dimension 3
boundary p p p  # p p m for periodic and tilt box
atom_modify sort 0 0.0

bond_style hybrid harmonic
angle_style hybrid harmonic
dihedral_style hybrid charmm
special_bonds amber lj 0.0 0.0 0.5 coul 0.0 0.0 0.83333 angle yes dihedral no

neighbor 2.0 multi
neigh_modify every 2 delay 4 check yes

# Pair Style
pair_style lj/cut/coul/long 9.0 9.0
pair_modify mix arithmetic shift no tail yes

box tilt large

# Read unique Atoms and Box for the current structure
variable atomfile string structures/lmp_dat_${index}
read_data ${atomfile} #add append

mass 1  12.0107800 #U00C1
mass 2  12.0107800 #U00C2
mass 3  15.9994300 #U00O3
mass 4  15.9994300 #U00O4
mass 5  12.0107800 #U00C5
mass 6  12.0107800 #U00C6
mass 7  12.0107800 #U00C7
mass 8  12.0107800 #U00C8
mass 9  12.0107800 #U00C9
mass 10  12.0107800 #U00C10
mass 11  12.0107800 #U00C11
mass 12  15.9994300 #U00O12
mass 13  15.9994300 #U00O13
mass 14   1.0079470 #U00H14
mass 15   1.0079470 #U00H15
mass 16   1.0079470 #U00H16
mass 17   1.0079470 #U00H17
mass 18   1.0079470 #U00H18
mass 19   1.0079470 #U00H19
mass 20   1.0079470 #U00H20
mass 21   1.0079470 #U00H21
    """

    for key in template.keys():
        items = template[key]
        if key.startswith('bond_coeff'):
            [id1, id2] = items
            strs += f'{key} harmonic {para_value[id1]} {para_value[id2]}\n'
        elif key.startswith('angle_coeff'):
            [id1, id2] = items
            strs += f'{key} harmonic {para_value[id1]} {para_value[id2]}\n'
        elif key.startswith('dihedral_coeff'):
            [id1, arg0, arg1, arg2] = items
            strs += f'{key} charmm {para_value[id1]} {arg0} {arg1} {arg2}\n'
        elif key.startswith('pair_coeff'):
            [id1, id2] = items
            fact = 2**(5/6)
            strs += f'{key} {para_value[id1]} {para_value[id2]*fact}\n'
    
    strs += """
# dynamically specify the mesh grid based on the box
# Read box dimensions
variable xdim equal xhi-xlo
variable ydim equal yhi-ylo
variable zdim equal zhi-zlo
# Set minimum FFT grid size
variable xfft equal ceil(${xdim})
variable yfft equal ceil(${ydim})
variable zfft equal ceil(${zdim})
# Apply a minimum grid size of 32
if "${xfft} < 32" then "variable xfft equal 32"
if "${yfft} < 32" then "variable yfft equal 32"
if "${zfft} < 32" then "variable zfft equal 32"
kspace_style pppm 0.0005
kspace_modify gewald 0.29202898720871845 mesh ${xfft} ${yfft} ${zfft} order 6

# Thermodynamic Variables
variable pxx equal pxx
variable pyy equal pyy
variable pzz equal pzz
variable pyz equal pyz
variable pxz equal pxz
variable pxy equal pxy
variable energy equal etotal
# Perform single point energy calculation
#run 0

# Write energy to energy.txt 
# Write forces to forces.txt
# Write stress to stress.txt

compute myForces all property/atom fx fy fz
dump forcesDump all custom 1 forces.txt id type fx fy fz
dump_modify forcesDump append yes
run 0
undump forcesDump
print "${pxx} ${pyy} ${pzz} ${pyz} ${pxz} ${pxy}" append stress.txt
print "${energy}" append eng.txt

# Iterate to the next structure
next index
jump SELF loop_start
    """   
    e_coef = 1
    f_coef = 1
    s_coef = 1

    with open(path, 'w') as f:
        f.write(strs)

    # Run LAMMPS
    os.system("rm -f eng.txt stress.txt forces.txt")
    os.system("lmp -in lmp.in > lmp.out")

    # Read energy from LAMMPS output
    engs1 = np.loadtxt("eng.txt") * units.kcal/units.mol
    stress1 = -np.loadtxt("stress.txt").flatten() * 101325 * units.Pascal
    forces1 = get_force_arr("forces.txt").flatten() * units.kcal/units.mol

    # Reference data
    (engs, forces, stress, numbers, mask_e, mask_f, mask_s) = ref_data
    engs0 = engs / numbers
    engs0 = engs0[mask_e]
    forces0 = forces[mask_f]
    stress0 = stress[mask_s]

    # FF data
    engs1 /= numbers
    engs1 = engs1[mask_e] 
    engs1 += e_offset
    forces1 = forces1[mask_f]
    stress1 = stress1[mask_s]

    # Calculate objective score
    if obj == "MSE":
        score = e_coef * np.sum((engs1 - engs0) **2)
        score += f_coef * np.sum((forces1 - forces0) ** 2)
        score += s_coef * np.sum((stress1 - stress0) ** 2)
    elif obj == "R2":
        score = -r2_score(engs1, engs0)#; print('eng', engs1, engs0, r2_score(engs1, engs0))
        score -= r2_score(forces1, forces0)
        score -= r2_score(stress1, stress0)
    return score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncpu", type=int, default=1, 
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
    ref_dics = params.load_references(args.ref)
    
    os.makedirs(args.dir, exist_ok=True)
    os.chdir(args.dir)

    # Prepare lmp.dat at once
    params.write_lmp_dat_from_ref_dics(ref_dics)
    
    e_offset, params_opt = params.optimize_offset(ref_dics, p0)
    params.update_ff_parameters(params_opt)
    errs = params.plot_ff_results("performance_init.png", ref_dics, [params_opt])
    t0 = time()
    print("R2 objective", params.get_objective(ref_dics, e_offset, obj="R2"))
    os.system("mv lmp.in lmp0.in")
    t1 = time(); print("computation from params", t1-t0)

    # To be generated from params later
    template = {
        "bond_coeff 1": [0, 1],
        "bond_coeff 2": [2, 3],
        "bond_coeff 3": [4, 5],
        "bond_coeff 4": [6, 7],
        "bond_coeff 5": [8, 9],
        "bond_coeff 6": [10, 11],
        "bond_coeff 7": [12, 13],
        "bond_coeff 8": [14, 15],
        "bond_coeff 9": [16, 17],
        "bond_coeff 10": [18, 19],
        "angle_coeff 1": [20, 21],
        "angle_coeff 2": [22, 23],
        "angle_coeff 3": [24, 25],
        "angle_coeff 4": [26, 27],
        "angle_coeff 5": [28, 29],
        "angle_coeff 6": [30, 31],
        "dihedral_coeff 1": [32, 2, 180, 0.0], #U00(C1-C2-O4-C5)
        "dihedral_coeff 2": [33, 2, 180, 0.0], #U00(C2-O4-C5-C6,C2-O
        "dihedral_coeff 3": [34, 1, 0, 0.0], #U00(O3-C2-C1-H14,O3-
        "dihedral_coeff 4": [35, 2, 0, 0.0], #U00(O3-C2-C1-H14,O3-
        "dihedral_coeff 5": [36, 3, 180, 0.0], #U00(O3-C2-C1-H14,O3-
        "dihedral_coeff 6": [37, 3, 0, 0.0], #U00(O4-C2-C1-H14,O4-
        "dihedral_coeff 7": [38, 2, 180, 0.0], #U00(O4-C5-C6-C7,O4-C
        "dihedral_coeff 8": [39, 2, 180, 0.0], #U00(C5-C10-C11-O12,C
        "dihedral_coeff 9": [40, 2, 180, 0.0], #U00(C10-C11-O13-H21)
        "dihedral_coeff 10": [41, 2, 180, 0.0], #U00(O12-C11-O13-H21)
        "dihedral_coeff 11": [42, 1, 0, 0.0], #U00(O12-C11-O13-H21)
        "pair_coeff 1 1":   [44, 43],
        "pair_coeff 2 2":   [46, 45],
        "pair_coeff 3 3":   [48, 47],
        "pair_coeff 4 4":   [50, 49],
        "pair_coeff 5 5":   [52, 51],
        "pair_coeff 6 6":   [54, 53],
        "pair_coeff 7 7":   [56, 55],
        "pair_coeff 8 8":   [58, 57],
        "pair_coeff 9 9":   [60, 59],
        "pair_coeff 10 10": [62, 61],
        "pair_coeff 11 11": [64, 63],
        "pair_coeff 12 12": [66, 65],
        "pair_coeff 13 13": [68, 67],  
        "pair_coeff 14 14": [70, 69],
        "pair_coeff 15 15": [72, 71],
        "pair_coeff 16 16": [74, 73],
        "pair_coeff 17 17": [76, 75],
        "pair_coeff 18 18": [78, 77],
        "pair_coeff 19 19": [80, 79],
        "pair_coeff 20 20": [82, 81],
        "pair_coeff 21 21": [84, 83],
        }

    ref_data = params.get_reference_data_and_mask(ref_dics)

    # Stepwise optimization loop
    terms = ["bond", "angle", "proper", "vdW"]
    
    sub_vals, sub_bounds, _ = params.get_sub_parameters(params_opt, terms)
    vals = np.concatenate(sub_vals)
    bounds = np.concatenate(sub_bounds)

    # Test obj_function
    print(obj_function(vals, template, ref_data, e_offset, path='lmp.in', obj='R2'))
    #import sys; sys.exit()

    # PSO-GA optimization
    optimizer = PSO(
            obj_function=obj_function_par,
            obj_args=(template, ref_data, e_offset),
            bounds=bounds,
            seed=vals.reshape((1, len(vals))),
            num_particles=96, 
            dimensions=len(bounds),
            inertia=0.5,
            cognitive=0.2,
            social=0.8,
            mutation_rate=0.3,
            crossover_rate=0.5,
            max_iter=10, #args.steps,
            ncpu=args.ncpu,
    )
    
    #best_position, best_score = optimizer.optimize()
    
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
