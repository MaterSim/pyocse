import numpy as np
import os
from collections import deque
from ase.io.lammpsrun import get_max_index, construct_cell, lammps_data_to_ase_atoms
from ase import Atoms

def read_lammps_dump_text(filename, **kwargs):
    # Load all dumped timesteps into memory simultaneously
    fileobj = open(filename)
    lines = deque(fileobj.readlines())
    fileobj.close()
    index_end = get_max_index(-1)

    n_atoms = 0
    images = []

    # avoid references before assignment in case of incorrect file structure
    cell, celldisp, pbc = None, None, False

    while len(lines) > n_atoms:
        line = lines.popleft()

        if "ITEM: TIMESTEP" in line:
            n_atoms = 0
            line = lines.popleft()
            # !TODO: pyflakes complains about this line -> do something
            # ntimestep = int(line.split()[0])  # NOQA

        if "ITEM: NUMBER OF ATOMS" in line:
            line = lines.popleft()
            n_atoms = int(line.split()[0])

        if "ITEM: BOX BOUNDS" in line:
            # save labels behind "ITEM: BOX BOUNDS" in triclinic case
            # (>=lammps-7Jul09)
            tilt_items = line.split()[3:]
            celldatarows = [lines.popleft() for _ in range(3)]
            celldata = np.loadtxt(celldatarows)
            diagdisp = celldata[:, :2].reshape(6, 1).flatten()

            # determine cell tilt (triclinic case!)
            if len(celldata[0]) > 2:
                # for >=lammps-7Jul09 use labels behind "ITEM: BOX BOUNDS"
                # to assign tilt (vector) elements ...
                offdiag = celldata[:, 2]
                # ... otherwise assume default order in 3rd column
                # (if the latter was present)
                if len(tilt_items) >= 3:
                    sort_index = [tilt_items.index(i)
                                  for i in ["xy", "xz", "yz"]]
                    offdiag = offdiag[sort_index]
            else:
                offdiag = (0.0,) * 3

            cell, celldisp = construct_cell(diagdisp, offdiag)
            # Handle pbc conditions
            if len(tilt_items) == 3:
                pbc_items = tilt_items
            elif len(tilt_items) > 3:
                pbc_items = tilt_items[3:6]
            else:
                pbc_items = ["f", "f", "f"]
            pbc = ["p" in d.lower() for d in pbc_items]

        if "ITEM: ATOMS" in line:
            colnames = line.split()[2:]
            datarows = [lines.popleft() for _ in range(n_atoms)]
            data = np.loadtxt(datarows, dtype=str)
            out_atoms = lammps_data_to_ase_atoms(
                data=data,
                colnames=colnames,
                cell=cell,
                celldisp=celldisp,
                atomsobj=Atoms,
                pbc=pbc,
                **kwargs
            )
            images.append(out_atoms)

        if len(images) > index_end >= 0:
            break

    return images


images = read_lammps_dump_text('dump.lammpstrj')
with open('center.dat', 'r') as f:
    lines = f.readlines()

center_images = []
count = 0
for i, l in enumerate(lines[3:]):
    tmp = l.split()
    print(i, tmp)
    if len(tmp) == 2:
        print(l[:-1])
        step, N_mols = int(tmp[0]), int(tmp[1])
        #f.write('{:d}\n'.format(int(N_mols/2)))
        #f.write('{:d}\n'.format(step))

        pos = np.zeros([int(N_mols/2), 3])
        vel = np.zeros([int(N_mols/2), 3])
        for cell in range(int(N_mols/2)): 
            tmp0 = np.array([float(t) for t in lines[i+4+cell*2+0].split()[1:]])
            tmp1 = np.array([float(t) for t in lines[i+4+cell*2+1].split()[1:]])
            pos[cell] += (tmp0[:3]+tmp1[:3])/2
            vel[cell] += (tmp0[3:]+tmp1[3:])/2
            #f.write('C {:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(*pos[cell*2], *vel[cell*2]))
            #f.write('C {:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n'.format(*pos[cell*2+1], *vel[cell*2+1]))
        i += N_mols
        struc = Atoms([6]*len(pos), 
                      positions=pos, 
                      cell=images[count].cell.array, 
                      pbc=[1, 1, 1], 
                      velocities = vel)
        count += 1
        center_images.append(struc)

if os.path.exists('center.xyz'): os.system('rm center.xyz')
for struc in center_images:
    struc.write('center.xyz', format='extxyz', append=True)
            

