import os, sys
from collections import deque
import numpy as np
from ase.io.lammpsrun import construct_cell, lammps_data_to_ase_atoms
from ase.atoms import Atoms
from ase.io import write
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from itertools import product

def get_structures(center_file='center.dat', box_file='box_parameters.dat'):

    # Read center dat and box dat
    with open(center_file, 'r') as f: lines = deque(f.readlines())
    box = np.loadtxt(box_file)
    
    # Create structure list
    images = []
    count = 0
    line = lines.popleft()#; print(line)
    line = lines.popleft()#; print(line)
    line = lines.popleft()#; print(line)
    n_atoms = 0
    
    while len(lines) > n_atoms:
        line = lines.popleft(); #print(line)
        tmp = line.split(); #print(tmp)
        n_atoms, step = int(tmp[1]), int(tmp[0])
        datarow = ['6 '+ lines.popleft() for _ in range(n_atoms)]
        xyz = np.loadtxt(datarow, dtype=str)[1:]
        [xhi, xlo, yhi, ylo, zhi, zlo, xy, xz, yz] = box[count]
        xlo += min([0, xy, xz, xy+xz])
        xhi += max([0, xy, xz, xy+xz])
        ylo += min([0, yz])
        yhi += max([0, yz])
        diagdisp = [xlo, xhi, ylo, yhi, zlo, zhi]
        offdiag = [xy, xz, yz]
        cell, celldisp = construct_cell(diagdisp, offdiag)
        atoms = lammps_data_to_ase_atoms(
                data=xyz,
                colnames=['type', 'id', 'x', 'y', 'z'],
                cell=cell,
                celldisp=celldisp,
                atomsobj=Atoms,
                pbc=[1, 1, 1],
                )
        print(count, atoms.cell.cellpar())#; import sys; sys.exit()
        images.append(atoms)
        count += 1
    return images

def get_deformation_id_from_ES(filename=''):
    """
    Get the first deformed structue id from Energy-Strain curve
    """
    pass
    #return s

def has_reduction(hkl):
    h, k, l = hkl
    gcf = gcd(h, gcd(k, l))
    if gcf > 1:
        # like [2, 2, 0]
        return True
    elif hkl[np.nonzero(np.array(hkl))[0][0]] < 0:
        # like [-2, 0, 0]
        return True
    return False


def find_best_hkl(input_vector):
    cutoff_range = range(-5, 6)
    best_integer_vector = None
    min_angle = float('inf')
    norm = np.linalg.norm(input_vector)

    # Enumerate all possible integer vectors within the cutoff range
    for i, j, k in product(cutoff_range, repeat=3):
        integer_vector = np.array([i, j, k])
        if [i, j, k] != [0, 0, 0] and not has_reduction(integer_vector):
        # Calculate the angle between the input vector and the integer vector
        angle = np.arccos(np.dot(input_vector, integer_vector) / (norm * np.linalg.norm(integer_vector)))
        # Update the best integer vector if a smaller angle is found
        if angle < min_angle:
            min_angle = angle
            best_integer_vector = [i, j, k]
    print("Closest Integer Vector:", best_integer_vector)
    return best_integer_vector


if __name__ == "__main__":

    if len(sys.argv)<2:
        raise RuntimeError('Needs to provide path')
    
    path = sys.argv[1]
    os.chdir(path)
    
    images = get_structures()
    write('center.xyz', images=images, format='extxyz')
    print("Complete", path+'/center.xyz')
    
    # Set the reference and deformed structure 
    s1 = images[0]
    s2 = images[15] #slip_id]
    
    # Compute the displacement vector 
    pos1 = s1.get_scaled_positions()
    pos2 = s2.get_scaled_positions()
    disps = pos2 - pos1
    disps -= np.round(disps)
    
    # Perform PCA on displacement vectors to find the slip direction
    pca = PCA(n_components=3) 
    pca.fit(disps)
    principal_components = pca.components_
    explained_variances = pca.explained_variance_ratio_
    dominant_direction = pca.components_[0]
    slip_direction = find_best_hkl(dominant_direction)
    
    print("Variance:       ", explained_variances)
    print("PCA  Direction:", dominant_direction)
    print("Slip Direction:", slip_direction)
    
    # Calculate dot products with the slip direction
    dot_products = np.abs(np.dot(diff, slip_direction))
    
    # Identify points belonging to the shifted slab.
    threshold = 0.8 * dot_products.max()
    shifted_slab_points = pos1[dot_products > threshold]
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(shifted_slab_points)
    unique_labels = np.unique(clustering.labels_)
    slip_planes = []
    for label in unique_labels:
        print("Cluster id", label)
        if label == -1:
            continue  # Skip noise points
        cluster_points = shifted_slab_points[clustering.labels_ == label]
        pca = PCA(n_components=3) 
        pca.fit(cluster_points)
        print("Shift direction", pca.components_[0])  # slip direction
        print("Shift plane:   ", pca.components_[-1]) # least variance
        print("Variance", pca.explained_variance_ratio_)
        plane = find_best_hkl(pca.components_[-1])
        if plane not slip_planes:
            slip_planes.append(plane)


    print('\n Final results')
    print("Slip Direction:", slip_direction)
    for plane in slip_planes:
        print("Shift plane:   ", plane)

