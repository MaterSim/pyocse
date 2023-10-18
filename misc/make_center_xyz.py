import os, sys
from collections import deque
import numpy as np
from ase.io.lammpsrun import construct_cell, lammps_data_to_ase_atoms
from ase.atoms import Atoms
from ase.io import write
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from itertools import product
from math import gcd
import numpy as np
from scipy.signal import savgol_filter, find_peaks

def get_deformation_ids_from_ES(filename='output.dat', window=10, dE_tol=0.02):
    """
    Get the deformed structue ids from Energy-Strain curve
    """
    data = np.loadtxt(filename)[:, -1]
    data = savgol_filter(data, window, 2)
    peaks, _ = find_peaks(data)

    # Identify the downhill indices after each peak
    good_peaks = []
    for peak in peaks:
        for i in range(peak, len(data)-1):
            if data[i] < data[i+1]:
                break
        dE, dEb = data[peak] - data[i], data[peak] - data[0]
        if dE > dE_tol:
            good_peaks.append([peak, i, dE, dEb])
    return good_peaks

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
        if count % 100 == 0: print(count, atoms.cell.cellpar()[:4])#; import sys; sys.exit()
        images.append(atoms)
        count += 1
    return images

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


def find_best_hkl(input_vector, hkl_list=None, constraint=None, cutoff=4):
    """
    Find the nearest integer hkl for a given input_vector

    Args:
        - input vector: a real valued 3-array
        - hkl_list: list of possible integer hkls
        - constraint: hkl of perpendicular axis
    """
    if hkl_list is None:
        cutoff_range = range(-cutoff, cutoff+1)
        hkl_list = product(cutoff_range, repeat=3)

    best_integer_vector = None
    min_angle = float('inf')
    norm = np.linalg.norm(input_vector)

    # Enumerate all possible integer vectors within the cutoff range
    for i, j, k in hkl_list:
        integer_vector = np.array([i, j, k])
        if [i, j, k] != [0, 0, 0] and not has_reduction(integer_vector):
            # Only compute the vectors that satisfy the orthogonality constraint
            good = True
            if constraint is not None:
                if np.abs(np.dot(integer_vector, constraint)) > 1e-4:
                    good = False

            if good:
                # Calculate the angle between the input vector and the integer vector
                angle = np.arccos(np.dot(input_vector, integer_vector) / (norm * np.linalg.norm(integer_vector)))
                if angle > np.pi/2: 
                    angle = np.pi - angle 
                    integer_vector *= -1
                # Update the best integer vector if a smaller angle is found
                if angle < min_angle:
                    min_angle = angle
                    best_integer_vector = [i, j, k]
    #print("Closest Integer Vector:", best_integer_vector)
    return best_integer_vector


if __name__ == "__main__":

    from pkg_resources import resource_filename
    from pyxtal.db import database
    from pyxtal.plane import planes
    
    if len(sys.argv)<2:
        raise RuntimeError('Needs to provide path')
    
    path = sys.argv[1]
    os.chdir(path)
    cwd = os.getcwd().split('/')
    for p in cwd:
        if p.find('MT-') >= 0:
            code = p.split('-')[1]
            break
    
    db_path = resource_filename("pyxtal", "database/mech.db")
    db = database(db_path)
    p = planes()
    p.set_xtal(db.get_pyxtal(code))
    cp_planes = p.search_close_packing_planes()
    hkl_list = None #[p[0] for p in cp_planes if p[-1][0]>-0.2]

    #p.gather(cp_planes)

    images = get_structures()
    write('center.xyz', images=images, format='extxyz')
    print("Complete", path+'/center.xyz')
    
    # Set the reference and deformed structure 
    good_peaks = get_deformation_ids_from_ES()
    for good_peak in good_peaks:
        (ref_id, deform_id, dE, dEb) = good_peak
        s1 = images[ref_id]
        s2 = images[deform_id]
        pos1 = s1.get_scaled_positions()
        pos2 = s2.get_scaled_positions()
    
        # Compute the displacement vector 
        disps = pos2 - pos1
        disps -= np.round(disps)
        print("Summation: ", disps.sum(axis=0))
        #disps /= window
        #disps = np.dot(disps.)
    
        # Perform PCA on displacement vectors to find the slip direction
        pca = PCA(n_components=3) 
        pca.fit(disps)
        principal_components = pca.components_
        explained_variances = pca.explained_variance_ratio_
        #dominant_direction = np.mean(disps, axis=0)#pca.components_[0]
        dominant_direction = pca.components_[0]
        slip_direction = find_best_hkl(dominant_direction, hkl_list)
    
        print("\nVariance:       ", explained_variances)
        print("PCA  Direction:", dominant_direction)
        print("Slip Direction:", slip_direction)
    
        # Calculate dot products with the slip direction
        dot_products = np.abs(np.dot(disps, slip_direction))
    
        # Identify points belonging to the shifted slab.
        threshold = 0.8 * dot_products.max()
        shifted_slab_points = pos1[dot_products > threshold]
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(shifted_slab_points)
        unique_labels = np.unique(clustering.labels_)
        slip_planes = []
        for label in unique_labels:
            #print("Cluster id", label)
            if label == -1:
                continue  # Skip noise points
            cluster_points = shifted_slab_points[clustering.labels_ == label]
            pca = PCA(n_components=3) 
            pca.fit(cluster_points)
            plane = find_best_hkl(pca.components_[-1], hkl_list, constraint=slip_direction)
            #print("Shift plane:   ", plane, pca.explained_variance_ratio_[-1])
            if plane not in slip_planes:
                slip_planes.append(plane)

        print('\nFinal results', good_peak)
        print("Slip Direction:", slip_direction)
        for plane in slip_planes:
            print("Shift plane:   ", plane)

