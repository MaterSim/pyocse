#!/usr/bin/env python
import numpy as np
from pyocse.interfaces.parmed import ParmEdStructure, ommffs_to_paramedstruc
from pyocse.lmp import LAMMPSStructure
from pkg_resources import resource_filename
from monty.serialization import loadfn
import os, re

bonds = loadfn(resource_filename("pyxtal", "database/bonds.json"))

class Builder():
    """
    The parser is used to generate both structure and force field that
    are needed for atomistic simulations. Essentially, it supports the
    following functions:
        - 1. read the `toml` file for the gas molecule
        - 2. prepare the structure xyz and force fields for the given molecule
        - 3. generate the structures with multiple molecules and force fields
        - 4. combination of multiple parser, e.g., solvent calculation model
    """
    def __init__(self,
                 toml_files=None,
                 smiles=None,
                 style='gaff',
                 chargemethod = 'am1bcc',
                 ):
        """
        Args:
            toml_files (list): list of file paths
            smiles (list): molecular smiles
            molnames (list): short name to represent the molecule
            style (str): 'gaff' or 'openff'
            chargemethod (str): 'am1bcc' or 'gasteiger'
        """
        self.dics = []
        if toml_files is not None:
            smis = []
            import toml
            for i, toml_file in enumerate(toml_files):
                d = toml.load(toml_file)
                self.dics.append(d)
                smis.append(d["mol_smi"])
            self.smiles = ".".join(smis)
        else:
            from pyocse.convert import convert_gaff, convert_openff

            for i, smi in enumerate(smiles):
                smi = smiles[i]
                if style == 'gaff':
                    dic = convert_gaff(smiles=smi,
                                       molname=str(i),
                                       chargemethod=chargemethod)
                else:
                    dic = convert_openff(smiles=smi,
                                         molname=str(i),
                                         chargemethod=chargemethod)
                self.dics.append(dic)

            self.smiles = ".".join(smiles)

        self.molecules = self.molecules_from_dicts()
        self.tol_layer = 1.5 #

    def molecules_from_dicts(self, cls=ParmEdStructure):
        """
        Generate the list of molecular objects from ff dictionary
        """
        molecules = []
        for i, d in enumerate(self.dics):
            residuename='U{:02d}'.format(i)
            ffdic = d["data"].pop("omm_info")
            #print(ffdic["omm_forcefield"])
            #for a in ffdic["mol2"]: print(a)
            molecule = ommffs_to_paramedstruc(ffdic["omm_forcefield"],
                                              ffdic["mol2"],
                                              cls=cls)
            # To correct the charge for single atom case
            if len(molecule.atoms) == 1:
                molecule.atoms[0].charge = d['mol_charge']
            molecule.ffdic = ffdic
            molecule.change_residue_name(residuename)
            molecules.append(molecule)
        return molecules

    def set_xtal(self, xtal=None, cif=None, para_min=None, T=None, dimer=False):
        """
        Get the xtal models to self.structrure and self.numMols

        Args:
            xtal: pyxtal structure
            cif: cif file path
            para_min: whether or not create super cell
            T: temperature, if yes, will run npt equilibration
        """
        from pyxtal import pyxtal

        # Get the xtal object
        if xtal is None:
            if cif is None:
                msg = 'Must provide pyxtal structure or cif file'
                raise ValueError(msg)
            else:
                xtal = pyxtal(molecular=True)
                xtal.from_seed(cif, [x+'.smi' for x in self.smiles.split('.')])
        else:
            # QZ: check if the xtal is compatible in smiles string
            smiles = ''
            for i, mol in enumerate(xtal.molecules):
                smiles += mol.smile
                if i + 1 < len(xtal.molecules):
                    smiles += '.'
            if smiles != self.smiles:
                msg = "smiles strings are inconsistent\n"
                msg += 'from xtal: {:s}\n'.format(smiles)
                msg += 'from builder: {:s}'.format(self.smiles)
                raise ValueError(msg)

        # Convert to atoms class
        self.xtal = xtal.to_ase(resort=False)
        self.xtal_n_mols = xtal.numMols
        self.xtal_mol_list = []
        for i, n in enumerate(xtal.numMols):
            self.xtal_mol_list += [i]*n

        # Build a supercell
        if para_min is not None:
            [a, b, c] = self.xtal.get_cell_lengths_and_angles()[:3]
            replicate = [int(np.ceil(para_min/a)), int(np.ceil(para_min/b)), int(np.ceil(para_min/c))]
            self.xtal *= replicate
            self.xtal_mol_list *= np.product(replicate)
        #print(self.xtal.get_chemical_symbols())
        #self.xtal.write('super.xyz', format='extxyz')
        # Perform equilibriation
        self.T = T
        if T is not None:
            self.update_cell_parameters(T)

        if dimer:
            self.xtal = self.reset_positions_by_dimer(self.xtal, self.xtal_mol_list)

    def reset_positions_by_dimer(self, struc, mol_list, id=11, tol=3.4):
        """
        Reset atomic positions so that the paired molecules form the dimer
        E.g., 4 aspirin molecules in the unit cell will be rearranged so
        that both the first/second groups of 2 molecules forms the dimers.
        The PBC condition is also considered when arranging the cartesian
        coordinates.
        Note: this is only for aspirin so far

        Args:
            - struc: ase atoms
            - id: atom id to form the short contact
        """
        N_atoms = len(self.molecules[mol_list[0]].atoms)
        N_mols = int(len(struc)/N_atoms)
        inv_cell = np.linalg.inv(struc.cell.array)
        visited = []
        shifts = []
        for i in range(N_mols):
            if i not in visited:
                id1 = i*N_atoms+id
                pos1 = struc.get_positions()[id1]
                visited.append(i)
                shifts.append(np.zeros(3))
                for j in range(N_mols):
                    if i != j and j not in visited:
                        id2 = j*N_atoms+id
                        pos2 = struc.get_positions()[id2]
                        dist = np.dot(pos2 - pos1, inv_cell)
                        dist0 = np.dot(dist - np.round(dist), struc.cell.array)
                        if np.linalg.norm(dist0) < tol:
                            shift = np.round(dist)
                            #print(pos2, pos1, dist, shift)
                            pos2 -= shift
                            #print('dimer', i, j, dist0)
                            visited.append(j)
                            shifts.append(np.dot(shift, struc.cell.array))
                            break
        p0 = struc.get_positions()
        positions = np.zeros([len(struc), 3])
        for it, mid in enumerate(visited):
            positions[it*N_atoms:(it+1)*N_atoms] = p0[mid*N_atoms:(mid+1)*N_atoms]
            positions[it*N_atoms:(it+1)*N_atoms] -= shifts[it]
            #if it % 2 == 1:
            #    print(p0[mid*N_atoms+id])
            #    print(positions[it*N_atoms+id])
            #    print(positions[(it-1)*N_atoms+id])
            #    print(np.linalg.norm(positions[it*N_atoms+id]-positions[(it-1)*N_atoms+id]))
            #if it == 0:
            #    print(positions[:21])

        struc.set_positions(positions)
        return struc

    def update_cell_parameters(self, T=300, P=1.0, folder='tmp'):
        """
        Update cell parameters according to the given temperature
        """
        from lammps import lammps
        from ase.io import read

        cwd = os.getcwd()
        if not os.path.exists(folder): os.makedirs(folder)
        os.chdir(folder)

        xtal_with_lmp = self.get_ase_lammps(self.xtal)
        xtal_with_lmp.write_lammps() # lmp.dat, lmp.in
        with open('npt.in', 'w') as f:
            f.write('variable dt    equal 1# fs\n')
            f.write('variable tdamp equal 100*${dt}\n')
            f.write('variable pdamp equal 1000*${dt}\n')
            f.write('variable temperature equal {:12.3f}\n'.format(T))
            f.write('variable pressure equal {:12.3f}\n'.format(P))
            f.write('variable xlo equal xlo\n')
            f.write('variable xhi equal xhi\n')
            f.write('variable ylo equal ylo\n')
            f.write('variable yhi equal yhi\n')
            f.write('variable zlo equal zlo\n')
            f.write('variable zhi equal zhi\n')
            f.write('variable xy equal xy\n')
            f.write('variable yz equal yz\n')
            f.write('variable xz equal xz\n')

            f.write('\ninclude lmp.in\n')
            f.write('thermo 1000\n')
            f.write('velocity all create ${temperature} 1938072 dist gaussian #mom yes rot no\n')
            f.write('fix 1 all npt temp ${temperature} ${temperature} ${tdamp} iso ${pressure} ${pressure} ${pdamp} \n')
            f.write('fix 2 all ave/time 2 100 10000 v_xlo v_xhi v_ylo v_yhi v_zlo v_zhi v_xy v_xz v_yz\n')
            f.write('run 10000\n')
            f.write('variable xloo equal f_2[1]\n')
            f.write('variable xhii equal f_2[2]\n')
            f.write('variable yloo equal f_2[3]\n')
            f.write('variable yhii equal f_2[4]\n')
            f.write('variable zloo equal f_2[5]\n')
            f.write('variable zhii equal f_2[6]\n')
            f.write('variable vxy equal f_2[7]\n')
            f.write('variable vxz equal f_2[8]\n')
            f.write('variable vyz equal f_2[9]\n')
            f.write('change_box all x final ${xloo} ${xhii} y final ${yloo} ${yhii} z final ${zloo} ${zhii} xy final ${vxy} xz final ${vxz} yz final ${vyz}\n')
            f.write('run 0\n')
            f.write('# ------------- Energy minimization\n')
            f.write('min_style cg\n')
            f.write('minimize 1.0e-25 1.0e-25 100000 1000000\n')
            f.write('write_dump all custom dump.min id mol type x y z\n')

        lmp = lammps()
        lmp.file('npt.in')
        lmp.close()
        at = read('dump.min', format='lammps-dump-text')
        self.xtal.set_cell(at.cell.array)
        self.xtal.set_positions(at.get_positions())
        os.chdir(cwd)
        #from ase.io.lammpsrun import construct_cell
        #(los, his, xy, xz, yz, pbc, flag) = lmp.extract_box()
        #x_lo, y_lo, z_lo = los[0], los[1], los[2]
        #x_hi, y_hi, z_hi = his[0], his[1], his[2]
        ##conver to ase_matrix
        #diagdisp = np.array([[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]]).reshape(6, 1).flatten()
        #offdiag = [xy, yz, xz]
        #print(diagdisp, offdiag)
        #cell, cell_disp = construct_cell(diagdisp, offdiag)
        #pos = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
        #print("CELL after conversion")
        #print(cell)
        # Set the optimized cell parameters and atomic position

    def matrix_from_hkl(self, hkl, cell, orthogonality=False, tol=1e-10):
        """
        derive the rotation matrix from hkl indices
        based on https://wiki.fysik.dtu.dk/ase/_modules/ase/build/general_surface.html#surface
        Args:
            hkl: sequence of three int
            matrix: Atoms.cell
            orthogonality: whether or not impose orthogonality
        """
        from ase.build.general_surface import ext_gcd
        from math import gcd

        # the current basis vectors
        h, k, l = hkl
        a1, a2, a3 = cell

        # special cases like (0,0,1) etc
        if h==0 and k==0: #(0,0,1)
            matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif h==0 and l==0: #(0,1,0)
            matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        elif k==0 and l==0:
            matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        else:
            # general cases like (1,1,1)
            # find the greatest common divisor of the two integers
            p, q = ext_gcd(k, l)
            # constants describing the dot product of basis c1 and c2:
            # dot(c1,c2) = k1+i*k2, i in Z
            k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3),
                        l * a2 - k * a3)
            k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3),
                        l * a2 - k * a3)

            if abs(k2) > tol:
                i = -int(round(k1 / k2))  # i corresponding to the optimal basis
                p, q = p + i * l, q - i * k

            a, b = ext_gcd(p * k + q * l, h)

            c1 = (p * k + q * l, -p * h, -q * h)
            c2 = np.array((0, l, -k)) // abs(gcd(l, k))
            c3 = (h, k, l)

            matrix = np.array([c1, c2, c3])

        if orthogonality:
            #print('optimize the a, b vectors')
            a, b = self.optimize_lattice(cell, matrix)
            matrix[0] = self.find_simple_hkl(a)
            matrix[1] = self.find_simple_hkl(b)

        if np.linalg.det(matrix) < 0:
            matrix[0] *= -1
            #print("positive det", matrix)

        print('vector a: ', matrix[0])
        print('vector b: ', matrix[1])
        print('vector c: ', matrix[2])
        return matrix

    def find_simple_hkl(self, hkl):
        """
        reduce the hkl like (2, 2, 0 ) to (1, 1, 0)
        """
        factor = np.gcd.reduce(hkl)
        return hkl/factor

    def optimize_lattice(self, cell, matrix, tol=0.05):
        """
        Optimize the a, b basis vectors to reach a nearly orthogonal slab
        e.g, in the [a, b, c],
        Args:
            cell: basis vectors
            matrix: supercell matrix
        """
        #get a, b vectors
        matrix0 = matrix[:2, :]
        mat = np.dot(matrix0, cell) # 2*3 matrix
        mults = []
        for i in range(-2, 2):
            for j in range(-2, 2):
                if i!=0 or j!=0:
                    mults.append(np.array([i, j]))
        cosines = []
        a_mults = []
        b_mults = []
        for a_mult in mults:
            a = a_mult @ mat
            norm_a = np.linalg.norm(a)
            for b_mult in mults:
                b = b_mult @ mat
                norm_b = np.linalg.norm(b)
                cos = abs(np.dot(a, b)/norm_a/norm_b)
                #print(a_mult, b_mult, cos)
                if cos < tol:
                    return np.dot(a_mult, matrix0), np.dot(b_mult, matrix0)
                cosines.append(cos)
                a_mults.append(a_mult)
                b_mults.append(b_mult)

        cosines = np.array(cosines)
        #print(cosines)
        id = np.argmin(cosines)
        a, b = np.dot(a_mults[id], matrix0), np.dot(b_mults[id], matrix0)
        print('Minimum angle: ', np.degrees(np.arccos(cosines[id])))

        return a, b

    def check_matrix(self):
        pass

    def set_slab(self, atoms0, mol_list0, matrix=None, hkl=None, replicate=1, dim=None,
                layers=None, vacuum=None, separation=10.0,
                orthogonality=False, reset=True, dimer=False):
        """
        Create the slab structure from crystal with three steps:
            - 1. rotate the crystal (either matrix or indices) to make a new unitcell
            - 2, replicate the structure
            - 2. extract the desired number of layers
            - 3. add vacuum along z axis
        then prepare the ase.atoms for lammps calculation

        Args:
            atoms0: ase.atoms object
            mol_list0: list of molecular ids
            matrix: a matrix representing structure rotation
            hkl: sequence of three int for Miller indices (h, k, l)
            dim: e.g. [100, 20, 20], length in xyz
            replicate:
            layers: int, number of equivalent layers of the slab
            vacuum: float: amount of vacuum added on both sides of the slab
            separation: float: amount of separation to the top of the slab
            orthogonality: whether or not impose orthogonality
            reset: reset atoms position based on PBC and orientation
            dimer: whether or not reset mol center
        """
        from ase.build.supercells import make_supercell

        # derive the matrix from hkl indices
        if matrix is None:
            matrix = self.matrix_from_hkl(hkl, atoms0.cell, orthogonality)

        if matrix.size != 9:
            raise ValueError("Cannot make 3*3 matrix from the input", matrix)

        if reset:
            atoms = make_supercell(atoms0, matrix)
            atoms = self.reset_cell_vectors(atoms)
            atoms = self.reset_positions(atoms, mol_list0*int(round(np.linalg.det(matrix)))) #can be expensive! QZ
            atoms = self.reset_molecular_centers(atoms, mol_list0)

        if dimer:
            atoms = self.reset_positions_by_dimer(atoms, mol_list0)
            #atoms.write('reset.xyz', format='extxyz')

        if dim is not None:
            [a, b, c] = atoms.get_cell_lengths_and_angles()[:3]
            replicate = [int(np.ceil(dim[0]/a)), int(np.ceil(dim[1]/b)), int(np.ceil(dim[2]/c))]
            print("Replicate: ", replicate, len(atoms))

        atoms *= replicate
        mol_list = mol_list0 * np.product(replicate)

        # extract number of layers
        if layers is not None:
            atoms, mol_list = self.cut_layers(atoms, mol_list, layers)

        # add vacuum
        if vacuum is not None:
            vacuum = (vacuum + separation)/2
            atoms.set_pbc([1, 1, 0])
            atoms.center(vacuum, axis=2)
            # move atoms upward
            #atoms.translate([0, 0, vacuum-separation])

        self.ase_slab = atoms
        self.ase_slab_mol_list = mol_list
        self.lammps_slab = self.get_ase_lammps(atoms)

    def rotate_molecules(self, ase_struc, mol_list, vertices, rotation, ids):
        """
        selectively rotate some molecules based on boundary conditions.

        Args:
            ase_struc: ase atoms
            mol_list: list of molecular ids
            vertices (tuple): vertices in xy plane to define the twinning region
            rotation (tuple): (rotation axis, angle)
        """
        from scipy.spatial import ConvexHull
        from shapely.geometry import Point, Polygon
        from scipy.spatial.transform import Rotation

        N_per_mols = [len(self.molecules[i].atoms) for i in mol_list]
        centers = np.zeros([len(N_per_mols), 3])
        pos = ase_struc.get_positions()

        #print(rotation)
        (axis, angle) = rotation
        count = 0

        # Compute the convex hull of the vertices
        convex_hull = ConvexHull(vertices)
        convex_hull_vertices = [vertices[i] for i in convex_hull.vertices]
        convex_hull_polygon = Polygon(convex_hull_vertices)

        for i, N_per_mol in enumerate(N_per_mols):
            start = sum(N_per_mols[:i])
            end = start + N_per_mols[i]
            tmp = pos[start:end, :]
            center = tmp.mean(axis=0)
            center_frac = ase_struc.cell.scaled_positions(center)
            # Check if (x, y) is inside the parallelogram defined by its vertices.
            if len(tmp)>1 and convex_hull_polygon.contains(Point(center_frac[ids[0]], center_frac[ids[1]])):
                count += 1
                print('Twinning', i, count, len(mol_list), center_frac)
                r2 = Rotation.from_rotvec(axis*angle)
                pos[start:end, :] = np.dot(tmp-center, r2.as_matrix().T) + center
        ase_struc.set_positions(pos)
        return ase_struc


    def cut_layers(self, atoms, mol_list, layers, tol=None):
        """
        cut the supercell structure by number of molecular layers
        here we ignore the first layer to avoid partial termination

        Args:
            atoms: ase atoms sorted by molecules
            layers: int, number of layers
            tol: tolerance in angstrom
        """
        if tol is None: tol = self.tol_layer
        N_per_mols = [len(self.molecules[i].atoms) for i in mol_list]

        # sort molecular ids by z axis
        centers = self.get_molecular_centers(atoms, mol_list)
        ids_sorted = np.argsort(centers[:, 2])
        ids_to_count = []
        N_layer = 0
        current_z = centers[ids_sorted[0], 2]

        for id in ids_sorted:
            z = centers[id, 2]
            if z - current_z < tol: #same layer
                if N_layer > 0:
                    ids_to_count.append(id)
            else: # new layer
                N_layer += 1
                if N_layer > layers:
                    break
                else:
                    ids_to_count.append(id)
                    current_z = z
                    #print('Layer', N_layer, current_z)

        if N_layer < layers:
            msg = 'Can cut only {:d}/{:d} layers'.format(N_layer, layers)
            msg += '\n Please Increase the replicate dimension in z'
            raise ValueError(msg)
        else:
            print("Total Number of Molecules from cutting:", len(ids_to_count))
            print("Total Number of Layers:", N_layer-1)
            # Delete atoms
            ids_to_delete = []
            mids_to_delete = []

            for i in range(len(centers)):
                if i not in ids_to_count:
                    start = sum(N_per_mols[:i])
                    end = start + N_per_mols[i]
                    block = list(range(start, end))
                    mids_to_delete.append(i)
                    ids_to_delete.extend(block)
            del atoms[ids_to_delete]
            mol_list = [i for j, i in enumerate(mol_list) if j not in mids_to_delete]

            return atoms, mol_list

    def dump_slab_centers(self, atoms, mol_list, border_ids=None, fix_ids=None, filename='centers.dump'):
        """
        Quickly dump the molecular centers in lammps format
        """
        if border_ids is not None:
            f_border_ids = [j for sub in border_ids for j in sub]
        else:
            f_border_ids = []
        centers = self.get_molecular_centers(atoms, mol_list)
        centers = np.dot(centers, atoms.cell.array)
        with open(filename, 'w') as of:
            of.write("ITEM: TIMESTEP\n")
            of.write("0\n")
            of.write("ITEM: NUMBER OF ATOMS\n")
            of.write("{:d}\n".format(len(centers)))
            bounds = ""
            for p in atoms.pbc:
                if p:
                    bounds += 'pp '
                else:
                    bounds += 'ff '
            of.write("ITEM: BOX BOUNDS {:s}\n".format(bounds))
            cell_par = atoms.get_cell_lengths_and_angles()
            if np.abs(cell_par[3:]-90).max() > 5.0:
                print(cell_par)
                raise ValueError('This is not an orthogonal box, needs to adjust')
            else:
                xlo = ylo = zlo = 0.0
                xhi, yhi, zhi = cell_par[:3]
                of.write("{:9.4f} {:9.4f}\n".format(xlo, xhi))
                of.write("{:9.4f} {:9.4f}\n".format(ylo, yhi))
                of.write("{:9.4f} {:9.4f}\n".format(zlo, zhi))
                of.write("ITEM: ATOMS id type x y z \n")
                for i, center in enumerate(centers):
                    id = i+1
                    if id in fix_ids:
                        mtype = 2
                    elif id in f_border_ids:
                        mtype = 1
                    else:
                        mtype = 0
                    of.write("{:6d} {:2d} {:9.3f} {:9.3f} {:9.3f}\n".format(id, mtype, *center))

    def get_molecular_centers(self, atoms, mol_list, absolute=False):
        """
        Quickly collect the molecular center data for a given atoms

        Args:
            atoms:
            mol_list
        """
        N_per_mols = [len(self.molecules[i].atoms) for i in mol_list]
        centers = np.zeros([len(N_per_mols), 3])
        pos = atoms.get_positions()

        for i, N_per_mol in enumerate(N_per_mols):
            start = sum(N_per_mols[:i])
            end = start + N_per_mols[i]
            tmp = pos[start:end].mean(axis=0)
            if not absolute:
                centers[i] = atoms.cell.scaled_positions(tmp)
            else:
                centers[i] = tmp
        return centers

    def get_ase_lammps(self, atoms):
        """
        Add the lammps ff information into the give atoms object

        Args:
            Atoms: the ase atoms following the atomic order in self.molecules

        Return:
            Atoms with lammps ff information
        """
        # first adjust the cell into lammps format
        pbc = atoms.pbc
        atoms = self.reset_lammps_cell(atoms)
        if len(self.molecules) == 1:
            pd_struc = self.molecules[0].copy(cls=ParmEdStructure)
            pd_struc.update(atoms)
            atoms = LAMMPSStructure.from_structure(pd_struc)
        else:
            from functools import reduce
            from operator import add
            mols = []
            for i, m in enumerate(self.xtal_n_mols):
                mols += [self.molecules[i]*m]
            pd_struc = reduce(add, mols) #self.molecules)
            pd_struc.update(atoms)
            atoms = LAMMPSStructure.from_structure(pd_struc)
            #struc.restore_ffdic()
        atoms.set_pbc(pbc)
        atoms.title = self.smiles
        return atoms

    def reset_lammps_cell(self, atoms0):
        """
        set the cell into lammps format
        """
        from ase.calculators.lammpslib import convert_cell

        atoms = atoms0.copy()
        mat, coord_transform = convert_cell(atoms0.cell)
        if coord_transform is not None:
            pos = np.dot(atoms0.positions, coord_transform.T)
            atoms.set_cell(mat.T)
            atoms.set_positions(pos)
        return atoms

    def reset_cell_vectors(self, atoms):
        """
        readjust the cell matrix into good shape
        """
        from ase import geometry

        cell_par = atoms.get_cell_lengths_and_angles()
        pos1 = atoms.get_scaled_positions(wrap=False)
        cell1 = geometry.cell.cellpar_to_cell(cell_par)
        atoms.set_cell(cell1)
        atoms.set_scaled_positions(pos1)
        return atoms

    def reset_positions(self, atoms, mol_list):
        """
        After supercell conversion, the atoms in each molecule are no longer connected
        Needs to reset it

        Args:
            atoms: ase atoms
            mol_list: the corresponding sequence of molecules
        """
        def check_one_layer(atoms, pos, ids, visited, lists):
            new_members = []
            for id in ids:
                ids_add, visited, pos = check_one_site(atoms, pos, id, visited, lists)
                new_members.extend(ids_add)
            return new_members, visited, pos

        def check_one_site(atoms, pos, id0, visited, lists, rmax=2.5):
            """
            find the short distances from the given site
            """
            ids_add = []
            dists = atoms.get_distances(id0, lists, True)
            for i, id in enumerate(lists):
                if id not in visited and dists[i] < rmax:
                    # Check if it is a bond:
                    key = "{:s}-{:s}".format(atoms.symbols[id0], atoms.symbols[id])
                    if dists[i] < bonds[key]:
                        ids_add.append(id)
                        visited.append(id)
                        #update position
                        diff = pos[id] - pos[id0]
                        shift = np.round(diff)
                        pos[id] -= shift
            #print(id0, visited); import sys; sys.exit()
            return ids_add, visited, pos

        # QZ: Rewrite for multicomponent systems
        N_per_mols = [len(self.molecules[i].atoms) for i in mol_list]
        pos = atoms.get_scaled_positions(wrap=False)

        #print('Debug', N_per_mols)

        for i, N_per_mol in enumerate(N_per_mols):
            if N_per_mol > 1:
                visited_ids = []
                start = sum(N_per_mols[:i])
                end = start + N_per_mols[i]

                lists = list(range(start, end))
                #print("Finding molecules", lists)
                for id in lists:
                    if id not in visited_ids:
                        id0 = id
                        visited = [id0]
                        n_iter, max_iter = 0, N_per_mol
                        while n_iter < max_iter:
                            if n_iter == 0:
                                new_ids, visited, pos = check_one_site(atoms, pos, id0, visited, lists)
                            else:
                                #print('one layer')
                                new_ids, visited, pos = check_one_layer(atoms, pos, new_ids, visited, lists)
                                #import sys; sys.exit()
                            n_iter += 1
                            if len(new_ids) == 0:
                                break
        # check molecular_centers?
        atoms.set_scaled_positions(pos)
        return atoms


    def reset_molecular_centers(self, atoms, mol_list):
        """
        Reset the molecular centers to the unit cell
        """
        N_per_mols = [len(self.molecules[i].atoms) for i in mol_list]
        centers = np.zeros([len(N_per_mols), 3])
        scaled_pos = atoms.get_scaled_positions(wrap=False)

        # make sure the molecular centers are within [0, 1]
        # loop over each center
        for i, N_per_mol in enumerate(N_per_mols):
            start = sum(N_per_mols[:i])
            end = start + N_per_mols[i]
            center = scaled_pos[start:end].mean(axis=0)
            shift = np.floor(center)
            scaled_pos[start:end] -= shift
            centers[i] = np.dot(center-shift, atoms.cell)

        atoms.set_scaled_positions(scaled_pos)
        #print(atoms.get_scaled_positions()[:,2].max())
        #print(atoms.get_scaled_positions()[:,2].min())
        return atoms

    def get_molecular_bord_ids(self, atoms, mol_list, width=0.01, axis=0):
        """
        find the groups for border and fixed molecules:

        Args:
            width:
            axis: 0 or 1
        """

        border_ids = []

        centers = self.get_molecular_centers(atoms, mol_list)
        # select molecules by x-axis
        ref = centers[:, axis]
        x_hi, x_lo = ref.max(), ref.min()
        #print(x_lo-width, x_lo+width, x_hi-width, x_hi+width)
        border_ids.append(np.where( (ref > (x_lo - width)) & (ref <(x_lo + width)) )[0] + 1)
        border_ids.append(np.where( (ref > (x_hi - width)) & (ref <(x_hi + width)) )[0] + 1)

        return border_ids


    def get_molecular_fix_ids(self, atoms, mol_list, width=0.01, shift=0.2, axis=0):
        """
        find the groups for border and fixed molecules:

        Args:
            width:
            axis: 0 or 1
        """

        border_ids = []
        fix_ids = []

        centers = self.get_molecular_centers(atoms, mol_list)
        # select molecules by x-axis
        ref = centers[:, axis]
        x_hi, x_lo = ref.max() - shift, ref.min() + shift
        # make sure the x_hi/x_lo can be found
        x_hi = ref[np.abs(ref - x_hi).argsort()[0]]
        x_lo = ref[np.abs(ref - x_lo).argsort()[0]]
        border_ids.append(np.where( (ref > (x_lo - width)) & (ref <(x_lo + width)) )[0] + 1)
        border_ids.append(np.where( (ref > (x_hi - width)) & (ref <(x_hi + width)) )[0] + 1)
        #print(border_ids); import sys; sys.exit()

        # select molecules for the lowest z
        for border_id in border_ids:
            zs = centers[border_id-1][:, 2]
            zs -= zs.min()
            for i in range(len(zs)):
                if zs[i] < self.tol_layer/atoms.cell.array[2, 2]:
                    fix_ids.append(border_id[i]) # + 1)

        return fix_ids

    def get_molecular_ids(self, atoms, mol_list, width=5.0, axis=0):
        """
        find the groups for border and fixed molecules:

        Args:
            width:
            axis: 0 or 1
        """

        border_ids = []
        fix_ids = []
        centers = self.get_molecular_centers(atoms, mol_list)
        x_hi, x_lo = centers[:, axis].max(), centers[:, axis].min()
        border_ids.append(np.where(centers[:, axis]<(x_lo + width))[0] + 1)
        border_ids.append(np.where(centers[:, axis]>(x_hi - width))[0] + 1)

        for border_id in border_ids:
            zs = centers[border_id-1][:, 2]
            zs -= zs.min()
            for i in range(len(zs)):
                if zs[i] < self.tol_layer:
                    fix_ids.append(border_id[i]) # + 1)
        return border_ids, fix_ids

    def set_task(self, task, filename=None):
        """
        set the master lammps input file from the give task

        Args:
            task: a dictionary specifying lammps parameters
        """
        from pkg_resources import resource_filename as rf
        # possible modes:
        # - bulk (single/cycle)
        # - slab (single/cycle/3pf bending)

        _task = {'type': 'single',          # single/cycle/3pf
                 'mode': 'uniaxial',
                 'pbc': 'bulk',
                 'temperature': 300.0,      # K
                 'pressure': 1.0,           # atmospheres
                 'timestep': 1.0,           # fs
                 'timerelax': 100000,       # fs
                 'max_strain': 0.1,         # unitless
                 'rate': 1e+8,              # A/s
                 'indenter_height': 107,    # A
                 'indenter_rate': 1e-4,     # A/fs (10 m/s)
                 'indenter_radius': 30.0,   # A
                 'indenter_distance': 100.0,# A
                 'indenter_k': 10.0,        # eV/^3
                 'indenter_t_hold': 300.0,  # ps
                 'border_mols': [[0,0]],    # List of [a, b]
                 'fix_mols': [0,0],         # List of [a, b]
                 'direction': 'xz',         # shear strain
                 'pxatm': 0,                # atm
                 'pyatm': 0,                # atm
                 'deform_steps': 50,
                 'replicate': None,
                }
        _task.update(task)
        elements = [a.name for m in self.molecules for a in m.atoms]
        pattern = r'[0-9]'
        ele_string = ' '.join(elements)
        _task['ele_string'] = re.sub(pattern, '', ele_string)

        if _task['mode'] == 'bend':
            _task['pbc'] = 'slab'

        if filename is None:
            if _task['mode'] == 'bend':
                filename = _task['pbc'] + '-bend-' + _task['type'] + '.in'
            else:
                filename = _task['pbc'] + '-' + _task['type'] + '.in'

        if _task['direction'] in ['xx', 'yy', 'zz']:
            _task['direction'] = ' ' + _task['direction'][0]

        if _task['mode'] == 'uniaxial':
            keywords = _task['direction'] + ' ${patm} ${patm} ${pdamp}'
        else:
            keywords = None


        # Read the template information
        template = (rf("pyocse", "templates/" + filename))
        if not os.path.exists(template):
            raise RuntimeError('Cannot find the template', template)
        else:
            with open(template, 'r') as f0:
                lines = f0.readlines()

        with open(filename, 'w') as f:
            f.write('include lmp.in\n')
            f.write('variable temperature equal {:12.3f}\n'.format(_task['temperature']))
            f.write('variable patm equal {:12.3f}\n'.format(_task['pressure']))
            f.write('variable dt equal {:12.3f}\n'.format(_task['timestep']))
            f.write('variable trelax equal {:12.3f}\n'.format(_task['timerelax']))
            f.write('variable deform_steps equal {:12.3f}\n'.format(_task['deform_steps']))
            f.write('variable ele_string string "{:s}"\n'.format(_task['ele_string']))

            if _task['mode'] == 'uniaxial':
                f.write('variable strain_total equal {:12.3f}\n'.format(_task['max_strain']))
                f.write('variable strainrate equal {:2e}\n'.format(_task['rate']))
                f.write('variable direction string {:s}\n'.format(_task['direction']))
            else: # 3pf bending
                f.write('variable ih equal {:.3f}\n'.format(_task['indenter_height']))
                f.write('variable id equal {:.3f}\n'.format(_task['indenter_distance']))
                f.write('variable vel equal {:2e}\n'.format(_task['indenter_rate']))
                f.write('variable R equal {:.3f}\n'.format(_task['indenter_radius']))
                f.write('variable K equal {:.3f}\n'.format(_task['indenter_k']))
                f.write('variable pxatm equal {:.2f}\n'.format(_task['pxatm']))
                f.write('variable pyatm equal {:.2f}\n'.format(_task['pyatm']))
                #if _task['type'] == '3pf_free.in':
                #    f.write('variable ib equal {:.3f}\n'.format(_task['indenters_basis']))

                # Output border molecules
                if _task['border_mols'] is not None:
                    f.write('\ngroup bord molecule ')
                    for b in _task['border_mols']:
                        for _b in b:
                            f.write('{:d} '.format(_b))
                if _task['fix_mols'] is not None:
                    f.write('\ngroup fix-bord molecule ')
                    for b in _task['fix_mols']:
                        f.write('{:d} '.format(b))
                    f.write('\n')

                # update box to make sure that xlo is smaller than indenter depth
                #if _task['indenter_distance'] > : #
                #    pass

            # add replicate
            if _task['replicate'] is not None:
                rep = [str(x) for x in _task['replicate']]
                f.write('replicate   ')
                f.write(' '.join(rep))
                f.write('\n')

            for line in lines:
                if not line.startswith('#variable'):
                    if keywords is not None and line.find(keywords) > 0:
                        line = line.replace(keywords, '')
                    f.write(line)
            #f.writelines(lines)

    def get_dim(self, direction):
        # For tensile, the preferred direction should be no less than 80
        # For shear, the preferred plane should be no less than 80*80
        # Other directions should be no less than 20
        if direction == 'xx':
            return [80, 40, 40]
        elif direction  == 'yy':
            return [40, 80, 40]
        elif direction  == 'zz':
            return [40, 40, 80]
        elif direction  == 'xy':
            return [80, 80, 40]
        elif direction  == 'xz':
            return [80, 40, 80]
        elif direction  == 'yz':
            return [40, 80, 80]
        else:
            raise RuntimeError("direction is not supported")

    def get_deform_type(self, direction):
        # For tensile, the preferred direction should be no less than 80
        # For shear, the preferred plane should be no less than 80*80
        # Other directions should be no less than 20
        if direction in ['xx', 'yy', 'zz']:
            return 'tension'
        elif direction == ['xy', 'yz', 'xz']:
            return 'shear'
        else:
            raise RuntimeError("direction is not supported")

    def plot(self, direction, label=None):
        import matplotlib.pyplot as plt
        # Create a single figure with subplots for each column
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

        if direction == 'xx':
            col = 2
        elif direction  == 'yy':
            col = 3
        elif direction  == 'zz':
            col = 4
        elif direction == 'xz':
            col = 5
        elif direction == 'yz':
            col = 6
        elif direction == 'xy':
            col = 7

        name = 'Cycle-' + direction
        if label is not None: name = label + '-' + name
        fig.suptitle(name, fontsize=16)

        # Load the data from the text file into a DataFrame
        file_path = 'output.dat'
        data = np.loadtxt(file_path)
        middle = int(len(data)/2)

        d = data[0, 1]
        for i in range(middle, len(data)):
            #print(i, d*(i-middle))
            data[i, 1] -= 2*d*(i-middle+2)
        #print(data[:, 1])

        # Plot the distribution of each column
        labels = ['S_' + direction + ' (MPa)', 'Potential Energy (Kcal/mol)']
        for i, column in enumerate([col, -1]):#, 5, 6, 8]):
            row, col = i, 0
            if row == 0:
                colors = ['r', 'b']
            else:
                colors = ['r', 'g']
            axs[row].plot(data[:middle, 1], data[:middle, column], color=colors[0], label = 'Forward')
            axs[row].plot(data[middle:, 1], data[middle:, column], color=colors[1], label = 'Backward')
            axs[row].grid(True)
            axs[row].legend(loc=1)
            axs[row].set_ylabel(labels[i])
            if i == 1:
                axs[row].set_xlabel('Strain')
            else:
                axs[row].set_xticks([])
        # Adjust the layout and spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.01)
        # Save the figure as '1.png'
        plt.savefig(name+'.png')


if __name__ == "__main__":

    import os

    #=== Set the crystal model
    cif  = "pyocse/data/ACSALA17.cif"
    toml_file = "pyocse/data/aspirin_gas.toml"
    bu = Builder(toml_files=[toml_file])
    bu.set_xtal(cif=cif)

    #=== Apply the orientation

    #=== Prepare lammps inputs
    # Example 1: tensile, assuming z direction
    task1 = {
             'mode': 'uniaxial',
             'direction': 'xx',
             'max_strain': 0.1,
             'rate': 1e+8,
           }

    # Example 2: shear
    task2 = {
             'mode': 'uniaxial',
             'direction': 'xz',
             'max_strain': 0.1,
             'rate': 1e+8,
            }

    # Example 3: bending
    task3 = {
             'mode': 'bend',
             'indenter_rate': 1e-3,     # A/fs (100 m/s)
             'border_mols': bord_ids,   # for thermostat
             'fix_mols': fix_ids,       # for freezing
             'z_max': z_max,
            }

    tasks = [task1, task2, task3]
    #=== Directory
    folder = "test"
    if not os.path.exists(folder): os.makedirs(folder)
    cwd = os.getcwd()
    os.chdir(folder)

    for task in tasks:
        bu.set_slab(bu.xtal, bu.xtal_mol_list, hkl=(0,1,0), replicate=[5, 10, 8], layers=10, vacuum=20, orthogonality=True)
        bord_ids = bu.get_molecular_bord_ids(bu.ase_slab, bu.ase_slab_mol_list)
        fix_ids = bu.get_molecular_fix_ids(bu.ase_slab, bu.ase_slab_mol_list)
        zs = bu.ase_slab.get_positions()[:,2]
        z_max, z_min = zs.max(), zs.min()
        print('Supercell:  ', bu.ase_slab.get_cell_lengths_and_angles())

        bu.set_task(task)
        if task['mode'] in ['bend', 'bend0']:
                bu.lammps_slab.write_lammps(orthogonality=True)
                bu.dump_slab_centers(bu.ase_slab, bu.ase_slab_mol_list, bord_ids, fix_ids)