#!/usr/bin/env python3
import contextlib
import os
import shutil
from shutil import which
import subprocess
import tempfile
import warnings
from decimal import ROUND_05UP, ROUND_HALF_EVEN, Decimal
from time import sleep
from xml.dom.minidom import parseString
import numpy as np

import toml
from xmltodict import unparse

warnings.filterwarnings("ignore")

def reset_lammps_cell(atoms0):
    from ase.calculators.lammpslib import convert_cell

    atoms = atoms0.copy()
    mat, coord_transform = convert_cell(atoms0.cell)
    if coord_transform is not None:
        pos = np.dot(atoms0.positions, coord_transform.T)
        atoms.set_cell(mat.T)
        atoms.set_positions(pos)
    return atoms



@contextlib.contextmanager
def temporary_cd(dir_path):
    """Context to temporary change the working directory."""
    prev_dir = os.getcwd()
    os.chdir(os.path.abspath(dir_path))
    try:
        yield
    finally:
        os.chdir(prev_dir)


@contextlib.contextmanager
def mkdir_cd(dir_path):
    """Context to mkdir and temporary change the working directory."""
    os.makedirs(dir_path, exist_ok=True)
    with temporary_cd(dir_path):
        yield dir_path


@contextlib.contextmanager
def temporary_directory(cleanup=True, prefix=None):
    """Context for safe creation of temporary directories."""
    prev_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix=prefix, dir=prev_dir)
    try:
        yield tmp_dir
    finally:
        if cleanup:
            try:
                sleep(1)
                shutil.rmtree(tmp_dir)
            except BaseException:
                print("cant remove tmp dir:", tmp_dir)


@contextlib.contextmanager
def temporary_directory_change(cleanup=True, prefix=None):
    with temporary_directory(cleanup, prefix) as tmpdir:
        with temporary_cd(tmpdir):
            yield tmpdir


def dump_toml(dict, fname):
    return toml.dump(dict, open(fname, "w"), encoder=toml.TomlNumpyEncoder())


def load_toml(fname):
    return toml.load(open(fname))


def dict_to_xmlstr(dict):
    xmlstr = unparse(dict)
    xmlstr = parseString(xmlstr).toprettyxml()
    return xmlstr


def procrun(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.SubprocessError as e:
        print("Error:", e)
    except BaseException as e:
        raise BaseException(f"Unexpected error in subprocess: {cmd} {e}")


def accurate_round(v, nofdigit=4):
    fmt0 = "." + nofdigit * "0" + "1"
    fmt1 = "." + (nofdigit - 1) * "0" + "1"
    v = Decimal(v).quantize(Decimal(fmt0), rounding=ROUND_HALF_EVEN)
    v = v.quantize(Decimal(fmt1), rounding=ROUND_05UP)
    return float(v)


def which_lmp(add=None):

    candidates = ["lmp", "lmp_mpi", "lmp_serial", "lmp_openmp", "lmp_kokkos_cuda_mpi"]
    if add is not None:
        candidates = [add, *candidates]

    for candidate in candidates:
        if which(candidate):
            return candidate
    return None

def find_outliers(data, threshold=3):
    """
    Find the outliers in the data according to the correlation

    Args:
        data (array): data array (N, 2)
        threshold (float): threshold for the outlier
    """
    from scipy.spatial.distance import mahalanobis

    # Compute the mean and covariance matrix
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data.T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distance for each point
    distances = [mahalanobis(point, mean, inv_cov_matrix) for point in data]

    # Set a threshold (e.g., 95th percentile)
    threshold = np.percentile(distances, threshold)
    outliers = np.where(distances > threshold)
    return outliers