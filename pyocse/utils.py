#!/usr/bin/env python3
import contextlib
import os
import shutil
from shutil import which
import subprocess
import tempfile
from decimal import ROUND_05UP, ROUND_HALF_EVEN, Decimal
from time import sleep
from xml.dom.minidom import parseString
import numpy as np

import toml
from xmltodict import unparse
import xml.etree.ElementTree as ET

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

def compute_r2(y_true, y_pred):
    """
    Compute the R-squared coefficient for the actual and predicted values.

    Args:
        y_true: The actual values.
        y_pred: The predicted values by the regression model.

    Return:
        The R-squared value.
    """
    if len(y_true) > 0:
        # Calculate the mean of actual values
        mean_y_true = sum(y_true) / len(y_true)

        # Total sum of squares (SST)
        sst = sum((y_i - mean_y_true) ** 2 for y_i in y_true)

        # Residual sum of squares (SSE)
        sse = sum((y_true_i - y_pred_i) ** 2 for y_true_i, y_pred_i in zip(y_true, y_pred))

        # R-squared
        r2 = 1 - (sse / sst)
    else:
        r2 = 0

    return r2

def string_to_array(s, dtype=float):
    """Converts a formatted string back into a 1D or 2D NumPy array."""
    # Split the string into lines
    lines = s.strip().split('\n')

    # Check if it's a 1D or 2D array based on the number of lines
    if len(lines) == 1:
        # Treat as 1D array if there's only one line
        array = np.fromstring(lines[0][1:-1], sep=',', dtype=dtype)
        #print(lines); print(lines[0][1:-1]); print(array); import sys; sys.exit()
    else:
        # Treat as 2D array if there are multiple lines
        array = [np.fromstring(line, sep=' ', dtype=dtype) for line in lines if line]
        array = np.array(array, dtype=dtype)

    return array

def array_to_string(arr):
    """Converts a 2D NumPy array to a string format with three numbers per line."""
    lines = []
    for row in arr:
        for i in range(0, len(row), 3):
            line_segment = ' '.join(map(str, row[i:i+3]))
            lines.append(line_segment)
    return '\n' + '\n'.join(lines) + '\n'

def get_lmp_efs(lmp_struc, lmp_in, lmp_dat):
    from pyocse.lmp import LAMMPSCalculator

    if not hasattr(lmp_struc, 'ewald_error_tolerance'):
        lmp_struc.complete()
    calc = LAMMPSCalculator(lmp_struc, lmp_in=lmp_in, lmp_dat=lmp_dat)
    return calc.express_evaluation()

def xml_to_dict_list(filename):
    """
    Parse the XML file and return a list of dictionaries.
    """
    import ast

    tree = ET.parse(filename)
    root = tree.getroot()

    data = []
    for item in root.findall('structure'):
        item_dict = {}
        for child in item:
            key = child.tag
            text = child.text.strip()
            # Check if the field should be converted back to an array
            if key in ['lattice', 'position', 'forces', 'numbers', 'stress',
                       'bond', 'angle', 'proper', 'vdW', 'charge', 'offset',
                       'rmse_values', 'r2_values', 'numMols']:

                if text != 'None':
                    #print(text)
                    if key in ['numbers', 'numMols']:
                        value = string_to_array(text, dtype=int)
                    else:
                        value = string_to_array(text)
                else:
                    value = None
            elif key in ['options']:
                value = ast.literal_eval(text) #print(value)
            else:
                # Attempt to convert numeric values back to float/int
                try:
                    value = float(text)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    if text == 'None':
                        value = None
                    else:
                        value = text

            item_dict[key] = value
        #print(item_dict.keys())
        data.append(item_dict)
    return data