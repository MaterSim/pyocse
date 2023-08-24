#from setuptools import find_packages, setup
from distutils.core import setup
import setuptools  # noqa
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

exec(open("ost/version.py").read())

setup(
    name="ost",
    version=__version__,
    author="Qiang Zhu, Shinnosule Hattori",
    description="Organic Simulation Toolkit",
    include_package_data=True,
    packages=[
        "ost",
        "ost.interfaces",
        "ost.lmp",
        "ost.data",
        "ost.templates",
    ],
    package_data={
        "ost.data": ["*"],
        "ost.templates": ['*.in'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #install_requires=[
        #"parmed>=3.4.3", 
        #"openmm>=7.6.0", 
        #"toml", 
        #"xmltodict"
    #],
    #python_requires=">=3.7, <=3.10",
    license="MIT",
)
