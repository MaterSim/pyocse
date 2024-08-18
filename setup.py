#from setuptools import find_packages, setup
from distutils.core import setup
import setuptools  # noqa

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="pyocse",
    version="0.1.1",
    author="Qiang Zhu, Shinnosule Hattori",
    description="Python Organic Crystal Simulation Environment",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaterSim/pyocse",
    packages=[
        "pyocse",
        "pyocse.interfaces",
        "pyocse.lmp",
        "pyocse.charmm",
        "pyocse.data",
        "pyocse.templates",
    ],
    package_data={
        "pyocse.data": ["*"],
        "pyocse.templates": ['*.in'],
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
