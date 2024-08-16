#from setuptools import find_packages, setup
from distutils.core import setup
import setuptools  # noqa

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="ost",
    version="0.1.0",
    author="Qiang Zhu, Shinnosule Hattori",
    description="Organic Simulation Toolkit",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaterSim/OST",
    packages=[
        "ost",
        "ost.interfaces",
        "ost.lmp",
        "ost.charmm",
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
