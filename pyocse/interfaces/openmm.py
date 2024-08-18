#!/usr/bin/env python


import numpy as np
from openmm import LangevinMiddleIntegrator, NonbondedForce
from openmm import unit as u
from openmm.app import PME, PDBReporter, Simulation, StateDataReporter
from pandas import read_csv
from parmed.openmm import energy_decomposition_system

from xtalmol.interfaces.parmed import ParmEdStructure


class OPENMMCalculator:
    def __init__(self, struc, out_iter=10):
        self.struc = struc
        self.out_iter = out_iter
        self.sim = None

    def run(self, step=10, temperature=300.0, minimize=False, rebuild=False):

        struc = self.struc
        temperature *= u.kelvin
        dt = 0.002 * u.picosecond
        tau = 0.30 / u.picosecond
        coords = struc.coordinates * u.angstrom

        if self.sim is None or rebuild is True:
            box_tmpset = False
            if struc.box is None:
                box_tmpset = True
                struc.box = struc.LARGEBOX
            box = struc.box_vectors

            integrator = LangevinMiddleIntegrator(temperature, tau, dt)
            system = struc.createSystem(
                nonbondedMethod=PME,
                switchDistance=1.725 * (struc.cutoff_ljout - struc.cutoff_ljin) * u.angstroms,
                nonbondedCutoff=struc.cutoff_coul * u.angstroms,
                constraints=None,
                rigidWater=False,
                ewaldErrorTolerance=struc.ewald_error_tolerance,
                verbose=True,
            )
            sim = Simulation(struc.topology, system, integrator)
            sim.context.setPeriodicBoxVectors(*box)
            sim.context.setPositions(coords)
            sim.context.setVelocitiesToTemperature(temperature)

        else:
            sim = self.sim
            sim.integrator.setTemperature(temperature)
            sim.integrator.setStepSize(dt)
            sim.integrator.setFriction(tau)

        if minimize:
            sim.minimizeEnergy()
        # Configure the information in the output files.
        pdb_reporter = PDBReporter("trajectory.pdb", self.out_iter, enforcePeriodicBox=True)
        state_data_reporter = StateDataReporter(
            "data.csv",
            self.out_iter,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            progress=True,
            speed=True,
            totalSteps=step,
        )
        sim.reporters.append(pdb_reporter)
        sim.reporters.append(state_data_reporter)
        sim.step(step)

        if box_tmpset:
            struc.box = None
        thermo = read_csv("data.csv")

        self.sim = sim
        return thermo

    def get_energy(self):
        self.out_iter = 1
        self.run(step=1, temperature=10.0, minimize=False)
        energy = energy_decomposition_system(self.struc, self.sim.system)
        dict = self.struc.generate_energy_dict(energy)

        charges = [atom.charge for atom in self.struc.atoms]
        self.struc.set_charges(np.zeros(len(self.struc.atoms)))
        self.run(step=1, temperature=10.0, minimize=False, rebuild=True)
        energy_noq = energy_decomposition_system(self.struc, self.sim.system)
        dict_noq = self.struc.generate_energy_dict(energy_noq)
        dict["vdw"] = dict_noq["nonbonded"]
        dict["coul"] = dict["nonbonded"] - dict_noq["nonbonded"]

        self.struc.set_charges(charges)
        return dict
