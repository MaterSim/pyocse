#!/usr/bin/env python

import ctypes
import os

import numpy as np
from ase.calculators.lammps import convert
from ase.calculators.lammpslib import LAMMPSlib
from ase.geometry import wrap_positions
from ase import units

from pyocse.utils import which_lmp

seed = 123456789


class LAMMPSCalculatorMixIn:
    def _easy_run(self, n, precmds=None, postcmds=None):
        for cmd in precmds:
            self.lmp.command(cmd)
        self.lmp.run(n)
        for cmd in postcmds:
            self.lmp.command(cmd)

    def easy_prepostcmds(self, ens):
        if "plumed" in ens:
            pre, post = self._easy_prepostcmds_plumed(ens)
        else:
            pre, post = self._easy_prepostcmds(ens)
        return pre, post

    def _easy_prepostcmds(self, ens, tdump=100.0, pdump=20000.0):
        precmds = []
        postcmds = []

        if "temperature" in ens:
            if isinstance(ens["temperature"], str) and ":" in ens["temperature"]:
                t0, t1 = ens["temperature"].split(":")
            else:
                t0 = ens["temperature"]
                t1 = t0
            precmds += [
                "velocity all create {} {} rot yes dist gaussian".format(t0, seed)
            ]

        if "pressure" in ens:
            if isinstance(ens["pressure"], str) and ":" in ens["pressure"]:
                p0, p1 = ens["pressure"].split(":")
            else:
                p0 = ens["pressure"]
                p1 = p0
            if "pressure_keyword" in ens:
                key = ens["pressure_keyword"]
            else:
                key = "iso"

        if "temperature" not in ens and "pressure" not in ens:
            precmds += ["fix ensamble all nve"]
        elif "temperature" in ens and "pressure" not in ens:
            precmds += ["fix ensamble all nvt temp {} {} {}".format(t0, t1, tdump)]
        elif "temperature" not in ens and "pressure" in ens:
            precmds += ["fix ensamble all nph {} {} {} {}".format(key, p0, p1, pdump)]
        elif "temperature" in ens and "pressure" in ens:
            precmds += [
                "fix ensamble all npt temp {} {} {} {} {} {} {}".format(
                    t0, t1, tdump, key, p0, p1, pdump
                )
            ]
        postcmds += ["unfix ensamble"]

        return precmds, postcmds

    def _easy_prepostcmds_plumed(self, ens, tdump=100.0, pdump=20000.0):
        precmds = []
        postcmds = []

        precmds += [
            "fix plumed all plumed plumedfile {} outfile lmp.log_plumed".format(
                ens["plumed"]
            )
        ]
        postcmds += ["unfix plumed"]

        if "temperature" not in ens:
            raise

        if type(ens["temperature"]) is str and ":" in ens["temperature"]:  # noqa: E721
            t0, t1 = ens["temperature"].split(":")
        else:
            t0 = ens["temperature"]
            t1 = t0
        precmds += ["velocity all create {} {} rot yes dist gaussian".format(t0, seed)]

        precmds += ["fix tcont all temp/csvr {} {} {} {}".format(t0, t1, tdump, seed)]
        # precmds += ["fix tcont all temp/csld {} {} {} {}".format(t0, t1, tdump, seed)] # langevin dynamics
        postcmds += ["unfix tcont"]

        if "pressure" in ens:
            if isinstance(ens["pressure"], str) and ":" in ens["pressure"]:
                p0, p1 = ens["pressure"].split(":")
            else:
                p0 = ens["pressure"]
                p1 = p0
            if "pressure_keyword" in ens:
                key = ens["pressure_keyword"]
            else:
                key = "iso"

        if "pressure" not in ens:
            precmds += ["fix ensamble all nve"]
        else:
            precmds += ["fix ensamble all nph {} {} {} {}".format(key, p0, p1, pdump)]
        postcmds += ["unfix ensamble"]
        precmds += ["fix reset all momentum 10000 linear 1 1 1 angular"]
        postcmds += ["unfix reset"]

        return precmds, postcmds

    def easy_run(self, params):
        """
        easy run for lammps using dict of md conditions
        inputs:
            params: dict
        """

        for prm in params:
            if "step" not in prm:
                print("skip", prm)
            else:
                precmds, postcmds = self.easy_prepostcmds(prm)
                if "timestep" in prm:
                    self.lmp.timestep(float(prm["timestep"]))
                if "minimize" in prm and prm["minimize"]:
                    self.minimize(nstep=prm["step"])
                else:
                    self._easy_run(prm["step"], precmds, postcmds)

    def get_energy(self):
        precmds = [
            "fix ensemble all nve",
        ]
        postcmds = ["unfix ensemble"]
        # self.lmp
        self._easy_run(0, precmds, postcmds)
        thermo = self.lmp.last_run[-1]
        energy = {}
        energy["bond"] = float(thermo.E_bond[-1])
        energy["angle"] = float(thermo.E_angle[-1])
        # energy["_dihedral"] = float(thermo.E_dihed[-1])
        # energy["_improper"] = float(thermo.E_impro[-1])
        energy["torsion"] = float(thermo.E_dihed[-1])
        energy["nonbonded"] = float(thermo.E_pair[-1])
        energy["vdw"] = float(thermo.E_vdwl[-1])
        energy["long"] = float(thermo.E_long[-1])
        energy["short"] = float(thermo.E_coul[-1])
        energy["coul"] = energy["short"] + energy["long"]
        energy["tail"] = float(thermo.E_tail[-1])
        energy["total"] = (
            energy["nonbonded"] + energy["bond"] + energy["angle"] + energy["torsion"]
        )
        energy["pot"] = float(thermo.PotEng[-1])
        return energy

    def express_evaluation(self):
        """
        Express evaluation of single point energy/force/stress

        lammps real units:
        energy = kcal/mol
        time = femtoseconds
        force = (kcal/mol)/Angstrom
        pressure = atmospheres
        """
        #precmds = [
        #    "fix ensemble all nve",
        #    "variable pxx equal pxx",
        #    "variable pyy equal pyy",
        #    "variable pzz equal pzz",
        #    "variable pyz equal pyz",
        #    "variable pxz equal pxz",
        #    "variable pxy equal pxy",
        #    "variable fx atom fx",
        #    "variable fy atom fz",
        #    "variable fz atom fz",
        #]
        #for cmd in precmds:
        #    self.lmp.command(cmd)
        #self.lmp.run(0)

        #self._easy_run(0, precmds, postcmds)
        #from time import time; t0 = time()
        self.lmp.run(0)
        #t1 = time(); print('stress', t1-t0)
        thermo = self.lmp.last_run[-1]
        energy = float(thermo.TotEng[-1]) * units.kcal/units.mol
        stress = np.zeros(6)
        # traditional Voigt order (xx, yy, zz, yz, xz, xy)
        stress_vars = ['pxx', 'pyy', 'pzz', 'pyz', 'pxz', 'pxy']
        for i, var in enumerate(stress_vars):
            stress[i] = self.lmp.variables[var].value
        #t1 = time(); print('stress', t1-t0)

        fx = np.frombuffer(self.lmp.variables['fx'].value)
        fy = np.frombuffer(self.lmp.variables['fy'].value)
        fz = np.frombuffer(self.lmp.variables['fz'].value)
        #t2 = time(); print('forces', t2-t1)

        stress = -stress * 101325 * units.Pascal
        forces = np.vstack((fx, fy, fz)).T * units.kcal/units.mol

        return energy, forces, stress


class LAMMPSAseCalculator(LAMMPSlib, LAMMPSCalculatorMixIn):
    default_parameters = dict(
        atom_types=None,
        atom_type_masses=None,
        log_file="aselog.lammps",
        lammps_name=None,
        keep_alive=True,
        lammps_header=[
            "units real",
            "atom_style full",
            "atom_modify map array sort 0 0",
            "box tilt large",
        ],
        amendments=[],
        post_changebox_cmds=None,
        boundary=True,
        create_box=True,
        create_atoms=True,
        read_molecular_info=True,
        comm=None,
    )

    def __init__(self, dumpdir="outputs", nproc=1, *args, **kwargs):
        self.dumpdir = dumpdir
        self.nproc = nproc
        os.environ["OMP_NUM_THREADS"] = str(nproc)
        super().__init__(*args, **kwargs)

        if self.parameters.lammps_name:
            binary_name = which_lmp("lmp_" + self.parameters.lammps_name)
        else:
            binary_name = which_lmp()
        if binary_name == "lmp":
            lammps_name = ""
        else:
            lammps_name = binary_name.split("lmp_")[-1]
        self.parameters.lammps_name = lammps_name

    def _easy_run(self, *args, **kwargs):
        # todo
        # do propagate_pre
        super()._easy_run(*args, **kwargs)
        # do propagate_post

    def build(self, atoms):
        info = atoms.info["lammps"]
        for i, (_, resname, nmol) in enumerate(info["lammps_molecules"]):
            cmd = "create_atoms 0 random {} {} NULL mol mol{} {} #{}".format(
                nmol, seed, i, seed, resname
            )
            self.lmp.command(cmd)

        # self.previous_atoms_numbers = atoms.numbers.copy()

    def initialise_lammps(self, atoms):
        info = atoms.info["lammps"]

        # Initialising commands
        if self.parameters.boundary:
            # if the boundary command is in the supplied commands use that
            # otherwise use atoms pbc
            self.lmp.command("boundary " + self.lammpsbc(atoms))

        # Initialize cell
        self.set_cell(atoms, change=not self.parameters.create_box)

        if self.parameters.atom_types is None:
            # if None is given, create from atoms object in order of appearance
            s = atoms.get_chemical_symbols()
            _, idx = np.unique(s, return_index=True)
            s_red = np.array(s)[np.sort(idx)].tolist()
            self.parameters.atom_types = {j: i + 1 for i, j in enumerate(s_red)}

        # Initialize molecules
        # Todo multi molecule support
        if self.parameters.read_molecular_info:
            # Initialize box
            if self.parameters.create_box:
                # count number of known types
                self.lmp.command(info["create_box_command"])

            for i, (molecule, _resname, _nmol) in enumerate(info["lammps_molecules"]):
                fmol = f"lmp{i}.molecule"
                open(fmol, "w").write(molecule)
                self.lmp.command("molecule mol{} {}".format(i, fmol))

        # Initialize the atoms with their types
        # positions do not matter here
        if self.parameters.create_atoms:
            self.lmp.command("echo none")  # don't echo the atom positions
            self.build(atoms)
            self.lmp.command("echo log")  # turn back on
        else:
            self.previous_atoms_numbers = atoms.numbers.copy()

        # execute the user commands
        for cmd in info["style_commands"]:
            self.lmp.command(cmd)

        # Set masses after user commands, e.g. to override
        # Set coeff
        for cmd in info["coeff_commands"]:
            self.lmp.command(cmd)

        # Define force & energy variables for extraction
        self.lmp.command("variable pxx equal pxx")
        self.lmp.command("variable pyy equal pyy")
        self.lmp.command("variable pzz equal pzz")
        self.lmp.command("variable pxy equal pxy")
        self.lmp.command("variable pxz equal pxz")
        self.lmp.command("variable pyz equal pyz")

        # I am not sure why we need this next line but LAMMPS will
        # raise an error if it is not there. Perhaps it is needed to
        # ensure the cell stresses are calculated
        # self.lmp.command('thermo_style custom pe pxx emol ecoul')
        cmd = (
            "thermo_style custom step temp vol press etotal pe ke epair ecoul elong evdwl ebond eangle edihed eimp emol etail \
    cella cellb cellc cellalpha cellbeta cellgamma density pxx pyy pzz pxy pxy pyz"
        )
        self.lmp.command(cmd)
        self.lmp.command("thermo_modify lost ignore flush yes")

        self.lmp.command("variable fx atom fx")
        self.lmp.command("variable fy atom fy")
        self.lmp.command("variable fz atom fz")

        # do we need this if we extract from a global ?
        self.lmp.command("variable pe equal pe")

        # self.lmp.command("neigh_modify delay 0 every 1 check yes")
        # self.lmp.command("neigh_modify delay 10 every 2")

        # dump settings
        self.lmp.command("compute pe_all all pe/atom")
        self.lmp.command("compute pe_pair all pe/atom pair")
        sthermo = 500
        sdump = 1000
        self.lmp.command(f"thermo {sthermo}")
        if not os.path.exists(self.dumpdir):
            os.mkdir(self.dumpdir)
        cmd = f"dump mydump all custom {sdump} {self.dumpdir}/out.*.lammpstrj id mol type q x y z ix iy iz c_pe_all c_pe_pair element"
        self.lmp.command(cmd)

        cmd = "dump_modify mydump element %s pad 9 sort id" % " ".join(
            info["element_list"]
        )
        self.lmp.command(cmd)

        self.initialized = True

    def propagate(
        self,
        atoms,
        properties,
        system_changes,
        n_steps,
        dt=None,
        dt_not_real_time=False,
        velocity_field=None,
    ):
        """ "atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'charges' and 'magmoms'.
        """
        if len(system_changes) == 0:
            return

        self.coord_transform = None

        if not self.started:
            self.start_lammps()
        if not self.initialized:
            self.initialise_lammps(atoms)
        else:  # still need to reset cell
            # NOTE: The whole point of ``post_changebox_cmds`` is that they're
            # executed after any call to LAMMPS' change_box command.  Here, we
            # rely on the fact that self.set_cell(), where we have currently
            # placed the execution of ``post_changebox_cmds``, gets called
            # after this initial change_box call.

            # Apply only requested boundary condition changes.  Note this needs
            # to happen before the call to set_cell since 'change_box' will
            # apply any shrink-wrapping *after* it's updated the cell
            # dimensions
            if "pbc" in system_changes:
                change_box_str = "change_box all boundary {}"
                change_box_cmd = change_box_str.format(self.lammpsbc(atoms))
                self.lmp.command(change_box_cmd)

            # Reset positions so that if they are crazy from last
            # propagation, change_box (in set_cell()) won't hang.
            # Could do this only after testing for crazy positions?
            # Could also use scatter_atoms() to set values (requires
            # MPI comm), or extra_atoms() to get pointers to local
            # data structures to zero, but then we would have to be
            # careful with parallelism.
            self.lmp.command("set atom * x 0.0 y 0.0 z 0.0")
            self.set_cell(atoms, change=True)

        if self.parameters.atom_types is None:
            raise NameError("atom_types are mandatory.")

        # do_rebuild = (not np.array_equal(atoms.numbers,
        #                                  self.previous_atoms_numbers)
        #               or ("numbers" in system_changes))
        # if not do_rebuild:
        #     do_redo_atom_types = not np.array_equal(
        #         atoms.numbers, self.previous_atoms_numbers)
        # else:
        #     do_redo_atom_types = False

        # self.lmp.command('echo none')  # don't echo the atom positions
        # if do_rebuild:
        #     self.rebuild(atoms)
        # elif do_redo_atom_types:
        #     self.redo_atom_types(atoms)
        # self.lmp.command('echo log')  # switch back log

        self.set_lammps_pos(atoms)

        if self.parameters.amendments is not None:
            for cmd in self.parameters.amendments:
                self.lmp.command(cmd)

        if n_steps > 0:
            if velocity_field is None:
                vel = convert(atoms.get_velocities(), "velocity", "ASE", self.units)
            else:
                # FIXME: Do we need to worry about converting to lammps units
                # here?
                vel = atoms.arrays[velocity_field]

            # If necessary, transform the velocities to new coordinate system
            if self.coord_transform is not None:
                vel = np.dot(self.coord_transform, vel.T).T

            # Convert ase velocities matrix to lammps-style velocities array
            lmp_velocities = list(vel.ravel())

            # Convert that lammps-style array into a C object
            c_double_array = ctypes.c_double * len(lmp_velocities)
            lmp_c_velocities = c_double_array(*lmp_velocities)
            self.lmp.scatter_atoms("v", 1, 3, lmp_c_velocities)

        # Run for 0 time to calculate
        if dt is not None:
            if dt_not_real_time:
                self.lmp.command("timestep %.30f" % dt)
            else:
                self.lmp.command(
                    "timestep %.30f" % convert(dt, "time", "ASE", self.units)
                )
        self.lmp.command("run %d" % n_steps)

        if n_steps > 0:
            # TODO this must be slower than native copy, but why is it broken?
            pos = np.array([x for x in self.lmp.gather_atoms("x", 1, 3)]).reshape(-1, 3)
            if self.coord_transform is not None:
                pos = np.dot(pos, self.coord_transform)

            # Convert from LAMMPS units to ASE units
            pos = convert(pos, "distance", self.units, "ASE")

            atoms.set_positions(pos)

            vel = np.array([v for v in self.lmp.gather_atoms("v", 1, 3)]).reshape(-1, 3)
            if self.coord_transform is not None:
                vel = np.dot(vel, self.coord_transform)
            if velocity_field is None:
                atoms.set_velocities(convert(vel, "velocity", self.units, "ASE"))

        # Extract the forces and energy
        self.results["energy"] = convert(
            self.lmp.extract_variable("pe", None, 0), "energy", self.units, "ASE"
        )
        self.results["free_energy"] = self.results["energy"]

        stress = np.empty(6)
        stress_vars = ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]

        for i, var in enumerate(stress_vars):
            stress[i] = self.lmp.extract_variable(var, None, 0)

        stress_mat = np.zeros((3, 3))
        stress_mat[0, 0] = stress[0]
        stress_mat[1, 1] = stress[1]
        stress_mat[2, 2] = stress[2]
        stress_mat[1, 2] = stress[3]
        stress_mat[2, 1] = stress[3]
        stress_mat[0, 2] = stress[4]
        stress_mat[2, 0] = stress[4]
        stress_mat[0, 1] = stress[5]
        stress_mat[1, 0] = stress[5]
        if self.coord_transform is not None:
            stress_mat = np.dot(
                self.coord_transform.T, np.dot(stress_mat, self.coord_transform)
            )
        stress[0] = stress_mat[0, 0]
        stress[1] = stress_mat[1, 1]
        stress[2] = stress_mat[2, 2]
        stress[3] = stress_mat[1, 2]
        stress[4] = stress_mat[0, 2]
        stress[5] = stress_mat[0, 1]

        self.results["stress"] = convert(-stress, "pressure", self.units, "ASE")

        # definitely yields atom-id ordered force array
        f = convert(
            np.array(self.lmp.gather_atoms("f", 1, 3)).reshape(-1, 3),
            "force",
            self.units,
            "ASE",
        )

        if self.coord_transform is not None:
            self.results["forces"] = np.dot(f, self.coord_transform)
        else:
            self.results["forces"] = f.copy()

        # self.thermo = get_thermo_data(self.parameters.log_file)

        # otherwise check_state will always trigger a new calculation
        self.atoms = atoms.copy()

        if not self.parameters.keep_alive:
            self.lmp.close()

    def set_lammps_pos(self, atoms, ignore_wrap=True):
        # Create local copy of positions that are wrapped along any periodic
        # directions
        cell = convert(atoms.cell, "distance", "ASE", self.units)
        pos = convert(atoms.positions, "distance", "ASE", self.units)

        # If necessary, transform the positions to new coordinate system
        if self.coord_transform is not None:
            pos = np.dot(pos, self.coord_transform.T)
            cell = np.dot(cell, self.coord_transform.T)

        # wrap only after scaling and rotating to reduce chances of
        # lammps neighbor list bugs.
        if not ignore_wrap:
            pos = wrap_positions(pos, cell, atoms.get_pbc())

        # Convert ase position matrix to lammps-style position array
        # contiguous in memory
        lmp_positions = list(pos.ravel())

        # Convert that lammps-style array into a C object
        c_double_array = ctypes.c_double * len(lmp_positions)
        lmp_c_positions = c_double_array(*lmp_positions)
        #        self.lmp.put_coosrds(lmp_c_positions)
        self.lmp.command("set atom * image 0 0 0")
        self.lmp.scatter_atoms("x", 1, 3, lmp_c_positions)


class LAMMPSCalculator(LAMMPSCalculatorMixIn):
    def __init__(
        self,
        struc,
        base="lmp",
        dumpdir="outputs",
        nproc=1,
        lammps_name=None,
        lmp_instance=None,
        lmp_in=None,
        lmp_dat=None,
        skip_dump = True,
        coulcut = False,
        workdir = '.',
        *args,
        **lwargs,
    ):

        self.struc = struc
        self.base = base
        self.workdir = workdir,
        cmdargs = ["-screen", "none", "-log", f"{workdir}/{base}.log", "-nocite"]
        self.lin = f"{workdir}/{base}.in"
        self.ldat = f"{workdir}/{base}.dat"
        self.dumpdir = dumpdir
        self.coulcut = coulcut

        self.nproc = nproc
        if self.nproc > 1:
            cmdargs += ["-sf", "omp"]
        os.environ["OMP_NUM_THREADS"] = str(nproc)

        # Set up the lammps instance
        if lmp_instance is not None:
            self.lmp = lmp_instance
            self.lmp.command('clear')
        else:
            from lammps import PyLammps  # , get_thermo_data
            if lammps_name:
                binary_name = which_lmp("lmp_" + lammps_name)
            else:
                binary_name = which_lmp()
            #print("binary name: ", binary_name)
            if binary_name == "lmp":
                lammps_name = ""
            else:
                lammps_name = None
                # lammps_name = binary_name.split("lmp_")[-1]
            self.lmp = PyLammps(name=lammps_name, cmdargs=cmdargs, *args, **lwargs)

        #from time import time; t0 = time()
        if lmp_in is None:
            struc.write_lammps(fin=self.lin, fdat=self.ldat, lmp_dat=lmp_dat)
        else:
            struc.write_lammps(fin=self.lin, fdat=self.ldat, fin_template=lmp_in, lmp_dat=lmp_dat)

        #t1=time(); print("lmp_instance", t1-t0)
        self.restart = False
        self.initialize()
        if not skip_dump:
            if not os.path.exists(dumpdir): os.mkdir(dumpdir)
            self.compute_and_dump_settings()
        #t2=time(); print("lmp_initialize", t2-t1)

    def initialize(self):
        #lines = open(self.lin).readlines()
        #for line0 in lines:
        #    line = line0.strip()
        #    if len(line) == 0 or line[0] == "#":
        #        continue
        #    self.lmp.command(line)
        self.lmp.file(self.lin)

    def compute_and_dump_settings(self, thermo=500, dump=1000):
        """
        compute and dump settings
        input:
            thermo: thermo interval
            dump: dump interval
        """
        # set elem, ndim for compute
        self.lmp.thermo(f"{thermo}")
        self.lmp.compute("pe_all all pe/atom")
        self.lmp.compute("pe_pair all pe/atom pair")

        # dump settings
        self.lmp.dump(
            f"1 all custom {dump} {self.dumpdir}/out.*.lammpstrj id mol type q x y z ix iy iz c_pe_all c_pe_pair element"
            # f"1 all custom {dump} {self.dumpdir}/out.*.lammpstrj id mol type q xu yu zu ix iy iz c_pe_all c_pe_pair element"
        )

        elements = self.struc.get_element_list()
        string = "1 element %s pad 9 sort id" % " ".join(elements)
        self.lmp.dump_modify(string)

    def minimize(self, nstep=1000):
        self.lmp.reset_timestep(0)
        self.lmp.command("minimize 1e-5 1e-5 {} {}".format(nstep, nstep))
        self.optimized = True
