import contextlib
import os
import shutil
import subprocess
import numpy as np
from pyocse.parameters import ForceFieldParameters
import lammps_logfile

def lammps_read(fname, sym_pos=-2):

    log = lammps_logfile.File(fname)
    step = log.get("Step")
    if not "[Sym" in step[sym_pos]:
        for i in range(1, 10):
            if "[Sym" in step[sym_pos-i]:
                sym_pos = sym_pos - i
                break
    last_thermo_pos = sym_pos - 1

    step = log.get("Step")
    spcpu = log.get("S/CPU")
    pe = log.get("PotEng")
    a = log.get("Cella")
    b = log.get("Cellb")
    c = log.get("Cellc")
    alp = log.get("CellAlpha")
    bet = log.get("CellBeta")
    gam = log.get("CellGamma")

    ret = [
        float(step[last_thermo_pos]) / float(spcpu[last_thermo_pos]),
        float(pe[last_thermo_pos]),
        (" ").join([
            str(a[last_thermo_pos]),
            str(b[last_thermo_pos]),
            str(c[last_thermo_pos]),
            str(alp[last_thermo_pos]),
            str(bet[last_thermo_pos]),
            str(gam[last_thermo_pos]),
        ]),
        int(spcpu[sym_pos])
    ]
    return ret

class LMP:
    """
    A calculator to perform oragnic crystal structure optimization in CHARMM.

    Args:
        - struc: pyxtal.molecular_crystal.molecular_crystal object
        - label (str): label for this calculation
        - prefix (str): prefix of this calculation
        - exe (str): charmm executable
    """

    def __init__(
        self,
        struc,
        atom_info=None,
        label="_",
        prefix="pyxtal",
        exe="lmp",
        timeout=300,
        debug=True,
    ):
        self.errorE = 1e+5
        self.error = False
        self.params = atom_info
        # check charmm Executable
        #if shutil.which(exe) is None:
        #    raise BaseException(f"{exe} is not installed")
        #else:
        self.exe = exe
        self.timeout = timeout
        self.debug = debug

        # Files IO
        self.prefix = prefix
        self.label = label
        self.inp = self.prefix + ".in"
        self.dat = self.prefix + ".dat"
        self.log = self.label + ".log"
        self.dump = "dump.lammpstrj"
        self.folder = "LMP_" + self.label

        # Structure Manipulation
        #struc.resort()
        self.structure = struc
        self.spacegroup_number = struc.group.number

    def write(self):
        xtal = self.structure
        lmp_struc, _ = self.params.get_lmp_input_from_structure(xtal.to_ase(resort = False), xtal.numMols, set_template=False)
        lmp_struc.write_lammps(fin="tmp.in", fdat=self.dat)

        additional_lmpcmds_box = """
variable vmax equal 0.005
variable ptarget equal 500
min_style cg

fix br all symmetry 1e-4
minimize 0 1e-4 200 200 #1

unfix br 
fix  br all box/relax/symmetry symprec 1e-4 x 1 y ${ptarget} z 1 xz 1 vmax ${vmax} fixedpoint 0 0 0 nreset 50
minimize 0 1e-4 500 500 #3

unfix br 
fix  br all box/relax/symmetry symprec 1e-4 x ${ptarget} y 1 z ${ptarget} xz ${ptarget} vmax ${vmax} fixedpoint 0 0 0 nreset 50
minimize 0 1e-4 500 500 #3

unfix br
fix br all box/relax/symmetry symprec  1e-4 x 1 y 1 z 1 xz ${ptarget} vmax ${vmax} fixedpoint 0 0 0 nreset 50
minimize 0 1e-6 500 500 #5

unfix br
fix br all box/relax/symmetry symprec 1e-4 symcell false symposs false x 1 y 1 z 1 xz 1 vmax ${vmax} fixedpoint 0 0 0 nreset 50
minimize 0 1e-6 500 500 #5

unfix br
fix br all symmetry 1e-4 false false
minimize 0 1e-6 200 200 #2
        """
        lmpintxt = open("tmp.in").read()
        lmpintxt = lmpintxt.replace("lmp.dat", self.dat)
        lmpintxt = lmpintxt.replace("custom step ", "custom step spcpu ")
        lmpintxt = lmpintxt.replace("#compute ", "compute ")
        lmpintxt = lmpintxt.replace("#dump 1 all custom 1 ", "dump 1 all custom 100 ")
        lmpintxt = lmpintxt.replace("#dump_modify ", "dump_modify ")
        lmpintxt += additional_lmpcmds_box
        open(self.inp, 'w').write(lmpintxt)

    def read(self):
        from ase.io import read
        step, eng, cell, sg = lammps_read(self.log)
        if sg != self.spacegroup_number:
            self.structure.energy = self.errorE
            if self.debug:
                print("Space group was changed during relaxation. ", self.spacegroup_number, " -> ", sg, "@", self.label)
        else:
            self.structure.energy = float(eng)
        ase_struc = read(self.dump, format='lammps-dump-text', index=-1)
        positions = ase_struc.get_positions()

        count = 0
        try:
            for _i, site in enumerate(self.structure.mol_sites):
                coords = positions[count: count + len(site.molecule.mol)]
                site.update(coords, self.structure.lattice, absolute=True)
                count += len(site.molecule.mol)
            # print("after relaxation  : ", self.structure.lattice, "iter: ", self.structure.iter)
            self.structure.optimize_lattice()
            self.structure.update_wyckoffs()
        except Exception:
            self.structure.energy = self.errorE
            self.error
            if self.debug:
                print("Cannot retrieve Structure after optimization")
                print("lattice", self.structure.lattice)
                self.structure.to_file("1.cif")
                print("Check 1.cif in ", os.getcwd())
                pairs = self.structure.check_short_distances()
                if len(pairs) > 0:
                    print(self.structure.to_file())
                    print("short distance pair", pairs)


    def run(self, clean=True):
        """
        Only run calc if it makes sense
        """
        if not self.error:
            os.makedirs(self.folder, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(self.folder)

            self.write()  # ; print("write", time()-t0)
            res = self.execute()  # ; print("exe", time()-t0)
            if res is not None:
                self.read()  # ; print("read", self.structure.energy)
            else:
                self.structure.energy = self.errorE
                self.error = True
            if clean:
                self.clean()

            os.chdir(cwd)

    def execute(self):
        cmd = f'{self.exe} -in {self.inp} -log {self.log} -nocite > /dev/null'
        # os.system(cmd)
        with open(os.devnull, 'w') as devnull:
            try:
                # Run the external command with a timeout
                result = subprocess.run(
                    cmd, shell=True, timeout=self.timeout, check=True, stderr=devnull)
                return result.returncode  # Or handle the result as needed
            except subprocess.CalledProcessError as e:
                print(f"Command '{cmd}' failed with return code {e.returncode}.")
                return None
            except subprocess.TimeoutExpired:
                print(f"External command {cmd} timed out.")
                return None

    def clean(self):
        os.remove(self.inp) if os.path.exists(self.inp) else None
        os.remove(self.dat) if os.path.exists(self.dat) else None
        os.remove(self.log) if os.path.exists(self.log) else None
        os.remove(self.dump) if os.path.exists(self.dump) else None
