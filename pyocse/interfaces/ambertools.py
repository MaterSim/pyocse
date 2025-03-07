#!/usr/bin/env python3
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
from pyocse.utils import procrun


# modules
def run_antechamber(
    molname: str,
    path: Path or str,
    net_charge: Optional[int] = 0,
    spin_multiplicity: Optional[int] = 1,
    resname: Optional[str] = "UNL",
    atomtyping: Optional[str] = "gaff",
    base: Optional[str] = "ff",
    chargemethod: Optional[str] = "bcc",  # or 'gas'
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Run antechamber to generate AMBER prmtop and inpcrd files.
    for more info on antechamber see:
        input: http://ambermd.org/antechamber/ac.html#antechamber
    """

    for exec in ["antechamber", "parmchk2"]:
        if shutil.which(exec) is None:
            raise Exception(f"{exec} is not installed")

    path = Path(path)
    mol_init = path.absolute()
    suffix = path.suffix[1:]

    amber_prepi = base + ".prepi"
    amber_frcmod = base + ".frcmod"
    amber_inpcrd = base + ".inpcrd"
    amber_prmtop = base + ".prmtop"
    labeledmol2 = f"{base}_label.mol2"

    if net_charge != 0 or suffix == "pdb":
        print("avoid gesteiger due to non-nutral molecule or pdb input: ", molname)
        if chargemethod is not None:
            chg = "-c bcc -nc " + str(net_charge)
        else:
            chg = "-nc " + str(net_charge)
    elif chargemethod is not None:
        chg = f"-c {chargemethod} -nc 0"
    else:
        chg = ''
    # prepare commands for ambertools
    acopts = f"{chg} -m {spin_multiplicity} -rn {resname} -s 2 -seq no -pf y -dr no "
    if atomtyping == "gaff":
        acopts += f" -at {atomtyping}"
        cmd = f"antechamber -i {mol_init} -fi {suffix} -o {labeledmol2} -fo mol2 {acopts} > /dev/null"
        #print("========", cmd)
        procrun(cmd)
    elif atomtyping == "skip" and suffix == "mol2":
        labeledmol2 = mol_init

    cmd = f"antechamber -i {labeledmol2} -fi mol2 -o {amber_prepi} -fo prepi {acopts} > /dev/null"
    procrun(cmd)
    if atomtyping is not None:
        cmd = f"parmchk2 -f mol2  -i {labeledmol2} -o {amber_frcmod} -s {atomtyping} -a Y > /dev/null"
        procrun(cmd)

    leapAmberFile = "leaprc.protein.ff14SB"
    leapGaffFile = "leaprc.gaff"
    open("tleap.in", "w").write(
        f"""
verbosity 1
source {leapAmberFile}
source {leapGaffFile}
mods = loadamberparams {amber_frcmod}
{resname} = loadmol2 {labeledmol2}
check {resname}
saveamberparm {resname} {amber_prmtop} {amber_inpcrd}
saveoff {resname} {base}.lib
quit
"""
    )
    cmd = "tleap -f tleap.in > /dev/null"
    procrun(cmd)
    shutil.move("leap.log", f"{base}.tleaplog")
    shutil.move("tleap.in", f"{base}.tleapin")

    amber_files = {}
    amber_files["prepi"] = amber_prepi
    amber_files["frcmod"] = amber_frcmod
    amber_files["inpcrd"] = amber_inpcrd
    amber_files["prmtop"] = amber_prmtop
    amber_files["mol2_labeled"] = labeledmol2
    return amber_files