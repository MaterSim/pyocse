import os
from pyxtal.db import database
from htocsp.interfaces.charmm import CHARMM
from ost.parameters import ForceFieldParameters
import pymatgen.analysis.structure_matcher as sm

name, code = 'Si.db', 'WAHJEW'
db = database('../../../HT-OCSP/benchmarks/' + name)
xtal = db.get_pyxtal(code)
if xtal.has_special_site(): xtal = xtal.to_subgroup()
pmg0 = xtal.to_pymatgen(); pmg0.remove_species('H')

smiles = [m.smile for m in xtal.molecules]
params = ForceFieldParameters(smiles, style='openff')

print("\nLattice before relaxation", xtal.lattice)

for filename in ['parameters_init.xml', 'parameters_opt.xml', '../../../HT-OCSP/parameters.xml']:
    parameters, _ = params.load_parameters(filename)
    params.ff.update_parameters(parameters)
    ase_with_ff = params.get_ase_charmm(parameters)
    ase_with_ff.write_charmmfiles(base='pyxtal')
    atom_info = ase_with_ff.get_atom_info()
    calc = CHARMM(xtal, prm='pyxtal.prm', rtf='pyxtal.rtf', atom_info=atom_info)
    calc.run()#clean=False)
    calc.structure.optimize_lattice()
    print("Lattice after relaxation", calc.structure.lattice)
    print("Energy", calc.structure.energy)
    #calc.structure.to_file(label+'opt.cif')
    pmg1 = calc.structure.to_pymatgen(); pmg1.remove_species('H')
    print(sm.StructureMatcher().get_rms_dist(pmg0, pmg1))
