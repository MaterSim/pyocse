import os
from pyxtal.db import database
from htocsp.interfaces.charmm import CHARMM
from ost.parameters import ForceFieldParameters


def test_interface(xtal, charge, filename, label):
    if xtal.has_special_site(): xtal = xtal.to_subgroup()

    # setup simulation
    cwd = os.getcwd()
    dir_name = 'test'
    if not os.path.exists(dir_name): os.mkdir(dir_name)

    smiles = [m.smile for m in xtal.molecules]
    params = ForceFieldParameters(smiles, style='openff')
    parameters = params.load_parameters(filename)

    os.chdir(dir_name)
    xtal.to_file(label+'init.cif')
    atoms = xtal.get_forcefield(code='charmm', chargemethod=charge, parameters=parameters)
    atom_info = atoms.get_atom_info()
    calc = CHARMM(xtal, prm='charmm.prm', rtf='charmm.rtf', atom_info=atom_info)
    print("\nLattice before relaxation", calc.structure.lattice)
    calc.run()#clean=False)
    calc.structure.optimize_lattice()
    print("Lattice after  relaxation", calc.structure.lattice)
    print("\nEnergy", calc.structure.energy)
    calc.structure.to_file(label+'opt.cif')
    os.system('mv charmm.rtf charmm.rtf-' + charge)
    os.chdir(cwd)

name, code = 'Si.db', 'WAHJEW'
db = database('../../../HT-OCSP/benchmarks/' + name)
for i, f in enumerate(['parameters_init.xml', 'parameters_opt.xml']):
    xtal = db.get_pyxtal(code)
    print(xtal)
    test_interface(xtal, 'am1bcc', f, str(i))

