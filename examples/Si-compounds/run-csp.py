# standard libraries
import os
from htocsp.GA import GA
from pyxtal.db import database
from ost.parameters import ForceFieldParameters

db_name, code = '../../../HT-OCSP/benchmarks/Si.db', 'WAHJEW'
db = database(db_name)
xtal = db.get_pyxtal(code)
smile = ""
for i, m in enumerate(xtal.molecules):
    smile += m.smile
    if i < len(xtal.molecules) - 1:
        smile += '.'
if xtal.has_special_site(): xtal = xtal.to_subgroup()

wdir = code
if not os.path.exists(wdir): os.mkdir(wdir)
print(code, smile)
cwd = os.getcwd()
os.chdir(wdir)

# Load parameters from the user-defined xml
if os.path.exists('parameters_opt.xml'):
    smiles = [m.smile for m in xtal.molecules]
    params = ForceFieldParameters(smiles, style='openff', f_coef=0.1); print(params)
    parameters = params.load_parameters('parameters_opt.xml')
else:
    parameters = None
atoms = xtal.get_forcefield(code='charmm', chargemethod='am1bcc', parameters=parameters)

# Prepare the charmm files
chm_info = atoms.get_atom_info()
os.system('cp charmm.rtf pyxtal.rtf')
os.system('cp charmm.prm pyxtal.prm')
os.chdir(cwd)

# Run GA-CSP 
ga = GA(smile,
        chm_info,
        wdir,
        [xtal.group.number],
        code.lower(),
        N_gen = 1,#00,
        N_pop = 100, #0,
        cif = 'pyxtal-' + code + '.cif',
        ncpu = 1, #20,#2,#0,
        factor = 1.5,
        skip_ani = True,
        )
ga.run()

