# Find your structure from https://www.ccdc.cam.ac.uk/structures/Search?
tag = {
       "csd_code": 'IMAZOL06',
       "ccdc_number": 1180146,
       "smiles": "N1C=CN=C1", #"CC(=O)OC1=CC=CC=C1C(O)=O",
}

from pyxtal import pyxtal
xtal = pyxtal(molecular=True)
xtal.from_seed('dataset/'+str(tag['csd_code'])+'.cif',
               molecules=[str(tag['smiles'])+'.smi'])
xtal.tag = tag
print(xtal)

from pyxtal.db import make_entry_from_pyxtal, database
entry = make_entry_from_pyxtal(xtal)
db = database('dataset/mech.db')
db.add(entry)
