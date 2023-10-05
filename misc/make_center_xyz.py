import os, sys

if len(sys.argv)<2:
    raise RuntimeError('Needs to provide path')

path = sys.argv[1]
os.chdir(path)

with open('center.dat', 'r') as f:
    lines = f.readlines()
with open('center.xyz', 'w') as f:
    for i, l in enumerate(lines[3:]):
        tmp = l.split()
        if len(tmp) == 2:
            print(l[:-1])
            f.write(tmp[1] + '\n')
            f.write(tmp[0] + '\n')
        else:
            f.write('C {:s} {:s} {:s}\n'.format(*tmp[1:]))
    print("Complete", path+'/center.xyz')
