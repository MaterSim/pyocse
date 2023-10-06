import matplotlib.pyplot as plt
import numpy as np
# Create a single figure with subplots for each column
def make_plot(filepath, label=None, num=600):      
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    path = filepath[:-11]
    direction = path.split('/')[-1]
    name = direction
    code = path.split('/')[0][3:-7]

    print(path, direction)

    if direction == 'xx':
        col = 2
    elif direction  == 'yy':
        col = 3
    elif direction  == 'zz':
        col = 4
    elif direction == 'xy':
        col = 7
    elif direction == 'xz':
        col = 5
    elif direction == 'yz':
        col = 6
    name = code + '-' + str(col-1) + '-' + name
    fig.suptitle(name, fontsize=22, y=0.95)
    # Load the data from the text file into a DataFrame
    os.chdir(path)
    file_path = 'output.dat'
    data = np.loadtxt(file_path)
    middle = int(num/2)

    d = data[0, 1]
    for i in range(middle, len(data)): data[i, 1] -= 2*d*(i-middle+2)
        
    labels = ['S_' + direction + ' (MPa)', 'Energy (Kcal/mol)']
    for i, column in enumerate([col, -1]):#, 5, 6, 8]):
        row, col = i, 0
        if row == 0:
            colors = ['r', 'b']
        else:
            colors = ['r', 'g']
        axs[row].plot(data[:middle, 1], data[:middle, column], color=colors[0], label = 'Load')
        if len(data) > middle:
            axs[row].plot(data[middle:, 1], data[middle:, column], color=colors[1], label = 'Unload')
        axs[row].set_xlim([0, 0.30])
        axs[row].grid(True)
        axs[row].legend(loc=1, fontsize=16)
        axs[row].set_ylabel(labels[i], fontsize=15)
        if i == 1:
            axs[row].set_xlabel('Strain', fontsize=15)
        else:
            axs[row].set_xticks([])
        if len(data) < num:
            axs[row].set_facecolor('grey')
    # Adjust the layout and spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.025)
    # Save the figure as '1.png'
    name = 'Cycle-' + name
    plt.savefig(name+'.png')
    plt.close()

from glob import glob
import os
cwd = os.getcwd()
os.system('rm MT-*-openff/*/*.png')
files = glob('MT-*-openff/*/output.dat')
files.sort()
for f in files:
    make_plot(f)
    os.chdir(cwd)
