import matplotlib.pyplot as plt
import numpy as np
# Create a single figure with subplots for each column
def make_plot(filepath, label=None):      
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

       
    path = filepath[:-11]
    direction = path.split('/')[-1]
    name = 'Cycle-' + direction
    code = path.split('/')[0]

    print(path, direction)

    fig.suptitle(name + '-' + code, fontsize=16)
    if direction == 'xx':
        col = 2
    elif direction  == 'yy':
        col = 3
    elif direction  == 'zz':
        col = 4
    elif direction == 'xz':
        col = 5
    elif direction == 'yz':
        col = 6
    elif direction == 'xy':
        col = 7
 
    # Load the data from the text file into a DataFrame
    os.chdir(path)
    file_path = 'output.dat'
    data = np.loadtxt(file_path)
    middle = int(len(data)/2)

    d = data[0, 1]
    for i in range(middle, len(data)): data[i, 1] -= 2*d*(i-middle+2)
        
    labels = ['S_' + direction + ' (MPa)', 'Potential Energy (Kcal/mol)']
    for i, column in enumerate([col, -1]):#, 5, 6, 8]):
        row, col = i, 0
        if row == 0:
            colors = ['r', 'b']
        else:
            colors = ['r', 'g']
        axs[row].plot(data[:middle, 1], data[:middle, column], color=colors[0], label = 'Forward')
        axs[row].plot(data[middle:, 1], data[middle:, column], color=colors[1], label = 'Backward')
        axs[row].grid(True)
        axs[row].legend(loc=1)
        axs[row].set_ylabel(labels[i])
        if i == 1:
            axs[row].set_xlabel('Strain')
        else:
            axs[row].set_xticks([])
    # Adjust the layout and spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.01)
    # Save the figure as '1.png'
    plt.savefig(name+'.png')

from glob import glob
import os
cwd = os.getcwd()
files = glob('MT-*-openff/*/output.dat')
for f in files:
    
    make_plot(f)
    os.chdir(cwd)
