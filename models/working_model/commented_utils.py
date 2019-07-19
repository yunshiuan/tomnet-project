import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory(name, tensor, save=False):
    i = 1
    for step in tensor[:]:
        if np.count_nonzero(step) > 0:
            z,x,y = step.nonzero()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, -z, c=y, zdir='z', cmap='rainbow')
            ax.view_init(elev=0, azim=-90)
            if save:
                plt.savefig(name + '_' + str(i) + '.png')
            else:
                plt.show()
            i += 1
