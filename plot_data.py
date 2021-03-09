import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

data = h5.File('030921_MOM_data.h5','r')
keys = [key for key in data.keys()]

fig, axs = plt.subplots(nrows=2,ncols=2)

for i in range(2):
    for j in range(2):

        idx = (i * 2) + j
        axs[i,j].plot(np.array(data[keys[idx]]['Strain']), 
                        np.array(data[keys[idx]]['Stress']), 'bo')
        axs[i,j].set_title(keys[idx])
        axs[i,j].set_xlabel('Strain')
        axs[i,j].set_ylabel('Stress')

plt.show()

import pdb;pdb.set_trace()
