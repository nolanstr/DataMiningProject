import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from MarkovChainMonteCarlo import MCMC_MH


f = h5.File('032021_MOM_elastic.hdf5', 'r')

data = np.vstack((np.array(f['Aluminum']['strain']),
                        np.array(f['Aluminum']['stress'])))

model = MCMC_MH(data, init_mu=11.e6)
model()

fig, axs = plt.subplots(2, 2)
fig.suptitle('Aluminum')
idx = 2000

axs[0,0].set_title('Mean')
axs[0,1].set_title('Standard Deviation')
axs[0,0].set_ylabel('ksi')
axs[0,0].set_xlabel('iterations')
axs[0,1].set_xlabel('iterations')

axs[1,0].set_ylabel('frequency')
axs[1,0].set_xlabel('ksi')
axs[1,1].set_xlabel('ksi')

axs[0,0].plot(model.rejected_iterations[:], 
            np.array(model.rejected)[:,0], 'r*', label='Rejected Values')
axs[0,0].plot(model.accepted_iterations[:], 
            np.array(model.accepted)[:,0], 'b*', label='Accepted Values')
axs[0,0].set_xlim(model.accepted_iterations[idx], model.accepted_iterations[-1])
axs[1,0].hist(np.array(model.accepted)[idx:,0])

axs[0,1].plot(model.rejected_iterations[:], 
            np.array(model.rejected)[:,1], 'r*', label='Rejected Values')
axs[0,1].plot(model.accepted_iterations[:], 
            np.array(model.accepted)[:,1], 'b*', label='Accepted Values')
axs[1,1].hist(np.array(model.accepted)[idx:,1])

axs[0,1].set_xlim(model.accepted_iterations[idx], model.accepted_iterations[-1])
plt.show()

import pdb;pdb.set_trace()
