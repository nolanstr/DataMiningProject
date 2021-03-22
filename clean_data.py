import h5py as h5
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


data = h5.File('030921_MOM_data.h5', 'r')

for key in data.keys():

    test_data = np.hstack((np.array(data[key]['Strain']).reshape([-1,1]), 
                            np.array(data[key]['Stress']).reshape([-1,1])))

    clus = KMeans(n_clusters=2, random_state=0).fit(test_data)
    
    idx = []
    for i, val in enumerate(clus.labels_):
        if val == 1:
            idx.append(i)
    import pdb;pdb.set_trace()
    plt.plot(test_data[idx, 0], test_data[idx,1], 'bo')
    plt.show()
    import pdb;pdb.set_trace()
