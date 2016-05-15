import numpy as np
import filters # filters created by our own
import robustness as rob
import matplotlib.pyplot as plt

### colon data set analysis

samples_colon = np.loadtxt('colon.data')
labels_colon_raw = np.loadtxt('colon.labels')
labels_colon = np.sign(labels_colon_raw)
# transform -1 labels to 0
labels_colon[labels_colon_raw == -1] = 0


reload(rob)
reload(filters)
robustness = rob.RobustnessMeasurement(samples_colon, labels_colon)
#robustness.calculate_robustness()

### MNIST data set analysis
from sklearn.datasets import load_digits

dataset = load_digits()
su = filters.SUFilter(dataset.data.T, dataset.target)
weights = su.apply_su_on_data(method='default')
ranking = robustness.rank_features(weights[np.newaxis,:])

#fig = plt.figure()
#im = plt.imshow(weights.reshape((8,8)),interpolation="nearest")
#fig.colorbar(im);
#plt.show()
#fig.savefig("MNIST_SU.png")

mean_number = np.zeros((10,64))
for i in range(9):
    data_single_number = dataset.data[dataset.target == i+1,:]
    mean_number[i] = np.mean(data_single_number, axis=0)

fig = plt.figure(0)
ax1 = plt.subplot2grid((3,6), (0,0), colspan=3, rowspan=3)
im = ax1.imshow(weights.reshape((8,8)),interpolation="nearest")
fig.colorbar(im);
for i in range(9):
    ax = plt.subplot2grid((3,6), ((i/3),3+(i%3)))
    ax.imshow(mean_number[i].reshape((8,8)),interpolation="nearest", cmap='gray')
    
plt.show()
