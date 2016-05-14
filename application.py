import numpy as np
import filters # filters created by our own
import robustness as rob
import mutual_information as mi # module with given su

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
print(ranking.reshape((8,8)))
