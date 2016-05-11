import scipy.stats
import numpy as np
import math


class SUFilter:
    # rows are features, cols are data points
    # class number should start with 0
    def __init__(self, samples, classes):
        self.samples = np.array(samples)
        self.classes = np.array(classes)


    def fit(self, ranked=False):
        """Calculates the symmetrical uncertainty for each feature
        and returns usually the selected features
            
        Parameters
        ----------
        TODO threshold, should it be in percentage or number of features?
        percantage seems more convenient, or should it be somewhere else
        ranked: boolean
        Determs if the features are returned ranked or with weight

        Returns
        -------
        out: 1D array
        The features with their rank or weight depending on the ranked value
        """
        
        su_values = np.zeros(self.samples.shape[0])
        for feature_index in range(self.samples.shape[0]):
            su_values[feature_index] = self.symmetrical_uncertainty(feature_index)
        if (ranked == True):
            # multiply with -1 go receive descending order
            return np.argsort(-su_values)
        else:
            return su_values

    def symmetrical_uncertainty(self, feature_index):
        h_f = self.feature_entropy(feature_index)
        h_c = self.class_entropy()
        h_fc = self.feature_class_joint_entropy(feature_index)
        # entropy f given c
        h_f_given_c = h_fc - h_c
        return 2 * (h_f - h_f_given_c) / (h_f + h_c)

    def feature_entropy(self, i):
        return scipy.stats.entropy(self.feature_distribution(i))

    def feature_distribution(self, i):
        _, counts = np.unique(self.samples[i, :], return_counts=True)
        return counts / float(counts.sum())

    def class_entropy(self):
        return scipy.stats.entropy(self.class_distribution())

    def class_distribution(self):
        _, counts = np.unique(self.classes, return_counts=True)
        return counts / float(counts.sum())

    def feature_class_joint_entropy(self, i):
        feature_class_distribution = self.feature_class_joint_distribution(i)
        # ignores logs by zero because we will replace the nan values
        # in the next step with zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            logPfc = np.log(feature_class_distribution)
        logPfc[np.isinf(logPfc)] = 0
        entropy = -np.sum(feature_class_distribution * logPfc)
        return entropy

    # rows are features, cols are classes
    def feature_class_joint_distribution(self, i):
        unique_features = np.unique(self.samples[i, :])
        distribution = np.zeros((unique_features.size, max(self.classes) + 1))

        for sample_index, feature_value in enumerate(self.samples[i, :]):
            feature_index = np.searchsorted(unique_features, feature_value)
            class_index = self.classes[sample_index]
            distribution[feature_index, class_index] += 1

        return distribution / distribution.sum()
