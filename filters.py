import scipy.stats
import numpy as np
import math


class DataSet:
    # rows are features, cols are data points
    # class number should start with 0
    def __init__(self, samples, classes):
        self.samples = np.array(samples)
        self.classes = np.array(classes)

    def symmetrical_uncertainty(self, feature_index):
        h_f = self.feature_entropy(feature_index)
        h_c = self.class_entropy()
        h_fc = self.feature_class_conditional_entropy(feature_index)

        return 2 * (h_f - h_fc) / (h_f + h_c)

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

    def feature_class_conditional_entropy(self, i):
        feature_class_distribution = self.feature_class_joint_distribution(i)
        class_distribution = self.class_distribution()
        
        # comuptes ration of P(X)/P(X,Y)
        PxPxy = np.log(class_distribution / feature_class_distribution)
        PxPxy[np.isinf(PxPxy)] = 0
        entropy = (feature_class_distribution * PxPxy).sum()

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


