from benchmarks import FeatureRanking
import numpy as np


class Dummy(FeatureRanking):
    def rank(self, data, classes):
        return np.arange(data.shape[0])
