from benchmarks import FeatureRanking
import numpy as np
from sklearn.dummy import DummyClassifier


class Dummy(FeatureRanking):
    def rank(self, data, classes):
        return np.arange(data.shape[0])

    @staticmethod
    def classifier():
        return DummyClassifier(strategy='constant', constant=1)
