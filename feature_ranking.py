from benchmarks import FeatureRanking
import numpy as np
from sklearn.dummy import DummyClassifier


class Dummy(FeatureRanking):
    def __init__(self, constant=1):
        self.constant = constant

    def rank(self, data, classes):
        return np.arange(data.shape[0])

    def classifier(self):
        return DummyClassifier(strategy='constant', constant=self.constant)
