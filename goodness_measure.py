from robustness_measure import Measure
from abc import ABCMeta, abstractmethod
import numpy as np


class RankData:
    def __init__(self, features_rank, n_significant_features):
        self.features_rank = features_rank
        self.sorted_indices = np.argsort(features_rank)[::-1]
        self.n_significant = n_significant_features

    def __len__(self):
        return len(self.features_rank)

    @property
    def true_positive(self):
        return (self.sorted_indices[:self.n_significant] < self.n_significant).sum()

    @property
    def false_positive(self):
        return (self.sorted_indices[:self.n_significant] >= self.n_significant).sum()

    @property
    def true_negative(self):
        return (self.sorted_indices[self.n_significant:] >= self.n_significant).sum()

    @property
    def false_negative(self):
        return (self.sorted_indices[self.n_significant:] < self.n_significant).sum()


class GoodnessMeasure(Measure, metaclass=ABCMeta):
    def __init__(self, n_significant_features):
        super().__init__()
        self.n_significant_features = n_significant_features

    def measure(self, features_ranks):
        goodness = []
        for i in range(features_ranks.shape[1]):
            goodness.append(self.goodness(
                RankData(features_ranks[:, i].T, self.n_significant_features)
            ))
        return np.mean(goodness)

    @abstractmethod
    def goodness(self, data: RankData):
        pass


class Dummy(GoodnessMeasure):
    def goodness(self, data: RankData):
        return data.features_rank[0]


class Accuracy(GoodnessMeasure):
    def goodness(self, data: RankData):
        return (data.true_negative + data.true_positive) / len(data)


class Precision(GoodnessMeasure):
    def goodness(self, data: RankData):
        return data.true_positive / data.n_significant


