from benchmarks import Benchmark
from feature_selector import FeatureSelector
import numpy as np
import scipy.stats
from abc import ABCMeta, abstractmethod
from data_sets import DataSets
from feature_selection import FeatureSelection


class EnsembleMethod(metaclass=ABCMeta):
    def __init__(self, feature_selectors, features_type="weight"):
        self.__name__ = type(self).__name__
        self.features_type = features_type

        if not isinstance(feature_selectors, list):
            feature_selectors = [feature_selectors]

        self.feature_selectors = [FeatureSelection(f) for f in feature_selectors]

    def rank(self, data_set, benchmark: Benchmark):
        bench_features_selection = []

        _, labels = DataSets.load(data_set)
        cv = benchmark.cv(labels.shape[0])

        for f in self.feature_selectors:
            bench_features_selection.append(f.load(data_set, cv, self.features_type))
        bench_features_selection = np.array(bench_features_selection)

        ranks = []
        for i in range(bench_features_selection.shape[1]):
            ranks.append(FeatureSelector.rank_weights(
                self.combine(bench_features_selection[:, i])
            ))

        return np.array(ranks)

    @abstractmethod
    def combine(self, feature_ranks):
        pass


class Mean(EnsembleMethod):
    def __init__(self, feature_selectors, power=1, **kwargs):
        super().__init__(feature_selectors, **kwargs)
        self.__name__ = "Mean - {}".format(power)
        self.power = power

    def combine(self, features_selection):
        return np.power(features_selection, self.power).mean(axis=0)


class SMean(EnsembleMethod):
    def __init__(self, feature_selectors, min_mean_max=[1, 1, 1]):
        super().__init__(feature_selectors)
        self.weights = np.array(min_mean_max)
        self.__name__ = "SMean - {} {} {}".format(*min_mean_max)

    def combine(self, features_selection):
        f_mean = np.mean(features_selection, axis=0)
        f_max = np.max(features_selection, axis=0)
        f_min = np.min(features_selection, axis=0)
        return (np.vstack((f_min, f_mean, f_max)) * self.weights[:, np.newaxis]).mean(axis=0)


class Stacking(FeatureSelector):
    def __init__(self, feature_selectors, combination="mean", p=1):
        super().__init__()
        self.feature_selectors = feature_selectors
        self.combination = combination
        self.p = p
        self.__name__ = "Stacking - {} {}".format(self.combination, self.p)

    def rank(self, data, classes):
        features_rankings = []
        for feature_selector in self.feature_selectors:
            features_rankings.append(feature_selector.rank(data, classes))
        features_rankings = np.array(features_rankings)
        return self.rank_weights(self.combine(features_rankings))

    def weight(self, data, classes):
        features_weights = []
        for feature_selector in self.feature_selectors:
            features_weights.append(feature_selector.weight(data, classes))
        features_weights = np.array(features_weights)
        return self.combine(features_weights)

    def combine(self, features_rankings):
        if self.combination == "mean":
            return np.power(features_rankings, self.p).mean(axis=0)
        if self.combination == "hmean":
            regularization_parameter = 1e-15
            return scipy.stats.hmean(np.power(features_rankings + regularization_parameter, self.p), axis=0)
        raise Exception("Unknown combination")
