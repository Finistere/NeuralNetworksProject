from benchmarks import FeatureSelector, FeatureSelectionGenerator, Benchmark
import numpy as np
import scipy.stats
from abc import ABCMeta, abstractmethod
from data_sets import DataSets
import os
import errno


class FeatureSelection(FeatureSelectionGenerator):
    def load(self, data_set, cv, method):
        try:
            return np.load(self.__file_name(data_set, cv, method) + ".npy")
        except FileNotFoundError:
            return self.__gen(data_set, cv, method)

    def __file_name(self, data_set, cv, method):
        return self.__dir_name(data_set, cv, method) + "/" + self.__name__

    @staticmethod
    def __dir_name(data_set, cv, method):
        return "{root_dir}/feature_{method}s/{data_set}/{cv}".format(
            root_dir=DataSets.root_dir,
            method=method,
            data_set=data_set,
            cv=type(cv).__name__
        )

    def __gen(self, data_set, cv, method):
        data, labels = DataSets.load(data_set)

        print("Generating feature {method}s of {data_set} ({cv}) with {feature_selector}".format(
            method=method,
            data_set=data_set,
            feature_selector=self.__name__,
            cv=type(cv).__name__
        ))
        ranks = self.generate(data, labels, cv)

        try:
            os.makedirs(self.__dir_name(data_set, cv, method))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        np.save(self.__file_name(data_set, cv, method), ranks)

        return ranks


class EnsembleMethod(metaclass=ABCMeta):
    def __init__(self, feature_selectors):
        self.__name__ = type(self).__name__

        if not isinstance(feature_selectors, list):
            feature_selectors = [feature_selectors]

        self.feature_selectors = [FeatureSelection(f) for f in feature_selectors]

    def ranks(self, data_set, benchmark: Benchmark):
        bench_features_selection = []

        _, labels = DataSets.load(data_set)
        cv = benchmark.cv(labels.shape[0])

        for f in self.feature_selectors:
            bench_features_selection.append(f.load(data_set, cv, benchmark.feature_selection_method))
        bench_features_selection = np.array(bench_features_selection)

        ranks = []
        for i in range(bench_features_selection.shape[1]):
            ranks.append(scipy.stats.rankdata(
                self.combine(
                    self.normalize_weights(bench_features_selection[:, i])
                ),
                method='ordinal'
            ))

        return np.array(ranks)

    @staticmethod
    def normalize_weights(weights):
        centred_weights = (weights - weights.mean(axis=1)[:, np.newaxis])
        normalized_weights = centred_weights / (weights.max(axis=1) - weights.min(axis=1))[:, np.newaxis]
        return normalized_weights + 0.500000001

    @abstractmethod
    def combine(self, feature_ranks):
        pass


class Mean(EnsembleMethod):
    def __init__(self, feature_selectors, power=1):
        super().__init__(feature_selectors)
        self.__name__ = "Mean - {}".format(power)
        self.power = power

    def combine(self, features_selection):
        return np.power(features_selection, self.power).mean(axis=0)


class HMean(EnsembleMethod):
    def __init__(self, feature_selectors, power=1):
        super().__init__(feature_selectors)
        self.__name__ = "HMean - {}".format(power)
        self.power = power

    def combine(self, features_selection):
        return scipy.stats.hmean(np.power(features_selection, self.power), axis=0)


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
