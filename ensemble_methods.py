from benchmarks import Benchmark
from feature_selector import FeatureSelector
import numpy as np
import scipy.stats
from abc import ABCMeta, abstractmethod
from data_sets import DataSets, PreComputedData
from feature_selection import FeatureSelection
from sklearn.cross_validation import KFold


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

        data, labels = DataSets.load(data_set)
        cv_indices = PreComputedData.load_cv(data_set, cv)

        feature_selection = []
        for i in range(bench_features_selection.shape[1]):
            feature_selection.append(FeatureSelector.rank_weights(self.combine(
                bench_features_selection[:, i],
                data[:, cv_indices[i][0]],
                labels[cv_indices[i][0]]
             )))

        return np.array(feature_selection)

    @abstractmethod
    def combine(self, feature_selection, data, labels):
        pass


class Mean(EnsembleMethod):
    def __init__(self, feature_selectors, power=1, **kwargs):
        super().__init__(feature_selectors, **kwargs)
        self.__name__ = "Mean - {}".format(power)
        self.power = power

    def combine(self, features_selection, data, labels):
        return np.power(features_selection, self.power).mean(axis=0)


class SMean(EnsembleMethod):
    def __init__(self, feature_selectors, min_mean_max=[1, 1, 1], **kwargs):
        super().__init__(feature_selectors, **kwargs)
        self.weights = np.array(min_mean_max)
        self.__name__ = "SMean - {} {} {}".format(*min_mean_max)

    def combine(self, features_selection, data, labels):
        f_mean = np.mean(features_selection, axis=0)
        f_max = np.max(features_selection, axis=0)
        f_min = np.min(features_selection, axis=0)
        return (np.vstack((f_min, f_mean, f_max)) * self.weights[:, np.newaxis]).mean(axis=0)


class SMeanWithClassifier(EnsembleMethod):
    def __init__(self, feature_selectors, classifiers, min_mean_max=[1, 1, 1], **kwargs):
        super().__init__(feature_selectors, **kwargs)
        self.weights = np.array(min_mean_max)
        self.__name__ = "SMeanWithClassifier - {} {} {}".format(*min_mean_max)
        self.classifiers = classifiers

    def combine(self, features_selection, data, labels):
        cv = KFold(labels.shape[0])
        accuracy = np.zeros(len(self.feature_selectors))

        for i in range(len(self.feature_selectors)):
            best_features_indices = np.argsort(features_selection[i])[:-int(features_selection[i].shape[0] / 100):-1]
            for train_index, test_index in cv:
                for c in self.classifiers:
                    c.fit(data[np.ix_(best_features_indices, train_index)].T, labels[train_index])
                    accuracy[i] += c.score(data[np.ix_(best_features_indices, test_index)].T, labels[test_index])

        print(np.exp(accuracy))
        features_selection = (features_selection.T * np.exp(accuracy)).T

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

    def rank(self, data, labels):
        features_rankings = []
        for feature_selector in self.feature_selectors:
            features_rankings.append(feature_selector.rank(data, labels))
        features_rankings = np.array(features_rankings)
        return self.rank_weights(self.combine(features_rankings))

    def weight(self, data, labels):
        features_weights = []
        for feature_selector in self.feature_selectors:
            features_weights.append(feature_selector.weight(data, labels))
        features_weights = np.array(features_weights)
        return self.combine(features_weights)

    def combine(self, features_rankings):
        if self.combination == "mean":
            return np.power(features_rankings, self.p).mean(axis=0)
        if self.combination == "hmean":
            regularization_parameter = 1e-15
            return scipy.stats.hmean(np.power(features_rankings + regularization_parameter, self.p), axis=0)
        raise Exception("Unknown combination")
