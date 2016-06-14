from benchmarks import FeatureRanking, FeatureRanksGenerator, Benchmark
import numpy as np
import scipy.stats
from abc import ABCMeta, abstractmethod
from data_sets import DataSets
import os
import errno


class FeaturesRanks(FeatureRanksGenerator):
    def load(self, data_set, cv):
        try:
            return np.load(self.__file_name(data_set, cv) + ".npy")
        except FileNotFoundError:
            return self.__gen(data_set, cv)

    def __file_name(self, data_set, cv):
        return self.__dir_name(data_set, cv) + "/" + self.__name__

    @staticmethod
    def __dir_name(data_set, cv):
        return DataSets.root_dir + "/feature_ranks/" + data_set + "/" + type(cv).__name__

    def __gen(self, data_set, cv):
        data, labels = DataSets.load(data_set)

        print("Generating Features Ranks of {} with {}".format(data_set, self.__name__))
        ranks = self.generate(data, labels, cv)

        try:
            os.makedirs(self.__dir_name(data_set, cv))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        np.save(self.__file_name(data_set, cv), ranks)

        return ranks


class EnsembleMethod(metaclass=ABCMeta):
    def __init__(self, feature_rankings):
        self.__name__ = type(self).__name__

        if not isinstance(feature_rankings, list):
            feature_rankings = [feature_rankings]

        self.feature_ranks = [FeaturesRanks(f) for f in feature_rankings]

    def ranks(self, data_set, benchmark: Benchmark):
        bench_features_ranks = []

        _, labels = DataSets.load(data_set)
        cv = benchmark.cv(labels.shape[0])

        for f in self.feature_ranks:
            bench_features_ranks.append(f.load(data_set, cv))
        bench_features_ranks = np.array(bench_features_ranks)

        ranks = []
        for i in range(bench_features_ranks.shape[1]):
            ranks.append(scipy.stats.rankdata(self.combine(bench_features_ranks[:, i]), method='ordinal'))

        return np.array(ranks)

    @abstractmethod
    def combine(self, feature_ranks):
        pass


class Mean(EnsembleMethod):
    def __init__(self, feature_rankings, power=1):
        super().__init__(feature_rankings)
        self.__name__ = "Mean - {}".format(power)
        self.power = power

    def combine(self, feature_ranks):
        return np.power(feature_ranks, self.power).mean(axis=0)


class Stacking(FeatureRanking):
    def __init__(self, feature_ranking_methods, combination="mean", p=1):
        super().__init__()
        self.feature_ranking_methods = feature_ranking_methods
        self.combination = combination
        self.p = p
        self.__name__ = "Stacking - {} {}".format(self.combination, self.p)

    def rank(self, data, classes):
        features_rankings = []
        for feature_ranking_method in self.feature_ranking_methods:
            features_rankings.append(feature_ranking_method.rank(data, classes))
        features_rankings = np.array(features_rankings)
        return self.rank_weights(self.combine(features_rankings))

    def combine(self, features_rankings):
        if self.combination == "mean":
            return np.power(features_rankings, self.p).mean(axis=0)
        if self.combination == "hmean":
            return scipy.stats.hmean(np.power(features_rankings, self.p), axis=0)
        raise Exception("Unknown combination")

