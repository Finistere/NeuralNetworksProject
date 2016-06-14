from benchmarks import FeatureRanking
import numpy as np
import scipy.stats


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

    def weight(self, data, classes):
        features_weights = []
        for feature_ranking_method in self.feature_ranking_methods:
            features_weights.append(feature_ranking_method.weight(data, classes))
        features_weights = np.array(features_weights)
        return self.combine(features_weights)

    def combine(self, features_rankings):
        if self.combination == "mean":
            return np.power(features_rankings, self.p).mean(axis=0)
        if self.combination == "hmean":
            regularization_parameter = 1e-15
            return scipy.stats.hmean(np.power(features_rankings + regularization_parameter, self.p), axis=0)
        raise Exception("Unknown combination")

