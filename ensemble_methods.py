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

    def combine(self, features_rankings):
        if self.combination == "mean":
            return np.power(features_rankings, self.p).mean(axis=0)
        if self.combination == "hmean":
            return scipy.stats.hmean(np.power(features_rankings, self.p), axis=0)
        raise Exception("Unknown combination")

