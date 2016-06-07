from benchmarks import FeatureRanking
import numpy as np
import scipy.stats


class Stacking(FeatureRanking):
    def __init__(self, feature_ranking_methods, combination="mean"):
        super().__init__()
        self.feature_ranking_methods = feature_ranking_methods
        self.combination = combination
        self.__name__ = "Stacking - {}".format(self.combination)

    def rank(self, data, classes):
        features_rankings = []
        for feature_ranking_method in self.feature_ranking_methods:
            features_rankings.append(feature_ranking_method.rank(data, classes))
        features_rankings = np.array(features_rankings)
        return self.rank_weights(self.combine(features_rankings))

    def combine(self, features_rankings):
        if self.combination == "mean":
            return features_rankings.mean(axis=0)
        if self.combination == "hmean":
            return scipy.stats.hmean(features_rankings, axis=0)
        else:
            raise Exception("Unknown combination")

