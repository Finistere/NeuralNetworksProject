import numpy as np
from sklearn.cross_validation import KFold, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier


class RobustnessMeasure:

    def measure(self, feature_ranks):
        features_rank = self.rank_features(features_weight)
        # apply spearman similarity measures
        spearman_coeffs = self.spearmean_coefficient(features_rank)
        # takes the lower triangular matrix to sum over
        spearman_total = self.mean_of_lower_triangular_of_symmetric_matrix(spearman_coeffs)
        jaccard_top1_indices = self.jaccard_index(features_rank, perc=0.01)
        jaccard_top1_total = self.mean_of_lower_triangular(jaccard_indices)
        jaccard_top5_indices = self.jaccard_index(features_rank, perc=0.05)
        jaccard_top5_total = self.mean_of_lower_triangular(jaccard_indices)
        return np.array([spearman_total, jaccard_top1_total, jaccard_top5_total])

    def spearmean_coefficient(self, features_rank):
        spearman_coeffs, _ = sp.stats.spearmanr(features_rank, axis=1)
        return spearman_coeffs

    def jaccard_index(self, features_rank, perc=0.1):
        if(np.any(np.min(features_rank, axis=1) != np.ones(features_rank.shape[0],dtype=np.int))):
            raise ValueError('features_rank raking does not always begin with 1')
        # the minimal rank a feature mast have to be chose
        minimal_rank = int((1-perc)*features_rank.shape[1]) + 1
        features_rank_c = np.copy(features_rank)
        # set everything below the minimal rank to zero and everything else to 1
        features_rank_c[minimal_rank > features_rank_c] = 0
        features_rank_c[0 != features_rank_c] = 1

        k = features_rank.shape[0]
        jaccard_indices = np.zeros((k,k))

        for i in range(k):
            for j in range(k):
                # both features have to be 1
                condition = np.logical_and(features_rank_c[i]==1, features_rank_c[j]==1)
                jaccard_indices[i,j] = np.sum(condition)

        # normalize
        jaccard_indices = jaccard_indices/float(features_rank.shape[1]-(minimal_rank-1))
        return jaccard_indices

    def mean_of_lower_triangular(self, matrix):
        upper_triangular = np.tril(matrix,-1)
        nonzero_entries = upper_triangular[upper_triangular != 0]
        return np.mean(nonzero_entries)


class FeatureRanking:
    def rank(self, data, classes):
        return np.arange(data.shape[0])

    @staticmethod
    def classifier():
        return KNeighborsClassifier()


class Benchmark:
    def __init__(self, robustness_measure: RobustnessMeasure, feature_ranking: FeatureRanking):
        self.robustness_measure = robustness_measure
        self.feature_ranking = feature_ranking

    def run(self, data, classes):
        robustness = self.robustness(data, classes)
        accuracy = self.classification_accuracy(data, classes)

        return robustness, accuracy

    def robustness(self, data, classes, n_iter=10, test_size=0.1):
        features_ranks = []

        cv = ShuffleSplit(len(classes), n_iter=n_iter, test_size=test_size)

        for train_index, test_index in cv:
            features_ranks.append(self.feature_ranking.rank(data[:, train_index], classes[train_index]))

        return self.robustness_measure.measure(features_ranks)

    def classification_accuracy(self, data, classes, n_folds=4):
        classification_accuracies = []

        cv = KFold(len(classes), n_folds=n_folds)

        for train_index, test_index in cv:

            features_rank = self.feature_ranking.rank(data[:, train_index], classes[train_index])
            features_index = self.__select_features_index(features_rank)

            classifier = self.feature_ranking.classifier()
            classifier.fit(data[features_index, train_index], classes[train_index])
            classification_accuracies.append(
                classifier.score(data[features_index, test_index], classes[test_index])
            )

        return np.mean(classification_accuracies)

    # 1% best features
    @staticmethod
    def __select_features_index(features_rank):
        size = len(features_rank) // 100
        return np.argsort(features_rank)[:size]


class Experiment:
    def __init__(self, robustness_measures=None, feature_rankings=None):
        results_shape = [1, 1]
        if isinstance(robustness_measures, list):
            results_shape[0] = len(robustness_measures)

        if isinstance(feature_rankings, list):
            results_shape[1] = len(feature_rankings)

        self.robustness_measures = robustness_measures
        self.feature_rankings = feature_rankings
        self.results = np.zeros((results_shape[0], results_shape[1], 2))

    def run(self, data, classes):
        for i in range(self.results.shape[0]):
            for j in range(self.results.shape[1]):
                benchmark = Benchmark(
                    robustness_measure=self.__get("robustness_measures", i),
                    feature_ranking=self.__get("feature_rankings", j)
                )
                self.results[i, j, 0], self.results[i, j, 1] = benchmark.run(data, classes)

    def __get(self, attr, i):
        obj = getattr(self, attr)
        if isinstance(obj, list):
            return obj[i]
        else:
            return obj

    def print_results(self):
        print("ROBUSTNESS \n")
        self.__print_table(0)

        print("ACCURACY \n")
        self.__print_table(1)

    def __print_table(self, result_index):
        header = "{:22} " + " | ".join(["{:10}"] * self.results.shape[1])
        row = "{:20} : " + " | ".join(["{:10.2%}"] * self.results.shape[1])

        print(header.format(
            "",
            *[type(self.__get("feature_rankings", i)).__name__ for i in range(self.results.shape[1])]
        ))
        for i in range(self.results.shape[0]):
            print(row.format(
                type(self.__get("robustness_measures", i)).__name__,
                *self.results[i, :, result_index]
            ))



