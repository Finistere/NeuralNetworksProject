import numpy as np
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod


class RobustnessMeasure(metaclass=ABCMeta):
    @abstractmethod
    # features ranks is matrix with each rows represent a feature, and the columns its rankings
    def measure(self, features_ranks) -> float:
        pass


class FeatureRanking(metaclass=ABCMeta):
    @abstractmethod
    # Each column is an observation, each row a feature
    def rank(self, data, classes):
        pass


class Benchmark:
    def __init__(self, feature_ranking: FeatureRanking, robustness_measure: RobustnessMeasure = None, classifier=None):
        self.feature_ranking = feature_ranking
        self.robustness_measure = robustness_measure
        self.classifier = classifier

    def run(self, data, classes):
        robustness = self.robustness(data, classes)
        accuracy = self.classification_accuracy(data, classes)

        return robustness, accuracy

    def robustness(self, data, classes, n_iter=10, test_size=0.1):
        if self.robustness_measure is None:
            raise ValueError("Robustness measure is not defined")

        features_ranks = []
        cv = ShuffleSplit(len(classes), n_iter=n_iter, test_size=test_size)

        for train_index, test_index in cv:
            features_ranks.append(self.feature_ranking.rank(data[:, train_index], classes[train_index]))

        return self.robustness_measure.measure(np.array(features_ranks).T)

    def classification_accuracy(self, data, classes, n_folds=10):
        if self.classifier is None:
            raise ValueError("Classifer is not defined")

        classification_accuracies = []

        cv = KFold(len(classes), n_folds=n_folds)

        for train_index, test_index in cv:

            features_rank = self.feature_ranking.rank(data[:, train_index], classes[train_index])
            features_index = self.highest_1percent(features_rank)

            self.classifier.fit(data[np.ix_(features_index, train_index)].T, classes[train_index])
            classification_accuracies.append(
                self.classifier.score(data[np.ix_(features_index, test_index)].T, classes[test_index])
            )

        return np.mean(classification_accuracies)

    # 1% best features
    @staticmethod
    def highest_1percent(features_rank):
        size = 1 + len(list(features_rank)) // 100
        return np.argsort(features_rank)[:-size:-1]


class RobustnessExperiment:
    def __init__(self, robustness_measures=None, feature_rankings=None):
        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        if not isinstance(feature_rankings, list):
            feature_rankings = [feature_rankings]

        results_shape = [len(robustness_measures), len(feature_rankings)]

        self.robustness_measures = robustness_measures
        self.feature_rankings = feature_rankings
        self.results = np.zeros((results_shape[0], results_shape[1]))

    def run(self, data, classes):
        for i in range(self.results.shape[0]):
            for j in range(self.results.shape[1]):
                benchmark = Benchmark(
                    robustness_measure=self.robustness_measures[i],
                    feature_ranking=self.feature_rankings[j]
                )
                self.results[i, j] = benchmark.robustness(data, classes)

        return self.results

    def print_results(self):
        print("ROBUSTNESS \n")
        header = "{:22} " + " | ".join(["{:10}"] * self.results.shape[1])
        row = "{:20} : " + " | ".join(["{:10.2%}"] * self.results.shape[1])

        print(header.format(
            "",
            *[type(self.robustness_measures[i]).__name__ for i in range(self.results.shape[1])]
        ))
        for i in range(self.results.shape[0]):
            print(row.format(
                type(self.robustness_measures[i]).__name__,
                *self.results[i, :]
            ))



