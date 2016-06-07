import numpy as np
import scipy.stats
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
from tabulate import tabulate
import warnings
from threading import Thread


class RobustnessMeasure(metaclass=ABCMeta):
    def run(self, features_ranks, results, result_index):
        results[result_index] = self.measure(features_ranks)

    @abstractmethod
    # features ranks is matrix with each rows represent a feature, and the columns its rankings
    def measure(self, features_ranks):
        pass


class FeatureRanking(metaclass=ABCMeta):
    def run(self, data, classes, list_to_which_append_the_result):
        list_to_which_append_the_result.append(self.rank(data, classes))

    @abstractmethod
    # Each column is an observation, each row a feature
    def rank(self, data, classes):
        pass

    def rank_weights(self, features_weight):
        features_rank = scipy.stats.rankdata(features_weight, method='ordinal') 
        return np.array(features_rank)


class Benchmark:
    def __init__(self, feature_ranking: FeatureRanking, robustness_measures, classifier=None):
        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        for robustness_measure in robustness_measures:
            if not isinstance(robustness_measure, RobustnessMeasure):
                warnings.warn("Not all robustness measures are of type RobustnessMeasure")

        self.feature_ranking = feature_ranking
        self.robustness_measures = robustness_measures
        self.classifier = classifier

    def run(self, data, classes):
        robustness = self.robustness(data, classes)
        accuracy = self.classification_accuracy(data, classes)

        return robustness, accuracy

    def robustness(self, data, classes, n_iter=10, test_size=0.1):
        if self.robustness_measures is None:
            raise ValueError("Robustness measures is not defined")

        features_ranks = []
        cv = ShuffleSplit(len(classes), n_iter=n_iter, test_size=test_size)

        threads = []
        for train_index, test_index in cv:
            t = Thread(target=self.feature_ranking.run, kwargs={
                'data': data[:, train_index],
                'classes': classes[train_index],
                'list_to_which_append_the_result': features_ranks
            })
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        features_ranks = np.array(features_ranks).T
        robustness = np.zeros(len(self.robustness_measures))

        threads = []
        for i in range(len(self.robustness_measures)):
            t = Thread(target=self.robustness_measures[i].run, kwargs={
                'features_ranks': features_ranks,
                'results': robustness,
                'result_index': i
            })
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return robustness

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

        results_shape = (len(robustness_measures), len(feature_rankings))

        self.robustness_measures = robustness_measures
        self.feature_rankings = feature_rankings
        self.results = np.zeros(results_shape)

    def run(self, data, classes):
        for i in range(self.results.shape[1]):
            benchmark = Benchmark(
                robustness_measures=self.robustness_measures,
                feature_ranking=self.feature_rankings[i]
            )
            self.results[:, i] = benchmark.robustness(data, classes)

        return self.results

    def print_results(self):
        print("Robustness Experiment : ")
        headers = [type(self.feature_rankings[i]).__name__ for i in range(self.results.shape[1])]
        rows = []
        for i in range(self.results.shape[0]):
            row = [self.robustness_measures[i].__name__]
            row += map(lambda i: "{:.2%}".format(i), self.results[i, :].tolist())
            rows.append(row)

        print(tabulate(rows, headers, tablefmt='pipe'))



