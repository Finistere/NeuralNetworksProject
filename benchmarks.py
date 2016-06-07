import numpy as np
import scipy.stats
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
import warnings
import multiprocessing
import ctypes


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


class ClassifierWrapper:
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, *args, **kwargs):
        self.classifier.fit(*args, **kwargs)

    def run(self, data, classes, list_to_which_append_the_result):
        list_to_which_append_the_result.append(self.classifier.score(data, classes))


class RobustnessBenchmark:
    def __init__(self, feature_ranking: FeatureRanking, robustness_measures):
        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        for robustness_measure in robustness_measures:
            if not isinstance(robustness_measure, RobustnessMeasure):
                warnings.warn("Not all robustness measures are of type RobustnessMeasure")

        self.feature_ranking = feature_ranking
        self.robustness_measures = robustness_measures

    def run(self, data, classes, n_iter=10, test_size=0.1):
        if self.robustness_measures is None:
            raise ValueError("Robustness measures is not defined")

        features_ranks = multiprocessing.Manager().list()
        cv = ShuffleSplit(len(classes), n_iter=n_iter, test_size=test_size)

        processes = []
        for train_index, test_index in cv:
            p = multiprocessing.Process(target=self.feature_ranking.run, kwargs={
                'data': data[:, train_index],
                'classes': classes[train_index],
                'list_to_which_append_the_result': features_ranks
            })
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        features_ranks = np.array(features_ranks).T

        shared_array_base = multiprocessing.Array(ctypes.c_double, len(self.robustness_measures))
        shared_robustness_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_robustness_array = shared_robustness_array.reshape(len(self.robustness_measures))

        processes = []
        for i in range(len(self.robustness_measures)):
            p = multiprocessing.Process(target=self.robustness_measures[i].run, kwargs={
                'features_ranks': features_ranks,
                'results': shared_robustness_array,
                'result_index': i
            })
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return shared_robustness_array


class AccuracyBenchmark:
    def __init__(self, feature_ranking: FeatureRanking, classifier=None):
        self.feature_ranking = feature_ranking
        self.classifier = classifier

    def run(self, data, classes, n_folds=10):
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




