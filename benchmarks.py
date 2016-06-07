import numpy as np
import scipy.stats
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
import warnings
import multiprocessing
import ctypes


class RobustnessMeasure(metaclass=ABCMeta):
    def run_and_set_in_results(self, features_ranks, results, result_index):
        results[result_index] = self.measure(features_ranks)

    @abstractmethod
    # features ranks is matrix with each rows represent a feature, and the columns its rankings
    def measure(self, features_ranks):
        pass


class FeatureRanking(metaclass=ABCMeta):
    def run_and_append_to_list(self, data, classes, results_list):
        results_list.append(self.rank(data, classes))

    def run_and_set_in_results(self, data, classes, results, result_index):
        results[result_index] = self.rank(data, classes)

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

    def run_and_set_in_results(self, data, classes, train_index, test_index, results, result_index):
        self.classifier.fit(
            data[:, train_index].T,
            classes[train_index]
        )
        results[result_index] = self.classifier.score(
            data[:, test_index].T,
            classes[test_index]
        )


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
            p = multiprocessing.Process(
                target=self.feature_ranking.run_and_append_to_list,
                kwargs={
                    'data': data[:, train_index],
                    'classes': classes[train_index],
                    'results_list': features_ranks
                }
            )
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
            p = multiprocessing.Process(
                target=self.robustness_measures[i].run_and_set_in_results,
                kwargs={
                    'features_ranks': features_ranks,
                    'results': shared_robustness_array,
                    'result_index': i
                }
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return shared_robustness_array


class AccuracyBenchmark:
    def __init__(self, feature_ranking: FeatureRanking, classifiers):
        self.feature_ranking = feature_ranking

        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        self.classifiers = [ClassifierWrapper(c) for c in classifiers]

    def run(self, data, classes, n_folds=10):
        classification_accuracies = np.zeros((n_folds, len(self.classifiers)))
        features_ranks = multiprocessing.Manager().dict()

        cv = KFold(len(classes), n_folds=n_folds)

        processes = []
        for i, (train_index, test_index) in enumerate(cv):
            p = multiprocessing.Process(
                target=self.feature_ranking.run_and_set_in_results,
                kwargs={
                    'data': data[:, train_index],
                    'classes': classes[train_index],
                    'results': features_ranks,
                    'result_index': i
                }
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # multi threading necessary ?
        features_indexes = [self.highest_1percent(ranking) for ranking in features_ranks]

        processes = []
        for i, (train_index, test_index) in enumerate(cv):
            for j, classifier in enumerate(self.classifiers):
                p = multiprocessing.Process(
                    target=classifier.run_and_set_in_results,
                    kwargs={
                        'data': data[features_indexes[i], :],
                        'classes': classes,
                        'train_index': train_index,
                        'test_index': test_index,
                        'results': classification_accuracies,
                        'result_index': (i, j)
                    }
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        return classification_accuracies.mean(axis=0)

    # 1% best features
    @staticmethod
    def highest_1percent(features_rank):
        size = 1 + len(list(features_rank)) // 100
        return np.argsort(features_rank)[:-size:-1]




