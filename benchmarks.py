import numpy as np
import scipy.stats
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
import multiprocessing
import ctypes


class RobustnessMeasure(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

    def run_and_set_in_results(self, features_ranks, results, result_index):
        results[result_index] = self.measure(features_ranks)

    @abstractmethod
    # features ranks is matrix with each rows represent a feature, and the columns its rankings
    def measure(self, features_ranks):
        pass


class FeatureRanking(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

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


class Benchmark(metaclass=ABCMeta):
    feature_ranking = None

    def __generate_features_ranks(self, data, labels):
        if self.feature_ranking is None:
            raise TypeError("feature_ranking needs to be defined")
        generator = FeatureRanksGenerator(self.feature_ranking)
        return generator.generate(data, labels, self.cv(labels.shape[0]))

    @staticmethod
    def cv(sample_size):
        pass

    @abstractmethod
    def run(self, data, labels, feature_ranks=None):
        pass

    @abstractmethod
    def get_measures(self):
        pass


class FeatureRanksGenerator:
    def __init__(self, feature_ranking: FeatureRanking):
        self.feature_ranking = feature_ranking
        self.__name__ = self.feature_ranking.__name__

    def generate(self, data, labels, cv):
        features_ranks = multiprocessing.Manager().dict()
        processes = []

        for i, (train_index, test_index) in enumerate(cv):
            p = multiprocessing.Process(
                target=self.feature_ranking.run_and_set_in_results,
                kwargs={
                    'data': data[:, train_index],
                    'classes': labels[train_index],
                    'results': features_ranks,
                    'result_index': i
                }
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        features_ranks_list = []
        for i, ranking in features_ranks.items():
            features_ranks_list.append(ranking)

        return np.array(features_ranks_list)


class RobustnessBenchmark(Benchmark):
    def __init__(self, robustness_measures, feature_ranking: FeatureRanking = None):
        self.feature_ranking = feature_ranking

        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        self.robustness_measures = robustness_measures

    def run(self, data, labels, features_ranks=None):
        if features_ranks is None:
            features_ranks = self.__generate_features_ranks(data, labels)

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

    @staticmethod
    def cv(sample_length):
        return ShuffleSplit(sample_length, n_iter=10, test_size=0.1)

    def get_measures(self):
        return self.robustness_measures


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


class AccuracyBenchmark(Benchmark):
    percentage_used_in_classification = 0.1
    n_fold = 10

    def __init__(self, classifiers, feature_ranking: FeatureRanking = None):
        self.feature_ranking = feature_ranking

        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        self.classifiers = [ClassifierWrapper(c) for c in classifiers]

    def run(self, data, labels, features_ranks=None):
        if features_ranks is None:
            features_ranks = self.__generate_features_ranks(data, labels)

        features_indexes = {}
        for i, ranking in enumerate(features_ranks):
            features_indexes[i] = self.highest_percent(ranking, self.percentage_used_in_classification)

        shared_array_base = multiprocessing.Array(ctypes.c_double, AccuracyBenchmark.n_fold * len(self.classifiers))
        classification_accuracies = np.ctypeslib.as_array(shared_array_base.get_obj())
        classification_accuracies = classification_accuracies.reshape((AccuracyBenchmark.n_fold, len(self.classifiers)))

        processes = []
        for i, (train_index, test_index) in enumerate(self.cv(labels.shape[0])):
            for j, classifier in enumerate(self.classifiers):
                p = multiprocessing.Process(
                    target=classifier.run_and_set_in_results,
                    kwargs={
                        'data': data[features_indexes[i], :],
                        'classes': labels,
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

    @staticmethod
    def cv(sample_length):
        return KFold(sample_length, n_folds=AccuracyBenchmark.n_fold)

    # 1% best features
    @staticmethod
    def highest_percent(features_rank, percentage):
        size = 1 + int(len(list(features_rank)) * percentage)
        return np.argsort(features_rank)[:-size:-1]

    def get_measures(self):
        return self.classifiers
