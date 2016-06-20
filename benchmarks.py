import numpy as np
import scipy.stats
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
import multiprocessing
import ctypes


class RobustnessMeasure(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

    def run_and_set_in_results(self, features_selection, results, result_index):
        results[result_index] = self.measure(features_selection)

    @abstractmethod
    # features ranks is matrix with each rows represent a feature, and the columns its rankings
    def measure(self, features_ranks):
        pass


class FeatureSelector(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

    def run_and_append_to_list(self, data, labels, results_list, method):
        results_list.append(getattr(self, method)(data, labels))

    def run_and_set_in_results(self, data, labels, results, result_index, method):
        results[result_index] = getattr(self, method)(data, labels)

    @abstractmethod
    # Each column is an observation, each row a feature
    def rank(self, data, classes):
        pass

    @abstractmethod
    # Each column is an observation, each row a feature
    def weight(self, data, classes):
        pass

    def normalize_vector(self, vector, range_begin=0, range_end=1):
        vector_min = np.min(vector)
        vector_normalized = range_begin + (
            (vector - vector_min) * (range_end - range_begin) / (np.max(vector) - vector_min)
        )
        return vector_normalized

    def rank_weights(self, features_weight):
        features_rank = scipy.stats.rankdata(features_weight, method='ordinal')
        return np.array(features_rank)


class ParallelProcessing:
    max_parallelism = multiprocessing.cpu_count()


class Benchmark(metaclass=ABCMeta):
    feature_selector = None
    feature_selection_method = "rank"

    def generate_features_selection(self, data, labels):
        if self.feature_selector is None:
            raise TypeError("feature_ranking needs to be defined")
        generator = FeatureSelectionGenerator(self.feature_selector)
        return generator.generate(data, labels, self.cv(labels.shape[0]), self.feature_selection_method)

    @staticmethod
    def cv(sample_size):
        pass

    @abstractmethod
    def run(self, data, labels, feature_ranks=None):
        pass

    @abstractmethod
    def get_measures(self):
        pass


class FeatureSelectionGenerator(ParallelProcessing):
    def __init__(self, feature_selectors: FeatureSelector):
        self.feature_selectors = feature_selectors
        self.__name__ = self.feature_selectors.__name__

    def generate(self, data, labels, cv, method="rank"):
        features_selection = multiprocessing.Manager().dict()

        with multiprocessing.Pool(processes=self.max_parallelism) as pool:
            for i, (train_index, test_index) in enumerate(cv):
                pool.apply_async(
                    self.feature_selectors.run_and_set_in_results,
                    kwds={
                        'data': data[:, train_index],
                        'labels': labels[train_index],
                        'results': features_selection,
                        'result_index': i,
                        'method': method
                    }
                )
            pool.close()
            pool.join()

        features_selection_list = []
        for i, ranking in features_selection.items():
            features_selection_list.append(ranking)

        return np.array(features_selection_list)


class RobustnessBenchmark(Benchmark, ParallelProcessing):
    def __init__(self, robustness_measures, feature_selector: FeatureSelector = None, feature_selection_method=None):
        self.feature_selector = feature_selector
        if feature_selection_method is not None:
            self.feature_selection_method = feature_selection_method

        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        self.robustness_measures = robustness_measures

    def run(self, data, labels, features_selection=None):
        if features_selection is None:
            features_selection = self.generate_features_selection(data, labels)

        features_selection = np.array(features_selection).T
        shared_array_base = multiprocessing.Array(ctypes.c_double, len(self.robustness_measures))
        shared_robustness_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_robustness_array = shared_robustness_array.reshape(len(self.robustness_measures))

        with multiprocessing.Pool(processes=self.max_parallelism) as pool:
            for i in range(len(self.robustness_measures)):
                pool.apply_async(
                    self.robustness_measures[i].run_and_set_in_results,
                    kwds={
                        'features_selection': features_selection,
                        'results': shared_robustness_array,
                        'result_index': i
                    }
                )
            pool.close()
            pool.join()

        return shared_robustness_array

    @staticmethod
    def cv(sample_length):
        return ShuffleSplit(sample_length, n_iter=10, test_size=0.1)

    def get_measures(self):
        return self.robustness_measures


class ClassifierWrapper:
    def __init__(self, classifier):
        self.classifier = classifier
        self.__name__ = type(classifier).__name__

    def run_and_set_in_results(self, data, labels, train_index, test_index, results, result_index):
        self.classifier.fit(
            data[:, train_index].T,
            labels[train_index]
        )
        results[result_index] = self.classifier.score(
            data[:, test_index].T,
            labels[test_index]
        )


class AccuracyBenchmark(Benchmark, ParallelProcessing):
    percentage_used_in_classification = 0.1
    n_fold = 10

    def __init__(self, classifiers, feature_selector: FeatureSelector = None, feature_selection_method=None):
        self.feature_selector = feature_selector
        if feature_selection_method is not None:
            self.feature_selection_method = feature_selection_method

        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        self.classifiers = [ClassifierWrapper(c) for c in classifiers]

    def run(self, data, labels, features_selection=None):
        if features_selection is None:
            features_selection = self.generate_features_selection(data, labels)

        features_indexes = {}
        for i, ranking in enumerate(features_selection):
            features_indexes[i] = self.highest_percent(ranking, self.percentage_used_in_classification)

        shared_array_base = multiprocessing.Array(ctypes.c_double, AccuracyBenchmark.n_fold * len(self.classifiers))
        classification_accuracies = np.ctypeslib.as_array(shared_array_base.get_obj())
        classification_accuracies = classification_accuracies.reshape((AccuracyBenchmark.n_fold, len(self.classifiers)))

        with multiprocessing.Pool(processes=self.max_parallelism) as pool:
            for i, (train_index, test_index) in enumerate(self.cv(labels.shape[0])):
                for j, classifier in enumerate(self.classifiers):
                    pool.apply_async(
                        classifier.run_and_set_in_results,
                        kwds={
                            'data': data[features_indexes[i], :],
                            'labels': labels,
                            'train_index': train_index,
                            'test_index': test_index,
                            'results': classification_accuracies,
                            'result_index': (i, j)
                        }
                    )
            pool.close()
            pool.join()

        return classification_accuracies.mean(axis=0)

    @staticmethod
    def cv(sample_length):
        return KFold(sample_length, n_folds=AccuracyBenchmark.n_fold)

    # best features
    @staticmethod
    def highest_percent(features_selection, percentage):
        size = 1 + int(len(list(features_selection)) * percentage)
        return np.argsort(features_selection)[:-size:-1]

    def get_measures(self):
        return self.classifiers
