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

    def run_and_set_in_results(self, data, labels, results, result_index, method):
        results[result_index] = getattr(self, method)(data, labels)

    # Each column is an observation, each row a feature
    def rank(self, data, classes):
        return self.rank_weights(self.weight(data, classes))

    @abstractmethod
    # Each column is an observation, each row a feature
    def weight(self, data, classes):
        pass

    @staticmethod
    def normalize(vector):
        v_min = np.min(vector)
        v_max = np.max(vector)
        return (vector - v_min) / (v_max - v_min)

    @staticmethod
    def rank_weights(features_weight):
        features_rank = scipy.stats.rankdata(features_weight, method='ordinal')
        return np.array(features_rank)


class Benchmark(metaclass=ABCMeta):
    feature_selector = None

    def generate_features_selection(self, data, labels):
        if self.feature_selector is None:
            raise TypeError("feature_ranking needs to be defined")
        generator = FeatureSelectionGenerator(self.feature_selector)
        return generator.generate(data, labels, self.cv(labels.shape[0]), "rank")

    @staticmethod
    def cv(sample_size):
        pass

    @abstractmethod
    def run(self, data, labels, features_selection=None):
        pass

    @abstractmethod
    def get_measures(self):
        pass


class FeatureSelectionGenerator:
    max_parallelism = multiprocessing.cpu_count()

    def __init__(self, feature_selectors: FeatureSelector):
        self.feature_selectors = feature_selectors
        self.__name__ = self.feature_selectors.__name__

    def generate(self, data, labels, cv, method):
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

        return np.array([ranking for i, ranking in features_selection.items()])


class RobustnessBenchmark(Benchmark):
    def __init__(self, robustness_measures, feature_selector: FeatureSelector = None):
        self.feature_selector = feature_selector

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

        processes = []
        for i in range(len(self.robustness_measures)):
            p = multiprocessing.Process(
                target=self.robustness_measures[i].run_and_set_in_results,
                kwargs={
                    'features_selection': features_selection,
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


class AccuracyBenchmark(Benchmark):
    percentage_of_features = 0.01
    n_fold = 10

    def __init__(self, classifiers, feature_selector: FeatureSelector = None, percentage_of_features=None):
        self.feature_selector = feature_selector

        if percentage_of_features is not None:
            self.percentage_of_features = percentage_of_features

        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        self.classifiers = [ClassifierWrapper(c) for c in classifiers]

    def run(self, data, labels, features_selection=None):
        if features_selection is None:
            features_selection = self.generate_features_selection(data, labels)

        features_indexes = {}
        for i, ranking in enumerate(features_selection):
            features_indexes[i] = self.highest_percent(ranking, self.percentage_of_features)

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
                        'labels': labels,
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
    def highest_percent(features_selection, percentage):
        size = 1 + int(len(list(features_selection)) * percentage)
        return np.argsort(features_selection)[:-size:-1]

    def get_measures(self):
        return self.classifiers


class FMeasureBenchmark:
    def __init__(self, classifiers, feature_selector: FeatureSelector = None, jaccard_percentage=0.01, beta=1):
        from robustness_measure import JaccardIndex
        self.robustness_benchmark = RobustnessBenchmark(
            JaccardIndex(percentage=jaccard_percentage),
            feature_selector=feature_selector
        )
        self.accuracy_benchmark = AccuracyBenchmark(
            classifiers,
            feature_selector=feature_selector,
            percentage_of_features=jaccard_percentage
        )
        self.beta = beta

    def run(self, data, labels, robustness_features_selection=None, accuracy_features_selection=None):
        return np.mean(self.f_measure(
            self.robustness_benchmark.run(data, labels, robustness_features_selection),
            self.accuracy_benchmark.run(data, labels, accuracy_features_selection),
            self.beta
        ))

    @staticmethod
    def f_measure(robustness, accuracy, beta=1):
        return (beta ** 2 * robustness * accuracy) / (beta ** 2 * robustness + accuracy)
