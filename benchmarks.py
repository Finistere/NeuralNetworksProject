import numpy as np
from sklearn.cross_validation import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
import multiprocessing
import ctypes
from feature_selector import FeatureSelector
from robustness_measure import RobustnessMeasure, JaccardIndex
from feature_selection import FeatureSelectionGenerator


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


class RobustnessBenchmark(Benchmark):
    def __init__(self, robustness_measures, feature_selector: FeatureSelector = None):
        self.feature_selector = feature_selector

        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        for robustness_measure in robustness_measures:
            if not isinstance(robustness_measure, RobustnessMeasure):
                raise ValueError("At least one robustness measure does not inherit RobustnessMeasure")

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
