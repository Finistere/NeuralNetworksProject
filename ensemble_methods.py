from benchmarks import Benchmark
from feature_selector import DataSetFeatureSelector
import numpy as np
import scipy.stats
from abc import ABCMeta, abstractmethod
from data_sets import DataSets, PreComputedData
from sklearn.cross_validation import KFold
import multiprocessing
from sklearn.metrics import f1_score


class EnsembleMethod(DataSetFeatureSelector, metaclass=ABCMeta):
    max_parallelism = multiprocessing.cpu_count()

    def __init__(self, data_set_feature_selectors):
        super().__init__()

        if not isinstance(data_set_feature_selectors, list):
            data_set_feature_selectors = [data_set_feature_selectors]

        for data_set_feature_selector in data_set_feature_selectors:
            if not isinstance(data_set_feature_selector, DataSetFeatureSelector):
                raise ValueError("Only DataSetFeatureSelector can be used")

        self.feature_selectors = data_set_feature_selectors

    def rank_data_set(self, data_set, cv_generator):
        super().rank_data_set(data_set, cv_generator)
        bench_features_selection = []

        _, labels = DataSets.load(data_set)
        cv = cv_generator(labels.shape[0])

        for f in self.feature_selectors:
            bench_features_selection.append(f.weight_data_set(data_set, cv_generator))
        bench_features_selection = np.array(bench_features_selection)

        data, labels = DataSets.load(data_set)
        cv_indices = PreComputedData.load_cv(data_set, cv)

        feature_selection = multiprocessing.Manager().dict()

        with multiprocessing.Pool(processes=self.max_parallelism) as pool:
            for i in range(bench_features_selection.shape[1]):
                pool.apply_async(
                    self.run_and_set_in_results,
                    kwds={
                        'results': feature_selection,
                        'result_index': i,
                        'feature_selection': bench_features_selection[:, i],
                        'data': data[:, cv_indices[i][0]],
                        'labels': labels[cv_indices[i][0]]
                    }
                )
            pool.close()
            pool.join()

        return np.array([ranking for i, ranking in feature_selection.items()])

    def weight_data_set(self, data_set, cv_generator):
        return self.normalize(self.rank_data_set(data_set, cv_generator))

    def run_and_set_in_results(self, results, result_index, feature_selection, data, labels):
        np.random.seed()
        results[result_index] = self.rank_weights(self.combine(feature_selection, data, labels))

    @abstractmethod
    def combine(self, feature_selection, data, labels):
        pass


class Influence(EnsembleMethod):
    def combine(self, feature_selection, data, labels):
        feature_selection = np.exp(feature_selection)
        return (feature_selection.T / feature_selection.sum(axis=1)).T.mean(axis=0)


class InfluenceStd(EnsembleMethod):
    def combine(self, feature_selection, data, labels):
        feature_selection = np.exp(feature_selection)
        influence = (feature_selection.T / feature_selection.sum(axis=1)).T
        return influence.mean(axis=0) / (1 + influence.std(axis=0))


class Mean(EnsembleMethod):
    def __init__(self, feature_selectors, power=1):
        super().__init__(feature_selectors)
        self.__name__ = "Mean - {}".format(power)
        self.power = power

    def combine(self, features_selection, data, labels):
        return np.power(features_selection, self.power).mean(axis=0)


class MeanStd(EnsembleMethod):
    def __init__(self, feature_selectors, power=1):
        super().__init__(feature_selectors)
        self.__name__ = "Mean std - {}".format(power)
        self.power = power

    def combine(self, features_selection, data, labels):
        influence = (features_selection.T / features_selection.sum(axis=1)).T
        return np.power(features_selection, self.power).mean(axis=0) / features_selection.std(axis=0)


class SMean(EnsembleMethod):
    def __init__(self, feature_selectors, min_mean_max=[1, 1, 1]):
        super().__init__(feature_selectors)
        self.weights = np.array(min_mean_max)
        self.__name__ = "SMean - {} {} {}".format(*min_mean_max)

    def combine(self, features_selection, data, labels):
        f_mean = np.mean(features_selection, axis=0)
        f_max = np.max(features_selection, axis=0)
        f_min = np.min(features_selection, axis=0)
        return (np.vstack((f_min, f_mean, f_max)) * self.weights[:, np.newaxis]).mean(axis=0)


class SMeanWithClassifier(EnsembleMethod):
    def __init__(self, feature_selectors, classifiers, min_mean_max=[1, 1, 1], **kwargs):
        super().__init__(feature_selectors, **kwargs)
        self.weights = np.array(min_mean_max)
        self.__name__ = "SMeanWithClassifier - {} {} {}".format(*min_mean_max)
        self.classifiers = classifiers

    def combine(self, features_selection, data, labels):
        cv = KFold(labels.shape[0])
        accuracy = np.zeros(len(self.feature_selectors))

        for i in range(len(self.feature_selectors)):
            best_features_indices = np.argsort(features_selection[i])[:-int(features_selection[i].shape[0] / 100):-1]
            for train_index, test_index in cv:
                for c in self.classifiers:
                    c.fit(data[np.ix_(best_features_indices, train_index)].T, labels[train_index])
                    accuracy[i] += f1_score(
                        labels[test_index],
                        c.predict(data[np.ix_(best_features_indices, test_index)].T)
                    )

        features_selection = (features_selection.T * np.exp(accuracy)).T

        f_mean = np.mean(features_selection, axis=0)
        f_max = np.max(features_selection, axis=0)
        f_min = np.min(features_selection, axis=0)
        return (np.vstack((f_min, f_mean, f_max)) * self.weights[:, np.newaxis]).mean(axis=0)

