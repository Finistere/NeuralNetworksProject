import numpy as np
import multiprocessing
import os
import errno
from data_sets import DataSets, PreComputedData
from feature_selector import FeatureSelector


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


class FeatureSelection(FeatureSelectionGenerator):
    def load(self, data_set, cv, method):
        try:
            return PreComputedData.load(data_set, cv, method, self)
        except FileNotFoundError:
            return getattr(self, method)(data_set, cv)

    def rank(self, data_set, cv):
        try:
            return PreComputedData.load(data_set, cv, "weight", self)
        except FileNotFoundError:
            weights = self.weight(data_set, cv)

            ranks = np.array([FeatureSelector.rank_weights(w) for w in weights])
            self.__save(data_set, cv, "rank", ranks)

            return ranks

    def weight(self, data_set, cv):
        print("=> Generating feature {method}s of {data_set} ({cv}) with {feature_selector}".format(
            method="weight",
            data_set=data_set,
            feature_selector=self.__name__,
            cv=type(cv).__name__
        ))

        data, labels = DataSets.load(data_set)

        try:
            cv_indices = PreComputedData.load_cv(data_set, cv)
        except FileNotFoundError:
            try:
                os.makedirs(PreComputedData.cv_dir(data_set, cv))
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            cv_indices = list(cv)
            np.save(PreComputedData.cv_file_name(data_set, cv), cv_indices)

        weights = self.generate(data, labels, cv_indices, "weight")
        self.__save(data_set, cv, "weight", weights)

        return weights

    def __save(self, data_set, cv, method, feature_selection):
        try:
            os.makedirs(PreComputedData.dir_name(data_set, cv, method))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        np.save(PreComputedData.file_name(data_set, cv, method, self), feature_selection)