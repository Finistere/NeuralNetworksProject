import numpy as np
import multiprocessing
import os
import errno
from data_sets import DataSets, Weights
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
            return Weights.load(data_set, cv, method, self)
        except FileNotFoundError:
            return self.__gen(data_set, cv, method)

    def __gen(self, data_set, cv, method):
        data, labels = DataSets.load(data_set)

        print("Generating feature {method}s of {data_set} ({cv}) with {feature_selector}".format(
            method=method,
            data_set=data_set,
            feature_selector=self.__name__,
            cv=type(cv).__name__
        ))
        ranks = self.generate(data, labels, cv, method)

        try:
            os.makedirs(self.__dir_name(data_set, cv, method))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        np.save(self.__file_name(data_set, cv, method), ranks)

        return ranks