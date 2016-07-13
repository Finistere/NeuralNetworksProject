from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.stats

from sklearn import preprocessing
# SU
import skfeature.utility.mutual_information
# Relief
import skfeature.function.similarity_based.reliefF
# SVM_RFE
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn_rfe import RFE
# Lasso
from sklearn.linear_model import LassoCV
from data_sets import DataSets, PreComputedData
import multiprocessing
import os
import errno


class DataSetFeatureSelector(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

    @staticmethod
    def check_data_set_and_cv(data_set, cv_generator):
        if not callable(cv_generator):
            raise ValueError("cv_generator should be callable")
        if data_set not in DataSets.data_sets:
            raise ValueError("No data set found with the name {}".format(data_set))

    @abstractmethod
    def rank_data_set(self, data_set, cv_generator):
        self.check_data_set_and_cv(data_set, cv_generator)

    @abstractmethod
    def weight_data_set(self, data_set, cv_generator):
        self.check_data_set_and_cv(data_set, cv_generator)

    @staticmethod
    def normalize(vector):
        return preprocessing.MinMaxScaler().fit_transform(vector)

    @staticmethod
    def rank_weights(features_weight):
        features_rank = scipy.stats.rankdata(features_weight, method='ordinal')

        # shuffle same features
        for unique_value in np.unique(features_weight):
            unique_value_args = np.argwhere(features_weight == unique_value).reshape(-1)
            unique_value_args_shuffled = np.random.permutation(unique_value_args)
            features_rank[unique_value_args] = features_rank[unique_value_args_shuffled]

        return features_rank


class FeatureSelector(DataSetFeatureSelector, metaclass=ABCMeta):
    max_parallelism = multiprocessing.cpu_count()

    # Each column is an observation, each row a feature
    def rank(self, data, labels):
        return self.rank_weights(self.weight(data, labels))

    @abstractmethod
    # Each column is an observation, each row a feature
    def weight(self, data, labels):
        pass

    def generate(self, data, labels, cv, method):
        features_selection = multiprocessing.Manager().dict()

        with multiprocessing.Pool(processes=self.max_parallelism) as pool:
            for i, (train_index, test_index) in enumerate(cv):
                pool.apply_async(
                    self.run_and_set_in_results,
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

    def run_and_set_in_results(self, data, labels, results, result_index, method):
        np.random.seed()
        results[result_index] = getattr(self, method)(data, labels)

    def rank_data_set(self, data_set, cv_generator):
        super().rank_data_set(data_set, cv_generator)

        data, labels = DataSets.load(data_set)
        cv = cv_generator(labels.shape[0])

        try:
            return PreComputedData.load(data_set, cv, "rank", self)
        except FileNotFoundError:
            weights = self.weight_data_set(data_set, cv_generator)

            ranks = np.array([self.rank_weights(w) for w in weights])
            self.__save(data_set, cv, "rank", ranks)

            return ranks

    def weight_data_set(self, data_set, cv_generator):
        super().weight_data_set(data_set, cv_generator)

        data, labels = DataSets.load(data_set)
        cv = cv_generator(labels.shape[0])

        try:
            return PreComputedData.load(data_set, cv, "weight", self)
        except FileNotFoundError:

            print("=> Generating feature {method}s of {data_set} ({cv}) with {feature_selector}".format(
                method="weight",
                data_set=data_set,
                feature_selector=self.__name__,
                cv=type(cv).__name__
            ))

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


class Dummy(FeatureSelector):
    __name__ = "Dummy"

    def rank(self, data, labels):
        return np.arange(data.shape[0])

    def weight(self, data, labels):
        ranks = np.arange(data.shape[0])
        return ranks / ranks.max()


class SymmetricalUncertainty(FeatureSelector):
    def weight(self, data, labels):
        features_weight = []
        for i in range(0, data.shape[0]):
            features_weight.append(
                skfeature.utility.mutual_information.su_calculation(data[i], labels)
            )
        return np.array(features_weight)


class Relief(FeatureSelector):
    def weight(self, data, labels):
        features_weight = skfeature.function.similarity_based.reliefF.reliefF(data.T, labels)

        return self.normalize(features_weight)


class ClassifierFeatureSelector(FeatureSelector, metaclass=ABCMeta):
    # TODO implement iterative grid search using scipy.stats.expon(scale=100) http://scikit-learn.org/stable/modules/grid_search.html
    @staticmethod
    def find_best_hyper_parameter(data, classes, classifier, parameter):
        grid_search = GridSearchCV(
            classifier,
            {
                parameter: [1, 10, 100, 1000]
            },
            cv=5,
            scoring='precision'
        )
        grid_search.fit(data.T, classes)
        return grid_search.best_params_[parameter]


class SVM_RFE(ClassifierFeatureSelector):
    def __init__(self, step=0.1):
        super().__init__()
        self.step = step

    def weight(self, data, labels):
        rfe = RFE(
            estimator=SVC(
                kernel='linear',
                C=self.find_best_hyper_parameter_SVC(data, labels)
            ),
            n_features_to_select=round(len(data) * 0.01),
            step=self.step,
            stepwise_selection=True
        )
        rfe.fit(data.T, labels)
        ordered_ranks = self.reverse_order(rfe.ranking_)

        return self.normalize(ordered_ranks)

    def find_best_hyper_parameter_SVC(self, data, classes):
        return self.find_best_hyper_parameter(data, classes, SVC(), "C")

    @staticmethod
    def reverse_order(ranks):
        ordered_ranks = -ranks + np.max(ranks) + 1
        return ordered_ranks


class LassoFeatureSelector(ClassifierFeatureSelector):
    def rank(self, data, labels):
        lasso = LassoCV(cv=2, normalize=True)
        lasso.fit(data.T, labels)
        features_rank = self.rank_weights(np.abs(lasso.coef_))
        return features_rank

    def weight(self, data, labels):
        lasso = LassoCV(cv=2, normalize=True)
        lasso.fit(data.T, labels)
        normalized = self.normalize(np.abs(lasso.coef_))
        return normalized


class Random(ClassifierFeatureSelector):
    # def rank(self, data, labels):
    #     features_rank = np.arange(1, len(data) + 1)
    #     np.random.shuffle(features_rank)
    #     return features_rank

    def weight(self, data, labels):
        weights = np.random.uniform(0, 1, len(data))
        return weights


class RF(FeatureSelector):
    def rank(self, data, labels):
        pass  # TODO

    def weight(self, data, labels):
        pass  # TODO
