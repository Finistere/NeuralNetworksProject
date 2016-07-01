from abc import ABCMeta, abstractmethod
from sklearn import preprocessing
import numpy as np
import scipy.stats
# SU
import skfeature.utility.mutual_information
# Relief
import skfeature.function.similarity_based.reliefF
# SVM_RFE
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFE
# Lasso
from sklearn.linear_model import LassoCV


class FeatureSelector(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

    def run_and_set_in_results(self, data, labels, results, result_index, method):
        results[result_index] = getattr(self, method)(data, labels)

    # Each column is an observation, each row a feature
    def rank(self, data, labels):
        return self.rank_weights(self.weight(data, labels))

    @abstractmethod
    # Each column is an observation, each row a feature
    def weight(self, data, labels):
        pass

    @staticmethod
    def normalize(vector):
        return preprocessing.MinMaxScaler().fit_transform(vector)

    @staticmethod
    def rank_weights(features_weight):
        features_rank = scipy.stats.rankdata(features_weight, method='ordinal')
        # shuffle same features
        unique_values = np.unique(features_weight)
        for unique_value in unique_values:
            unique_value_args = np.argwhere(features_weight == unique_value).reshape(-1)
            unique_value_args_shuffled = np.random.permutation(unique_value_args)
            features_rank[unique_value_args] = features_rank[unique_value_args_shuffled]
        return features_rank


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
            n_features_to_select=round(len(data)*0.01),
            step=self.step
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
    def rank(self, data, labels):
        features_rank = np.arange(1,len(data)+1)
        np.random.shuffle(features_rank)
        return features_rank

    def weight(self, data, labels):
        weights = np.random.uniform(0,1,len(data))
        return weights

class RF(FeatureSelector):
    def rank(self, data, labels):
        pass  # TODO

    def weight(self, data, labels):
        pass  # TODO
