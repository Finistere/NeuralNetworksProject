from benchmarks import FeatureSelector
from abc import ABCMeta
import numpy as np
# SU
import skfeature.utility.mutual_information
# Relief
import skfeature.function.similarity_based.reliefF
# SVM_RFE
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFE
# Lasso
from sklearn.linear_model import LassoLarsCV


class Dummy(FeatureSelector):
    __name__ = "Dummy"

    def rank(self, data, classes):
        return np.arange(data.shape[0])

    def weight(self, data, classes):
        ranks = np.arange(data.shape[0])
        return ranks / ranks.max()


class SymmetricalUncertainty(FeatureSelector):
    def weight(self, data, classes):
        features_weight = []
        for i in range(0, data.shape[0]):
            features_weight.append(
                skfeature.utility.mutual_information.su_calculation(data[i], classes)
            )
        return self.normalize(features_weight)


class Relief(FeatureSelector):
    def weight(self, data, classes):
        features_weight = skfeature.function.similarity_based.reliefF.reliefF(data.T, classes)

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
    def weight(self, data, classes):
        rfe = RFE(
            estimator=SVC(
                kernel='linear',
                C=self.find_best_hyper_parameter_SVC(data, classes)
            ),
            n_features_to_select=1,
            step=0.1
        )
        rfe.fit(data.T, classes)
        ordered_ranks = self.reverse_order(rfe.ranking_)

        return self.normalize(ordered_ranks)

    def find_best_hyper_parameter_SVC(self, data, classes):
        return self.find_best_hyper_parameter(data, classes, SVC(), "C")

    @staticmethod
    def reverse_order(ranks):
        ordered_ranks = -ranks + np.max(ranks) + 1
        return ordered_ranks


class LassoFeatureSelector(ClassifierFeatureSelector):
    def rank(self, data, classes):
        lasso = LassoLarsCV()
        lasso.fit(data.T, classes)
        nonzero_regularization_parameters = np.ma.masked_array(lasso.coef_path_, [lasso.coef_path_ == 0])
        regularization_parameters_dimension = 1
        features_weight = np.mean(nonzero_regularization_parameters, axis=regularization_parameters_dimension)
        features_rank = self.rank_weights(features_weight)
        return features_rank

    def weight(self, data, classes):
        lasso = LassoLarsCV()
        lasso.fit(data.T, classes)
        nonzero_regularization_parameters = np.ma.masked_array(lasso.coef_path_, [lasso.coef_path_ == 0])
        regularization_parameters_dimension = 1
        features_weight = np.mean(nonzero_regularization_parameters, axis=regularization_parameters_dimension)

        return self.normalize(features_weight)


class RF(FeatureSelector):
    def rank(self, data, classes):
        pass  # TODO

    def weight(self, data, classes):
        pass  # TODO
