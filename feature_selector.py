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
from sklearn.cross_validation import KFold



class Dummy(FeatureSelector):
    __name__ = "Dummy"

    def rank(self, data, classes):
        return np.arange(data.shape[0])

    def weight(self, data, classes):
        ranks = np.arange(data.shape[0])
        return ranks / ranks.max()


class SymmetricalUncertainty(FeatureSelector):
    def rank(self, data, classes):
        features_weight = self.weight_features(data, classes)
        features_rank = self.rank_weights(features_weight)
        return features_rank

    def weight(self, data, classes):
        features_weight = self.weight_features(data, classes)
        return features_weight

    def weight_features(self, data, classes):
        features_weight = [skfeature.utility.mutual_information.su_calculation(data[i], classes)
                           for i in range(0, data.shape[0])]
        return features_weight


class Relief(FeatureSelector):
    def rank(self, data, classes):
        features_weight = self.weight_features(data, classes)
        features_rank = self.rank_weights(features_weight)
        return features_rank

    def weight(self, data, classes):
        features_weight = skfeature.function.similarity_based.reliefF.reliefF(data.T, classes)
        features_weight_normalized = self.normalize_vector(features_weight)
        return features_weight_normalized

    def weight_features(self, data, classes):
        features_weight = skfeature.function.similarity_based.reliefF.reliefF(data.T, classes)
        return features_weight


class ClassifierFeatureSelector(FeatureSelector, metaclass=ABCMeta):
    # TODO implement iterative grid search using scipy.stats.expon(scale=100) http://scikit-learn.org/stable/modules/grid_search.html
    def find_best_hyperparameter(self, data, classes, classifier, parameter):
        tuned_parameters = [{parameter: [1, 10, 100, 1000]}]
        classifiers = GridSearchCV(
            classifier, tuned_parameters, cv=5, scoring='precision')
        classifiers.fit(data.T, classes)
        return classifiers.best_params_[parameter]


class SVM_RFE(ClassifierFeatureSelector):
    def rank(self, data, classes):
        hyperparameter = self.find_best_hyperparameter_SVC(data, classes)
        linear_svm = SVC(kernel='linear', C=hyperparameter)
        recursive_feature_elimination = RFE(
            estimator=linear_svm, n_features_to_select=1, step=0.1)
        recursive_feature_elimination.fit(data.T, classes)
        ordered_ranks = self.reverse_order(recursive_feature_elimination.ranking_)
        features_rank = self.rank_weights(ordered_ranks)
        return features_rank

    def weight(self, data, classes):
        hyperparameter = self.find_best_hyperparameter_SVC(data, classes)
        linear_svm = SVC(kernel='linear', C=hyperparameter)
        recursive_feature_elimination = RFE(
            estimator=linear_svm, n_features_to_select=1, step=0.1)
        recursive_feature_elimination.fit(data.T, classes)
        ordered_ranks = self.reverse_order(recursive_feature_elimination.ranking_)
        rank_normalized = self.normalize_vector(ordered_ranks)
        return rank_normalized

    def find_best_hyperparameter_SVC(self, data, classes):
        return self.find_best_hyperparameter(data, classes, SVC(), "C")

    def reverse_order(self, ranks):
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
        features_weight_normalized = self.normalize_vector(features_weight)
        return features_weight_normalized

class RF(FeatureSelector):
    def rank(self, data, classes):
        pass  # TODO

    def weight(self, data, classes):
        pass  # TODO
