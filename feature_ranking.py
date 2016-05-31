from benchmarks import FeatureRanking
import numpy as np
# SU
import skfeature.utility.mutual_information
# Relief
import skfeature.function.similarity_based.reliefF
# SVM_RFE
import sklearn.svm
import sklearn.grid_search
import sklearn.feature_selection


class Dummy(FeatureRanking):

    def rank(self, data, classes):
        return np.arange(data.shape[0])
      
class SymmetricalUncertainty(FeatureRanking):

    def rank(self, data, classes):
        features_weight = self.weight_features(data, classes)
        features_rank = self.rank_weights(features_weight)
        return features_rank

    def weight_features(self, data, classes):
        features_weight = [skfeature.utility.mutual_information.su_calculation(data[i], classes)
                            for i in range(0, data.shape[0])]
        return features_weight


class Relief(FeatureRanking):

    def rank(self, data, classes):
        features_weight = self.weight_features(data, classes)
        features_rank = self.rank_weights(features_weight)
        return features_rank

    def weight_features(self, data, classes):
        features_weight = skfeature.function.similarity_based.reliefF.reliefF(data.T, classes)
        return features_weight 

class SVM_RFE(FeatureRanking):

    def rank(self, data, classes):
        hyperparameter = self.find_hyperparameter_with_grid_search_cv(data,classes)
        linear_svm = sklearn.svm.SVC(kernel='linear', C=hyperparameter)
        recursive_feature_elimination = sklearn.feature_selection.RFE(estimator=linear_svm,
            n_features_to_select=1, step=0.1)
        recursive_feature_elimination.fit(data.T, classes)
        features_rank = self.rerank_reverse_ordered_nonordinal_ranks(recursive_feature_elimination.ranking_)
        return features_rank

    #TODO implement iterative grid search using scipy.stats.expon(scale=100) http://scikit-learn.org/stable/modules/grid_search.html
    def find_hyperparameter_with_grid_search_cv(self, data, classes):
        tuned_parameters = [{'C': [1, 10, 100, 1000]}]
        classifiers = sklearn.grid_search.GridSearchCV(sklearn.svm.SVC(C=1), tuned_parameters, cv=5,
                           scoring='precision')
        classifiers.fit(data.T, classes)
        return classifiers.best_params_['C']

    def rerank_reverse_ordered_nonordinal_ranks(self, ranks):
        ordered_ranks = self.rank_weights(-ranks+np.max(ranks))
        return ordered_ranks

class RF(FeatureRanking):
    def rank(self, data, classes):
        pass #TODO


