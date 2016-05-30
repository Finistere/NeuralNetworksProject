import feature_ranking
import math
import numpy as np

class TestDummy():
    features_weight = [0.5,1.0,0.5,0.7, 0.7]

    def test_rank_weights(self):
        dummy = feature_ranking.Dummy()
        features_rank = dummy.rank_weights(self.features_weight)
        print(features_rank)
        assert np.allclose([1,5,2,3,4], dummy.rank_weights(self.features_weight))

class TestSymmetricalUncertainty():
    data = np.array([[1, 2, 3],
                     [0, 0, 0],
                     [1, 0, 1]])
    classes = np.array([1, 0, 1])

    #TODO def test_rank(self):
        

    def test_weight_features(self):
        h_f = -3 * 1. / 3 * math.log(1./3)
        h_c = -1./3 * math.log(1./3) - 2./3 * math.log(2./3)
        h_fc = 2 * 1./3 * math.log((2./3) / (1./3)) + 1./3 * math.log((1./3) / (1./3))
        su = 2 * (h_f - h_fc) / (h_f + h_c)
        su_ranker = feature_ranking.SymmetricalUncertainty()
        assert np.allclose(su, su_ranker.weight_features(self.data, self.classes)[0])
        
#class TestRelief:

class TestSVM_RFE():
    reversed_ordered_nonordinal_ranks = np.array([1,3,2,1,1,3,4])

    #TODO def test_rank(self):

    #TODO def test_find_hyperparameter_with_grid_search_cv(self):

    def test_rerank_reverse_nonordinal_ordered_ranking(self):
        svm_rfe = feature_ranking.SVM_RFE()
        correctly_ordered_ordinal_ranks = svm_rfe.rerank_reverse_ordered_nonordinal_ranks(
                self.reversed_ordered_nonordinal_ranks)
        assert np.allclose([5,2,4,6,7,3,1], correctly_ordered_ordinal_ranks)
