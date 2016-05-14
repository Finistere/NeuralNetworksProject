import robustness as rob
import numpy as np

def random_state(seed): return np.random.mtrand.RandomState(seed)

class TestRobustnessMeasurement():
    measurement = rob.RobustnessMeasurement(
        [
            [1, 2, 3],
            [0, 0, 0],
            [1, 0, 1]
        ],
        [1, 0, 1]
    )

    # for now I do not test functions with random variables
    # see http://programmers.stackexchange.com/questions/225523/what-are-good-unit-tests-to-cover-the-use-case-of-rolling-a-die
    #def test_simple_random_sampling(self):
    def test_rank_features(self):
        features_weight = np.array([[ 0.8,  0.9,  0.4],
                                    [ 0.1,  0.8,  0.9]])
        ranking = self.rank_features(features_weight)
        assert np.all(ranking == [[2,3,1],[1,2,3]])

    def test_spearmean_coefficient(self):
        features_rank = np.array([
                                    [1,2,3,4],
                                    [2,4,1,3],
                                    [1,3,4,2]
                                 ])
        spearman_coeffs = self.measurement.spearmean_coefficient(features_rank)  
        number_features = 4
        normalization_term = float(number_features*(number_features**2 -1))
        spearman_coeff_01 = ((features_rank[0] - features_rank[1])**2 / normalization_term).sum()
        spearman_coeff_02 = ((features_rank[0] - features_rank[2])**2 / normalization_term).sum()
        spearman_coeff_12 = ((features_rank[1] - features_rank[2])**2 / normalization_term).sum()
        spearman_coeffs_byhand = 1 - 6*np.array([[0, spearman_coeff_01, spearman_coeff_02],
                                                [spearman_coeff_01, 0, spearman_coeff_12],
                                                [spearman_coeff_02, spearman_coeff_12, 0]
                                               ])
        assert abs(np.sum(spearman_coeffs_byhand - spearman_coeffs)) < 10**-5

    def test_mean_of_lower_triangular_of_symmetric_matrix(self):
        matrix = np.array([
                             [1,2,1,],
                             [2,4,3,],
                             [1,3,4,]
                          ])
        mean = self.measurement.mean_of_lower_triangular_of_symmetric_matrix(matrix)
        mean_byhand = (2+1+3)/3.
        assert (abs(np.sum(mean_byhand - mean))) < 10**-5

    def test_jaccard_index(self):
        features_rank = np.array([
                                    [1,2,3,4,5,6,7,10,9,8],
                                    [2,4,1,3,10,5,7,9,8,6],
                                    [1,3,4,2,7,5,9,10,6,8]
                                 ])
        jaccard_indices = rob.jaccard_index(features_rank, perc=0.1)
        jaccard_indices_byhand = np.array([
                                            [1,0,1],
                                            [0,1,0],
                                            [1,0,1]
                                          ])

        assert abs(np.sum(jaccard_indices_byhand - jaccard_indices)) < 10**-5
