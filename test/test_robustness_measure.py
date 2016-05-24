from robustness_measure import *
import numpy as np


class TestMeanOfLowerTriangle:
    def test(self):
        matrix = np.array([
            [1, 2, 1, ],
            [6, 4, 3, ],
            [1, 3, 4, ]
        ])
        expected_mean = (6 + 1 + 3) / 3.

        assert abs(expected_mean - mean_of_lower_triangular(matrix)) < 10 ** -5


class TestSpearman:
    def test_spearman(self):
        features_rank = np.array([
            [1, 2, 3, 4],
            [2, 4, 1, 3],
            [1, 3, 4, 2]
        ])

        n = features_rank.shape[1]
        normalization_term = float(n * (n ** 2 - 1))
        s01 = ((features_rank[0] - features_rank[1]) ** 2 / normalization_term).sum()
        s02 = ((features_rank[0] - features_rank[2]) ** 2 / normalization_term).sum()
        s12 = ((features_rank[1] - features_rank[2]) ** 2 / normalization_term).sum()

        expected_spearman = np.mean(1 - 6 * np.array([s01, s02, s12]))

        spearman = Spearman()

        assert abs(expected_spearman - spearman.measure(features_rank.T)) < 10 ** -5


class TestJaccardIndex:
    def test_matrix(self):
        jaccard = JaccardIndex(percentage=0.3)
        features_ranks = np.array([
            [1, 2, 3, 4, 5, 6, 7, 10, 9, 8],
            [2, 4, 3, 1, 10, 5, 7, 9, 8, 6],
            [1, 3, 4, 2, 7, 5, 9, 10, 6, 8]
        ])

        expected_matrix = [
            [1, 1 / 2, 1 / 2],
            [1 / 2, 1, 1 / 5],
            [1 / 2, 1 / 5, 1]
        ]

        assert expected_matrix == jaccard.matrix(features_ranks.T).tolist()

    def test_measure(self):
        jaccard = JaccardIndex(percentage=0.3)
        features_ranks = np.array([
            [1, 2, 3, 4, 5, 6, 7, 10, 9, 8],
            [2, 4, 3, 1, 10, 5, 7, 9, 8, 6],
            [1, 3, 4, 2, 7, 5, 9, 10, 6, 8]
        ])

        expected = (1 / 2 + 1 / 2 + 1 / 5) / 3

        assert abs(expected - jaccard.measure(features_ranks.T)) < 10 ** -5
