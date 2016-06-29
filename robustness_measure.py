from abc import ABCMeta, abstractmethod
import scipy.stats
import numpy as np


class Measure(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

    def run_and_set_in_results(self, features_selection, results, result_index):
        results[result_index] = self.measure(features_selection)

    @abstractmethod
    # features ranks is matrix with each rows represent a feature, and the columns its rankings
    def measure(self, features_ranks):
        pass


class Dummy(Measure):
    def measure(self, features_ranks):
        return 1


def mean_of_lower_triangular(matrix):
    lower_triangular_index = np.tril_indices(matrix.shape[0], -1)
    return np.mean(matrix[lower_triangular_index])


class Spearman(Measure):

    def __init__(self):
        super().__init__()
        self.__name__ = "Spearman Coefficient"

    def measure(self, features_ranks):
        spearman_correlation_matrix, _ = scipy.stats.spearmanr(features_ranks)
        return mean_of_lower_triangular(spearman_correlation_matrix)


class JaccardIndex(Measure):

    def __init__(self, percentage=0.1):
        super().__init__()
        self.percentage = percentage
        self.__name__ = "Jaccard Index {:.2%}".format(percentage)

    def measure(self, features_ranks):
        indices = self.matrix(features_ranks)
        return mean_of_lower_triangular(indices)

    def matrix(self, features_ranks):
        if np.any(np.min(features_ranks, axis=0) != np.ones(features_ranks.shape[1], dtype=np.int)):
            print(features_ranks)
            raise ValueError('features_rank ranking does not always begin with a 1')

        # the minimal rank a feature mast have to be chose
        minimal_rank = int((1 - self.percentage) * features_ranks.shape[0]) + 1

        # set everything below the minimal rank to zero and everything else to 1
        features_ranks[features_ranks < minimal_rank] = 0
        features_ranks[0 != features_ranks] = 1

        k = features_ranks.shape[1]
        jaccard_indices = np.identity(k)

        # jaccard_indices is symmetric
        for i in range(1, k):
            for j in range(i):
                f_i = features_ranks[:, i] == 1
                f_j = features_ranks[:, j] == 1

                intersection = np.logical_and(f_i, f_j)
                union = np.logical_or(f_i, f_j)

                jaccard_indices[i, j] = np.sum(intersection) / np.sum(union)
                jaccard_indices[j, i] = jaccard_indices[i, j]

        return jaccard_indices
