import unittest
import filters
import math


class TestSUFilter():
    # data set for integer valued entropy
    data_set = filters.SUFilter(
        [
            [1, 2, 3],
            [0, 0, 0],
            [1, 0, 1]
        ],
        [1, 0, 1]
    )

    def test_class_distribution(self):
        assert [1./3, 2./3] == self.data_set.class_distribution().tolist()

    def test_class_entropy(self):
        assert abs(-1./3 * math.log(1./3) - 2./3 * math.log(2./3) - self.data_set.class_entropy()) < 10 ** -5

    def test_feature_distribution(self):
        assert [1./3, 1./3, 1./3] == self.data_set.feature_distribution(0).tolist()
        assert [1.] == self.data_set.feature_distribution(1).tolist()
        assert [1./3, 2./3] == self.data_set.feature_distribution(2).tolist()

    def test_feature_entropy(self):
        assert abs(-3. * 1. / 3 * math.log(1./3) - self.data_set.feature_entropy(0)) < 10 ** -5
        assert abs(self.data_set.feature_entropy(1)) < 10 ** -5
        assert abs(- 1. / 3 * math.log(1./3) - 2. / 3 * math.log(2./3) - self.data_set.feature_entropy(2)) < 10 ** -5

    def test_feature_class_joint_distribution(self):
        assert self.data_set.feature_class_joint_distribution(0).tolist() == [
            [0, 1./3],
            [1./3, 0],
            [0, 1./3]
        ]
        assert self.data_set.feature_class_joint_distribution(1).tolist() == [
            [1./3, 2./3],
        ]
        assert self.data_set.feature_class_joint_distribution(2).tolist() == [
            [1./3, 0],
            [0, 2./3],
        ]

    def test_feature_class_entropy(self):
        assert abs(
            2 * 1./3 * math.log((2./3) / (1./3)) + 1./3 * math.log((1./3) / (1./3)) -
            self.data_set.feature_class_conditional_entropy(0)
        ) < 10 ** -5

        assert abs(
            1./3 * math.log((1./3) / (1./3)) + 2./3 * math.log((2./3) / (2./3)) -
            self.data_set.feature_class_conditional_entropy(1)
        ) < 10 ** -5

        assert abs(
            1./3 * math.log((1./3) / (1./3)) + 2./3 * math.log((2./3) / (2./3)) -
            self.data_set.feature_class_conditional_entropy(2)
        ) < 10 ** -5

    def test_symmetrical_uncertainty(self):
        h_f = -3 * 1. / 3 * math.log(1./3)
        h_c = -1./3 * math.log(1./3) - 2./3 * math.log(2./3)
        h_fc = 2 * 1./3 * math.log((2./3) / (1./3)) + 1./3 * math.log((1./3) / (1./3))
        su = 2 * (h_f - h_fc) / (h_f + h_c)
        assert abs(su - self.data_set.symmetrical_uncertainty(0)) < 10 ** -5

