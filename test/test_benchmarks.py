from benchmarks import *
import numpy as np
import robustness_measure
import feature_ranking
from sklearn.dummy import DummyClassifier


class TestRobustnessBenchmark:
    benchmark = RobustnessBenchmark(
        feature_ranking=feature_ranking.Dummy(),
        robustness_measures=robustness_measure.Dummy()
    )

    def test_robustness(self):
        assert 1 == self.benchmark.run(np.ones((10, 4)), np.arange(4))


class TestAccuracyBenchmark:
    benchmark = AccuracyBenchmark(
        feature_ranking=feature_ranking.Dummy(),
        classifier=DummyClassifier(strategy='constant', constant=1)
    )

    def test_best_percent(self):
        l = np.arange(200)
        assert [199, 198] == AccuracyBenchmark.highest_1percent(l).tolist()

    def test_classification_accuracy(self):
        data = np.random.randn(200, 10)
        classes = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        expected_accuracy = 4 / 10

        assert expected_accuracy == self.benchmark.run(data, classes, n_folds=10)



