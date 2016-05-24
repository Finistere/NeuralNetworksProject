from benchmarks import *
import numpy as np
import robustness_measure
import feature_ranking


class TestBenchmark:
    benchmark = Benchmark(robustness_measure.Dummy(), feature_ranking.Dummy())

    def test_best_percent(self):
        l = np.arange(200)
        assert [199, 198] == Benchmark.highest_1percent(l).tolist()

    def test_robustness(self):
        assert 1 == self.benchmark.robustness(np.ones((10, 4)), np.arange(4))

    def test_classification_accuracy(self):
        data = np.random.randn(200, 10)
        classes = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])

        expected_accuracy = (3 / 5 + 1 / 5) / 2

        assert expected_accuracy == self.benchmark.classification_accuracy(data, classes, n_folds=2)


