from benchmarks import *
import numpy as np
import robustness_measure
import feature_ranking
from sklearn.dummy import DummyClassifier


class TestBenchmark:
    benchmark = Benchmark(
        feature_ranking=feature_ranking.Dummy(),
        robustness_measures=robustness_measure.Dummy(),
        classifier=DummyClassifier(strategy='constant', constant=1)
    )

    def test_best_percent(self):
        l = np.arange(200)
        assert [199, 198] == Benchmark.highest_1percent(l).tolist()

    def test_robustness(self):
        assert 1 == self.benchmark.robustness(np.ones((10, 4)), np.arange(4))

    def test_classification_accuracy(self):
        data = np.random.randn(200, 10)
        classes = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        expected_accuracy = 4 / 10

        assert expected_accuracy == self.benchmark.classification_accuracy(data, classes, n_folds=10)


class TestExperiment:
    experiment = RobustnessExperiment(
        robustness_measure.Dummy(),
        [feature_ranking.Dummy(), feature_ranking.Dummy()]
    )

    def test_run(self):
        data = np.random.randn(200, 10)
        classes = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        expected_results = [
            [1, 1]
        ]

        assert expected_results == self.experiment.run(data, classes).tolist()

    def test_print_results(self):
        experiment = RobustnessExperiment(
            robustness_measure.Dummy(),
            [feature_ranking.Dummy(), feature_ranking.Dummy()]
        )
        experiment.results= np.array([[0.89,0.1]])
        experiment.print_results()


