from experiments import *
import numpy as np
import robustness_measure
import feature_ranking
from sklearn.dummy import DummyClassifier


class TestRobustnessExperiment:
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


class TestAccuracyExperiment:
    experiment = AccuracyExperiment(
        DummyClassifier(strategy='constant', constant=1),
        [feature_ranking.Dummy(), feature_ranking.Dummy()]
    )

    def test_run(self):
        data = np.random.randn(200, 10)
        classes = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        expected_accuracy = [
            [4 / 10, 4 / 10],

        ]

        assert expected_accuracy == self.experiment.run(data, classes).tolist()