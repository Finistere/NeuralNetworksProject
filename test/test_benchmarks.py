from benchmarks import *
import numpy as np
import robustness_measure
import feature_selector
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import KFold


class TestFeatureRanksGenerator:
    featureRankGenerator = FeatureSelectionGenerator(feature_selector.Dummy())

    def test_generate(self):
        data, labels = np.ones((10, 10)), np.arange(10)
        expected_results = np.array([
            np.arange(10),
            np.arange(10),
        ]).tolist()
        assert expected_results == self.featureRankGenerator.generate(data, labels, KFold(10, n_folds=2)).tolist()


class TestRobustnessBenchmark:
    benchmark = RobustnessBenchmark(
        feature_selector=feature_selector.Dummy(),
        robustness_measures=robustness_measure.Dummy()
    )

    def test_robustness(self):
        assert 1 == self.benchmark.run(np.ones((10, 4)), np.arange(4))


class TestClassifierWrapper:
    classifier = ClassifierWrapper(DummyClassifier(strategy='constant', constant=1))

    def test_run(self):
        data = np.random.randn(200, 10)
        labels = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        train_index = [range(7)]
        test_index = [7, 8, 9]

        results = np.zeros((2, 2))
        result_index = (0, 1)

        expected_result = [
            [0, 1/3],
            [0, 0]
        ]

        self.classifier.run_and_set_in_results(
            data,
            labels,
            train_index,
            test_index,
            results,
            result_index
        )

        assert expected_result == results.tolist()


class TestAccuracyBenchmark:
    benchmark = AccuracyBenchmark(
        feature_selector=feature_selector.Dummy(),
        classifiers=[
            DummyClassifier(strategy='constant', constant=1),
            DummyClassifier(strategy='constant', constant=2),
        ]
    )

    def test_best_percent(self):
        l = np.arange(200)
        assert [199, 198] == AccuracyBenchmark.highest_percent(l, 0.01).tolist()

    def test_classification_accuracy(self):
        data = np.random.randn(200, 10)
        labels = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        expected_accuracy = [4 / 10, 2 / 10]

        assert expected_accuracy == self.benchmark.run(data, labels).tolist()



