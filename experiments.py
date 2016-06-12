from benchmarks import RobustnessBenchmark, AccuracyBenchmark
from tabulate import tabulate
import numpy as np


class RobustnessExperiment:
    def __init__(self, robustness_measures=None, feature_rankings=None):
        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        if not isinstance(feature_rankings, list):
            feature_rankings = [feature_rankings]

        results_shape = (len(robustness_measures), len(feature_rankings))

        self.robustness_measures = robustness_measures
        self.feature_rankings = feature_rankings
        self.results = np.zeros(results_shape)

    def run(self, data, classes):
        for i in range(self.results.shape[1]):
            benchmark = RobustnessBenchmark(
                robustness_measures=self.robustness_measures,
                feature_ranking=self.feature_rankings[i]
            )
            self.results[:, i] = benchmark.run(data, classes)

        return self.results

    def print_results(self):
        print("Robustness Experiment : ")
        print_results(self.feature_rankings, self.robustness_measures, self.results)


class AccuracyExperiment:
    def __init__(self, classifiers=None, feature_rankings=None):
        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        if not isinstance(feature_rankings, list):
            feature_rankings = [feature_rankings]

        results_shape = (len(classifiers), len(feature_rankings))

        self.classifiers = classifiers
        self.feature_rankings = feature_rankings
        self.results = np.zeros(results_shape)

    def run(self, data, classes):
        for i in range(self.results.shape[1]):
            benchmark = AccuracyBenchmark(
                classifiers=self.classifiers,
                feature_ranking=self.feature_rankings[i]
            )
            self.results[:, i] = benchmark.run(data, classes)

        return self.results

    def print_results(self):
        print("Accuracy Experiment : ")
        print_results(self.feature_rankings, self.classifiers, self.results)


def f_measure(robustness, accuracy, beta=1):
    return (beta ** 2 + 1) * robustness * accuracy / (beta ** 2 * robustness + accuracy)


def print_results(feature_rankings, tests, results):
        headers = [feature_rankings[i].__name__ for i in range(results.shape[1])]
        rows = []
        for i in range(results.shape[0]):
            if hasattr(tests[i], '__name__'):
                row = [tests[i].__name__]
            else:
                row = [type(tests[i]).__name__]
            row += map(lambda i: "{:.2%}".format(i), results[i, :].tolist())
            rows.append(row)

        print(tabulate(rows, headers, tablefmt='pipe'))
