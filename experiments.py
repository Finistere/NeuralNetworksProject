from benchmarks import RobustnessBenchmark
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
        headers = [type(self.feature_rankings[i]).__name__ for i in range(self.results.shape[1])]
        rows = []
        for i in range(self.results.shape[0]):
            row = [self.robustness_measures[i].__name__]
            row += map(lambda i: "{:.2%}".format(i), self.results[i, :].tolist())
            rows.append(row)

        print(tabulate(rows, headers, tablefmt='pipe'))