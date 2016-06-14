from benchmarks import RobustnessBenchmark, AccuracyBenchmark, Benchmark
from tabulate import tabulate
import numpy as np
import csv
from data_sets import DataSets
import os
import errno


class Experiment:
    results = np.array([])
    row_labels = []
    col_labels = []

    def __generate_results_table(self):
        rows = [
            ["Measure"] + self.col_labels
        ]
        for i in range(self.results.shape[0]):
            row = [self.row_labels[i]]
            row += map(lambda i: "{:.2%}".format(i), self.results[i, :].tolist())
            rows.append(row)

        return rows

    def print_results(self):
        rows = self.__generate_results_table()
        print(tabulate(rows[1:len(rows)], rows[0], tablefmt='pipe'))

    def save_results(self, file_name="output.csv"):
        root_dir = DataSets.root_dir + "/results/" + type(self).__name__

        try:
            os.makedirs(root_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        with open(root_dir + "/" + file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.__generate_results_table())


class RobustnessExperiment(Experiment):
    def __init__(self, feature_rankings, robustness_measures):
        if not isinstance(robustness_measures, list):
            robustness_measures = [robustness_measures]

        if not isinstance(feature_rankings, list):
            feature_rankings = [feature_rankings]

        self.robustness_measures = robustness_measures
        self.feature_rankings = feature_rankings
        self.results = np.zeros((len(robustness_measures), len(feature_rankings)))

        self.row_labels = [type(r).__name__ for r in self.robustness_measures]
        self.col_labels = [f.__name__ for f in self.feature_rankings]

    def run(self, data, labels):
        for i in range(self.results.shape[1]):
            benchmark = RobustnessBenchmark(
                robustness_measures=self.robustness_measures,
                feature_ranking=self.feature_rankings[i]
            )
            self.results[:, i] = benchmark.run(data, labels)

        return self.results

    def print_results(self):
        print("Robustness Experiment : ")
        super().print_results()


class AccuracyExperiment(Experiment):
    measure_name = "classifiers"

    def __init__(self, feature_rankings, classifiers):
        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        if not isinstance(feature_rankings, list):
            feature_rankings = [feature_rankings]

        results_shape = (len(classifiers), len(feature_rankings))

        self.classifiers = classifiers
        self.feature_rankings = feature_rankings
        self.results = np.zeros(results_shape)
        self.row_labels = [type(c).__name__ for c in self.classifiers]
        self.col_labels = [f.__name__ for f in self.feature_rankings]

    def run(self, data, labels):
        for i in range(self.results.shape[1]):
            benchmark = AccuracyBenchmark(
                classifiers=self.classifiers,
                feature_ranking=self.feature_rankings[i]
            )
            self.results[:, i] = benchmark.run(data, labels)

        return self.results

    def print_results(self):
        print("Accuracy Experiment : ")
        super().print_results()


class DataSetExperiments:
    def __init__(self, feature_rankings, robustness_measures, classifiers, working_dir=".."):
        self.robustness_experiment = RobustnessExperiment(feature_rankings, robustness_measures)
        self.accuracy_experiment = AccuracyExperiment(feature_rankings, classifiers)
        self.results_folder = working_dir

    def run_data_set(self, name, file_name="output.csv"):
        data, labels = DataSets.load(name)

        self.robustness_experiment.run(data, labels)
        self.robustness_experiment.print_results()
        self.robustness_experiment.save_results(file_name)

        self.accuracy_experiment.run(data, labels)
        self.accuracy_experiment.print_results()
        self.accuracy_experiment.save_results(file_name)


class EnsembleMethodExperiment(Experiment):
    def __init__(self, ensemble_methods, benchmark: Benchmark):
        if not isinstance(ensemble_methods, list):
            ensemble_methods = [ensemble_methods]

        self.ensemble_methods = ensemble_methods
        self.benchmark = benchmark
        self.results = np.zeros((len(benchmark.get_measures()), len(ensemble_methods)))

        self.row_labels = [m.__name__ for m in self.benchmark.get_measures()]
        self.col_labels = [f.__name__ for f in ensemble_methods]

    def run(self, data_set):

        data, labels = DataSets.load(data_set)
        for i, ensemble_method in enumerate(self.ensemble_methods):
            self.results[:, i] = self.benchmark.run(
                data,
                labels,
                ensemble_method.ranks(data_set, self.benchmark)
            )
        return self.results

    def print_results(self):
        print("Ensemble Method: ")
        super().print_results()

