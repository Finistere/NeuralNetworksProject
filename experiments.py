from benchmarks import RobustnessBenchmark, AccuracyBenchmark
from tabulate import tabulate
import numpy as np
import csv
from data_sets import DataSets


class Experiment:
    feature_rankings = []
    results = np.array([])
    measure_name = ""

    def __generate_results_table(self):
        rows = [
            ["Measure"] + [self.feature_rankings[i].__name__ for i in range(self.results.shape[1])]
        ]
        measures = getattr(self, self.measure_name)
        for i in range(self.results.shape[0]):
            if hasattr(measures[i], '__name__'):
                row = [measures[i].__name__]
            else:
                row = [type(measures[i]).__name__]
            row += map(lambda i: "{:.2%}".format(i), self.results[i, :].tolist())
            rows.append(row)

        return rows

    def print_results(self):
        rows = self.__generate_results_table()
        print(tabulate(rows[1:len(rows)], rows[0], tablefmt='pipe'))

    def save_results(self, path):
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.__generate_results_table())


class RobustnessExperiment(Experiment):
    measure_name = "robustness_measures"

    def __init__(self, feature_rankings=None, robustness_measures=None):
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
        super().print_results()


class AccuracyExperiment(Experiment):
    measure_name = "classifiers"

    def __init__(self, feature_rankings=None, classifiers=None):
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
        super().print_results()


class DataSetExperiment:
    def __init__(self, feature_rankings, robustness_measures, classifiers, working_dir=".."):
        self.robustness_experiment = RobustnessExperiment(feature_rankings, robustness_measures)
        self.accuracy_experiment = AccuracyExperiment(feature_rankings, classifiers)
        self.results_folder = working_dir
        self.data_sets = DataSets(working_dir)

    def run_data_set(self, name, csv_prefix=""):
        prefix = self.results_folder + "/" + name + "_" + csv_prefix + "_"
        data, classes = self.data_sets.load(name)

        self.robustness_experiment.run(data, classes)
        self.robustness_experiment.print_results()
        self.robustness_experiment.save_results(prefix + "robustness.csv")

        self.accuracy_experiment.run(data, classes)
        self.accuracy_experiment.print_results()
        self.accuracy_experiment.save_results(prefix + "accuracy.csv")
