from experiments import DataSetExperiment, RawDataSetExperiment
from benchmarks import MeasureBenchmark, AccuracyBenchmark
from sklearn.neighbors import KNeighborsClassifier
from sklearn_utilities import SVC_Grid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import robustness_measure
import goodness_measure
import numpy as np
from sklearn.cross_validation import KFold
from data_sets import DataSets
import sys
from accuracy_measure import ber

default_classifiers = [
    KNeighborsClassifier(3),
    SVC_Grid(kernel="linear"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    LogisticRegressionCV(penalty='l1', solver='liblinear')
]


def full(data_sets, feature_selectors, jaccard_percentage=0.01, classifiers=None, precision_measures=False):
    if classifiers is None:
        classifiers = default_classifiers

    if isinstance(data_sets, str):
        data_sets = [data_sets]

    measures = [
        robustness_measure.Spearman(),
        robustness_measure.JaccardIndex(percentage=jaccard_percentage),
    ]

    accuracy_exp = DataSetExperiment(
        AccuracyBenchmark(classifiers, percentage_of_features=jaccard_percentage),
        feature_selectors
    )

    for data_set in data_sets:
        print("Data Set: {}".format(data_set))

        if precision_measures:
            measures += [
                goodness_measure.Precision(data_set),
                goodness_measure.XPrecision(data_set)
            ]

        robustness_exp = DataSetExperiment(
            MeasureBenchmark(measures),
            feature_selectors
        )

        robustness_exp.run(data_set)
        robustness_exp.print_results()
        robustness_exp.save_results(data_set + "_rob.csv")

        accuracy_exp.run(data_set)
        accuracy_exp.print_results()
        accuracy_exp.save_results(data_set + "_acc.csv")


def raw(data_sets, feature_selectors, jaccard_percentage=0.01, classifiers=None):
    if classifiers is None:
        classifiers = default_classifiers

    if isinstance(data_sets, str):
        data_sets = [data_sets]

    robustness_exp = RawDataSetExperiment(
        MeasureBenchmark([
            robustness_measure.JaccardIndex(percentage=jaccard_percentage),
        ]),
        feature_selectors
    )

    accuracy_exp = RawDataSetExperiment(
        AccuracyBenchmark(classifiers, percentage_of_features=jaccard_percentage),
        feature_selectors
    )

    jcp = int(jaccard_percentage * 1e3)
    robustness_exp.run(data_sets)
    robustness_exp.save_results("jc{}_robustness".format(jcp))

    accuracy_exp.run(data_sets)
    accuracy_exp.save_results("jc{}_accuracy".format(jcp))


def without_feature_selectors(data_sets, classifiers):
    if classifiers is None:
        classifiers = default_classifiers

    if isinstance(data_sets, str):
        data_sets = [data_sets]

    results = np.zeros((len(data_sets), len(classifiers), 10))

    nc = len(classifiers)
    n = nc * len(data_sets) * 10

    for i, data_set in enumerate(data_sets):
        data, labels = DataSets.load(data_set)
        for j, classifier in enumerate(classifiers):
            for k, (train_index, test_index) in enumerate(KFold(labels.shape[0], n_folds=10)):
                sys.stdout.write("\rProgress: {:.2%}".format((i * nc * 10 + j * 10 + k + 1)/n))
                classifier.fit(
                    data[:, train_index].T,
                    labels[train_index]
                )
                results[i, j, k] = ber(
                    labels[test_index],
                    classifier.predict(data[:, test_index].T)
                )

    np.save("../results/RAW/all_features", results)

    def write(name, data):
        with open("../results/RAW/{}.txt".format(name), "w") as f:
            for d in data:
                f.write(d + "\n")

    write("all_features_0", data_sets)
    write("all_features_1", [type(m).__name__ for m in classifiers])
