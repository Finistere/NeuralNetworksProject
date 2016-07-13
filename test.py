from experiments import *
from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector, Random, FeatureSelector
import robustness_measure
import goodness_measure
import ensemble_methods
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from artificial_data_sets2 import ArtificialData, ArtificialLabels
from sklearn.grid_search import GridSearchCV

feature_selectors = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE(),
    LassoFeatureSelector(),
]


class SVC_Grid:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.classifier = None

    def fit(self, data, labels):
        grid_search = GridSearchCV(
            SVC(),
            {
                "C": [1, 10, 100, 1000]
            },
            cv=5,
            scoring='precision'
        )
        grid_search.fit(data, labels)
        self.kwargs["C"] = grid_search.best_params_["C"]

        self.classifier = SVC(*self.args, **self.kwargs)

        self.classifier.fit(data, labels)

    def predict(self, data):
        return self.classifier.predict(data)

    def score(self, data, labels):
        return self.classifier.score(data, labels)


classifiers = [
    KNeighborsClassifier(3),
    SVC_Grid(kernel="linear"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
]

import warnings

warnings.filterwarnings('ignore')

# total_features = 1e4
# n_significant_features = int(total_features*0.01)
# DataSets.save_artificial(
#     *ArtificialData.generate(
#         n_samples=300,
#         n_features=total_features,
#         n_significant_features=n_significant_features,
#         feature_distribution=ArtificialData.multivariate_normal(
#             mean=np.zeros(n_significant_features),
#             cov=ArtificialData.uniform(-2, 2)
#         ),
#         insignificant_feature_distribution=ArtificialData.multiple_distribution(
#             distributions=[ArtificialData.uniform(-1, 1), ArtificialData.normal(0, 1)],
#             shares=[0.5, 0.5]
#         ),
#         noise_distribution=ArtificialData.normal(0, 0.05),
#         labeling=ArtificialLabels.linear(weights=np.ones(n_significant_features))
#     )
# )

e_methods = [
    ensemble_methods.Mean(feature_selectors)
]
for i in range(3):
    for j in range(3):
        for k in range(3):
            e_methods.append(ensemble_methods.SMean(feature_selectors, min_mean_max=[i * 3 + 1, j * 3 + 1, k * 3 + 1]))


# e_methods = [
#     ensemble_methods.Mean(feature_selectors),
#     ensemble_methods.SMean(feature_selectors, min_mean_max=[9, 3, 1]),
#     ensemble_methods.SMeanWithClassifier(
#         feature_selectors,
#         classifiers,
#         min_mean_max=[9, 3, 1]
#     ),
# ]

# exp = EnsembleFMeasureExperiment(
#     classifiers,
#     e_methods,
#     feature_selectors=feature_selectors,
#     beta=2,
#     jaccard_percentage=0.05
# )
# exp.run(["colon", "arcene", "dexter"])
# exp.print_results()


fmeasure_exp = EnsembleFMeasureExperiment(
    classifiers,
    feature_selectors + e_methods,
)


accuracy_exp = DataSetExperiment(
    AccuracyBenchmark(classifiers, percentage_of_features=.01),
    feature_selectors + e_methods
)


def do_complete_analysis(data_set, append_results=False):
    print(data_set.upper())
    measures = [
        robustness_measure.Spearman(),
        robustness_measure.JaccardIndex(percentage=0.01),
        robustness_measure.JaccardIndex(percentage=0.05),
        # goodness_measure.Precision(data_set),
        # goodness_measure.XPrecision(data_set)
    ]
    robustness_exp = DataSetExperiment(
        MeasureBenchmark(measures),
        feature_selectors + e_methods
    )

    robustness_exp.run(data_set)
    robustness_exp.print_results()
    robustness_exp.save_results(data_set + "_rob.csv", append_results)
    accuracy_exp.run(data_set)
    accuracy_exp.print_results()
    accuracy_exp.save_results(data_set + "_acc.csv", append_results)
    fmeasure_exp.run([data_set])
    fmeasure_exp.print_results()
    fmeasure_exp.save_results(data_set + ".csv", append_results)
    return

# data_sets = ["artificial"]
# fs = feature_selectors + e_methods
# measures = [
#     robustness_measure.JaccardIndex(percentage=0.01),
# ]
#
# exp = RawDataSetExperiment(MeasureBenchmark(measures), fs)
#
# results = exp.run(data_sets)
# print(results.shape)
# exp.save_results("robustness")
# np.save("../results/RAW/accuracy.npy", results)
#
#
# def write(name, data):
#     with open("../results/RAW/{}.txt".format(name), "w") as f:
#         for d in data:
#             f.write(d + "\n")
#
# write("accuracy_0", data_sets)
# write("accuracy_1", [f.__name__ for f in fs])
# write("accuracy_2", [type(m).__name__ for m in measures])

# do_complete_analysis("artificial", True)
# do_complete_analysis("colon")
# do_complete_analysis("arcene")
# do_complete_analysis("dexter")
# do_complete_analysis("gisette")
# do_complete_analysis("dorothea")

fmeasure_exp.run(["colon", "arcene", "dexter", "gisette"])
fmeasure_exp.print_results()

# jc = robustness_measure.JaccardIndex(0.01)
#
# print(jc.measure(np.array([
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
# ]).T))