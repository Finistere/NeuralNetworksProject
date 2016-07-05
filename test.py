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

em_feature_selectors = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE(),
    LassoFeatureSelector(),
]

feature_selectors = em_feature_selectors + [
    Random()
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, gamma=0, coef0=2, C=0.1),
    SVC(kernel="poly", degree=1, gamma=0, coef0=1, C=0.5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
]

import warnings

warnings.filterwarnings('ignore')

total_features = 1e4
n_significant_features = int(total_features*0.01)
DataSets.save_artificial(
    *ArtificialData.generate(
        n_samples=300,
        n_features=total_features,
        n_significant_features=n_significant_features,
        feature_distribution=ArtificialData.multivariate_normal(
            mean=np.zeros(n_significant_features),
            cov=ArtificialData.uniform(-2, 2)
        ),
        insignificant_feature_distribution=ArtificialData.multiple_distribution(
            distributions=[ArtificialData.uniform(-1, 1), ArtificialData.normal(0, 1)],
            shares=[0.5, 0.5]
        ),
        noise_distribution=ArtificialData.normal(0, 0.05),
        labeling=ArtificialLabels.linear(weights=np.ones(n_significant_features))
    )
)

# e = []
# for i in range(1, 10):
#     for j in range(1, 10):
#         for k in range(1, 10):
#             e.append(ensemble_methods.SMean(feature_selectors, min_mean_max=[i, j, k]))


e_methods = [
    ensemble_methods.Mean(em_feature_selectors),
    ensemble_methods.SMean(em_feature_selectors, min_mean_max=[9, 3, 1]),
    ensemble_methods.SMeanWithClassifier(
        em_feature_selectors,
        classifiers,
        min_mean_max=[9, 3, 1]
    ),
]

# exp = EnsembleFMeasureExperiment(
#     classifiers,
#     e_methods,
#     feature_selectors=feature_selectors,
#     beta=2,
#     jaccard_percentage=0.05
# )
# exp.run(["colon", "arcene", "dexter"])
# exp.print_results()


robustnessf_exp = EnsembleFMeasureExperiment(
    classifiers,
    e_methods,
    feature_selectors=feature_selectors
)


accuracy_exp = EnsembleMethodExperiment(
    e_methods,
    AccuracyBenchmark(classifiers, percentage_of_features=.01),
    feature_selectors
)


def do_complete_analysis(data_set, append_results=False):
    print(data_set)
    measures = [
        robustness_measure.Spearman(),
        robustness_measure.JaccardIndex(percentage=0.01),
        robustness_measure.JaccardIndex(percentage=0.05),
        goodness_measure.Precision(data_set),
        goodness_measure.XPrecision(data_set)
    ]
    robustness_exp = EnsembleMethodExperiment(
        e_methods,
        MeasureBenchmark(measures),
        feature_selectors
    )

    robustness_exp.run(data_set)
    robustness_exp.print_results()
    robustness_exp.save_results(data_set + "_rob.csv", append_results)
    accuracy_exp.run(data_set)
    accuracy_exp.print_results()
    accuracy_exp.save_results(data_set + "_acc.csv", append_results)
    robustnessf_exp.run([data_set])
    robustnessf_exp.print_results()
    robustnessf_exp.save_results(data_set + ".csv", append_results)
    return

do_complete_analysis("gisette")
do_complete_analysis("artificial", True)
# do_complete_analysis("arcene")
# do_complete_analysis("dexter")

# jc = robustness_measure.JaccardIndex(0.01)
#
# print(jc.measure(np.array([
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
#     FeatureSelector.rank_weights(np.random.uniform(0, 1, 1e4)),
# ]).T))