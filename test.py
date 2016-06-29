from experiments import *
from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE
import robustness_measure
import goodness_measure
import ensemble_methods
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from artificial_data_sets2 import ArtificialData, ArtificialLabels

feature_selectors = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE()
]
measures = [
    robustness_measure.Spearman(),
    robustness_measure.JaccardIndex(percentage=0.01),
    robustness_measure.JaccardIndex(percentage=0.05),
    # goodness_measure.Precision(100)
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
]

import warnings

warnings.filterwarnings('ignore')

DataSets.save_artificial(
    *ArtificialData.generate(
        n_samples=300,
        n_features=1e4,
        n_significant_features=100,
        feature_distribution=ArtificialData.multivariate_normal(
            mean=ArtificialData.uniform(0, 1),
            cov=ArtificialData.uniform(0, 2)
        ),
        insignificant_feature_distribution=ArtificialData.multiple_distribution(
            distributions=[ArtificialData.uniform(0, 1), ArtificialData.uniform(1, 2)],
            shares=[0.5, 0.5]
        ),
        noise_distribution=None,
        labeling=ArtificialLabels.linear_power()
    )
)

# e = []
# for i in range(1, 10):
#     for j in range(1, 10):
#         for k in range(1, 10):
#             e.append(ensemble_methods.SMean(feature_selectors, min_mean_max=[i, j, k]))


e_methods = [
    ensemble_methods.Mean(feature_selectors),
    ensemble_methods.SMean(feature_selectors, min_mean_max=[9, 3, 1]),
    ensemble_methods.SMeanWithClassifier(
        feature_selectors,
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

robustness_exp = EnsembleMethodExperiment(
    e_methods,
    MeasureBenchmark(measures),
    feature_selectors
)
robustness_exp.run("dexter")
robustness_exp.print_results()

accuracy_exp = EnsembleMethodExperiment(
    e_methods,
    AccuracyBenchmark(classifiers),
    feature_selectors
)
accuracy_exp.run("dexter")
accuracy_exp.print_results()

