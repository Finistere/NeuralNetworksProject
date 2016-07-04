from experiments import *
from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector, Random
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
    SVM_RFE(),
    LassoFeatureSelector(),
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
n_significant_features = total_features*0.01
DataSets.save_artificial(
    *ArtificialData.generate(
        n_samples=300,
        n_features=total_features,
        n_significant_features=n_significant_features,
        feature_distribution=ArtificialData.multivariate_normal(
            mean=np.zeros(n_significant_features),
            cov=ArtificialData.uniform(-1, 1)
        ),
        insignificant_feature_distribution=ArtificialData.multiple_distribution(
            distributions=[ArtificialData.uniform(0, 1), ArtificialData.uniform(1, 2)],
            shares=[0.5, 0.5]
        ),
        noise_distribution=None,
        labeling=ArtificialLabels.linear(weights=np.ones(n_significant_features))
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


def do_complete_analysis(dataset):
    measures = [
        robustness_measure.Spearman(),
        robustness_measure.JaccardIndex(percentage=0.01),
        robustness_measure.JaccardIndex(percentage=0.05),
        goodness_measure.Precision(dataset),
        goodness_measure.XPrecision(dataset)
    ]
    robustness_exp = EnsembleMethodExperiment(
        e_methods,
        MeasureBenchmark(measures),
        feature_selectors
    )

    robustness_exp.run(dataset)
    robustness_exp.print_results()
    robustness_exp.save_results(dataset+"_rob.csv")
    accuracy_exp.run(dataset)
    accuracy_exp.print_results()
    accuracy_exp.save_results(dataset+"_acc.csv")
    robustnessf_exp.run([dataset])
    robustnessf_exp.print_results()
    robustnessf_exp.save_results(dataset+".csv")
    return

do_complete_analysis("artificial")