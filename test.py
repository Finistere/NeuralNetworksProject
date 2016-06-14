from experiments import *
from feature_ranking import SymmetricalUncertainty, Relief, SVM_RFE
import robustness_measure
import ensemble_methods
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

feature_rankings = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE()
]
robustness_measures = [
    robustness_measure.Spearman(),
    robustness_measure.JaccardIndex(percentage=0.01),
    robustness_measure.JaccardIndex(percentage=0.05),
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
]

import warnings

warnings.filterwarnings('ignore')

exp = EnsembleMethodExperiment(
    [ensemble_methods.Mean(feature_rankings)],
    RobustnessBenchmark(robustness_measures)
)
exp.run("dorothea")
exp.print_results()
