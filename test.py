from experiments import DataSetExperiment
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
    SVM_RFE(),
    ensemble_methods.Stacking(
        [SymmetricalUncertainty(), Relief()],
        combination="mean"
    ),
    ensemble_methods.Stacking(
        [SymmetricalUncertainty(), Relief()],
        combination="hmean"
    ),
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

experiment = DataSetExperiment(
    feature_rankings,
    robustness_measures,
    classifiers,
    working_dir=".."
)

experiment.run_data_set("dorothea")
