from experiments import RobustnessExperiment, AccuracyExperiment
from feature_ranking import SymmetricalUncertainty, Relief, SVM_RFE
import robustness_measure
import ensemble_methods
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# data
import sklearn.datasets
mnist = sklearn.datasets.load_digits()
data = np.loadtxt("../DOROTHEA/DOROTHEA/dorothea_train.data")
classes = np.loadtxt("../DOROTHEA/DOROTHEA/dorothea_train.labels")

feature_rankings = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE(),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="mean"),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="hmean"),
]
robustness_measures = [
    robustness_measure.Spearman(),
    robustness_measure.JaccardIndex(percentage=0.01),
    robustness_measure.JaccardIndex(percentage=0.05),
]
classifiers = [KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

import warnings
warnings.filterwarnings('ignore')

robustness_experiment = RobustnessExperiment(robustness_measures, feature_rankings)
robustness_experiment.run(data, classes)
robustness_experiment.print_results()

# accuracy_exp = AccuracyExperiment(classifiers, feature_rankings)
# accuracy_exp.run(data, classes)
# accuracy_exp.print_results()

