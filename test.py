import experiments
from feature_ranking import SymmetricalUncertainty, Relief, SVM_RFE
import robustness_measure
import ensemble_methods
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data = np.loadtxt("../ARCENE/ARCENE/arcene_train.data")
classes = np.loadtxt("../ARCENE/ARCENE/arcene_train.labels")

feature_rankings = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE(),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="mean", p=2),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="mean", p=1.9),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="mean", p=1.8),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="mean", p=1.7),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="mean", p=1.5),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="hmean", p=1.5),
    ensemble_methods.Stacking([SymmetricalUncertainty(), Relief()], combination="hmean"),
]
robustness_measures = [
    robustness_measure.Spearman(),
    robustness_measure.JaccardIndex(percentage=0.02),
    robustness_measure.JaccardIndex(percentage=0.05),
]
classifiers = [KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

import warnings
warnings.filterwarnings('ignore')

robustness_experiment = experiments.RobustnessExperiment(robustness_measures, feature_rankings)
robustness_experiment.run(data, classes)
robustness_experiment.print_results()

accuracy_exp = experiments.AccuracyExperiment(classifiers, feature_rankings)
accuracy_exp.run(data, classes)
accuracy_exp.print_results()

beta = 1
print("F Measure with beta={} :".format(beta))
experiments.print_results(
    feature_rankings,
    classifiers,
    experiments.f_measure(robustness_experiment.results[0, :], accuracy_exp.results, beta=beta)
)
experiments.print_results(
    feature_rankings,
    classifiers,
    experiments.f_measure(robustness_experiment.results[1, :], accuracy_exp.results, beta=beta)
)
experiments.print_results(
    feature_rankings,
    classifiers,
    experiments.f_measure(robustness_experiment.results[2, :], accuracy_exp.results, beta=beta)
)



