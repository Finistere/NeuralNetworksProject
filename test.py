from experiments import RobustnessExperiment, AccuracyExperiment
from feature_ranking import SymmetricalUncertainty  
from feature_ranking import Relief
from feature_ranking import SVM_RFE
from robustness_measure import Spearman 
from robustness_measure import JaccardIndex
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time

# data
import sklearn.datasets
mnist = sklearn.datasets.load_digits()
data = mnist.data.T[:,:1000]
classes = mnist.target[:1000]

feature_rankings = [SymmetricalUncertainty(), Relief(), SVM_RFE()]
robustness_measures = [Spearman(),
                       JaccardIndex()]
classifiers = [KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

import warnings
warnings.filterwarnings('ignore')

# robustness_experiment = RobustnessExperiment(robustness_measures, feature_rankings)
# start = time.time()
# robustness_experiment.run(data, classes)
# end = time.time()
# print("Time:", end - start)
# robustness_experiment.print_results()

accuracy_exp = AccuracyExperiment(classifiers, feature_rankings)
start = time.time()
accuracy_exp.run(data, classes)
end = time.time()
print("Time:", end - start)
accuracy_exp.print_results()

