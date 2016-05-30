from benchmarks import RobustnessExperiment
from feature_ranking import SymmetricalUncertainty  
from feature_ranking import Relief
from feature_ranking import SVM_RFE
from robustness_measure import Spearman 
from robustness_measure import JaccardIndex
# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# data
import sklearn.datasets
mnist = sklearn.datasets.load_digits()
data = mnist.data[:500].T
classes = mnist.target

feature_rankings = [SymmetricalUncertainty(),
                    Relief(),
                    SVM_RFE()]
robustness_measures = [Spearman(),
                       JaccardIndex()]
classifiers = [KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

import warnings
warnings.filterwarnings('ignore')

robustness_experiment = RobustnessExperiment(robustness_measures, feature_rankings)
robustness_experiment.run(mnist.data.T, mnist.target)
robustness_experiment.print_results()

