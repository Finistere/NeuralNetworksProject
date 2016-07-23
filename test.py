from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector, Random, FeatureSelector
import ensemble_methods
import analysis
import artificial_data
import numpy as np
from data_sets import DataSets

import warnings

warnings.filterwarnings('ignore')

# total_features = 1e4
# n_significant_features = int(total_features*0.01)
# DataSets.save_artificial(
#     *artificial_data.generate(
#         n_samples=300,
#         n_features=total_features,
#         n_significant_features=n_significant_features,
#         feature_distribution=artificial_data.multivariate_normal(
#             mean=np.zeros(n_significant_features),
#             cov=artificial_data.uniform(-1, 1)
#         ),
#         insignificant_feature_distribution=artificial_data.multiple_distribution(
#             distributions=[artificial_data.uniform(-1, 1), artificial_data.normal(0, 1)],
#             shares=[0.5, 0.5]
#         ),
#         noise_distribution=artificial_data.normal(0, 0.05),
#         labeling=artificial_data.linear_labels(weights=np.ones(n_significant_features))
#     )
# )

feature_selectors = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE(),
    LassoFeatureSelector(),
]

e_methods = [
    ensemble_methods.Mean(feature_selectors),
    ensemble_methods.SMeanWithClassifier(
        feature_selectors,
        analysis.default_classifiers,
        min_mean_max=[0, 1, 0]
    ),
]

fs = feature_selectors + e_methods

data_sets = ["artificial", "colon", "arcene", "dexter", "gisette"]
# analysis.full("artificial", fs)
analysis.full("colon", fs)
# analysis.full("arcene", fs)
# analysis.full("dexter", fs)
# analysis.full("gisette", fs)
# analysis.full("dorothea", fs)
