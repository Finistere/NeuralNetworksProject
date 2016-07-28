from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector, Random, FeatureSelector
import ensemble_methods
import analysis
import artificial_data
import numpy as np
import matplotlib.pyplot as plt
from data_sets import DataSets

import warnings

warnings.filterwarnings('ignore')

total_features = 1e3
n_significant_features = 100
DataSets.save_artificial(
    *artificial_data.generate(
        n_samples=300,
        n_features=total_features,
        n_significant_features=n_significant_features,
        feature_distribution=artificial_data.multiple_distribution(
            distributions=[
                artificial_data.multivariate_normal(
                    mean=artificial_data.constant(0),
                    cov=artificial_data.uniform(0, 1)
                ),
                artificial_data.normal(0, 1)
            ],
            shares=[0.5, 0.5]
        ),
        insignificant_feature_distribution=artificial_data.multiple_distribution(
            distributions=[
                artificial_data.multivariate_normal(
                    mean=artificial_data.constant(0),
                    cov=artificial_data.uniform(0, 1)
                ),
                artificial_data.normal(0, 1)
            ],
            shares=[0.5, 0.5]
        ),
        labeling=artificial_data.linear_labeling(weights=np.ones(n_significant_features))
    )
)

data, _ = DataSets.load("artificial")
cov = np.cov(data[:200])
plt.imshow(cov, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
plt.clf()

# feature_selectors = [
#     SymmetricalUncertainty(),
#     Relief(),
#     SVM_RFE(),
#     LassoFeatureSelector(),
# ]
#
# e_methods = [
#     ensemble_methods.Mean(feature_selectors),
#     ensemble_methods.Influence(feature_selectors),
#     ensemble_methods.SMeanWithClassifier(
#         feature_selectors,
#         analysis.default_classifiers,
#         min_mean_max=[0, 1, 0]
#     ),
# ]
#
# fs = feature_selectors + e_methods
#
# data_sets = ["artificial", "colon", "arcene", "dexter", "gisette"]
#
# analysis.artificial(fs)
