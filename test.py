from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector, Random, FeatureSelector
import ensemble_methods
import analysis
import artificial_data
import numpy as np
import matplotlib.pyplot as plt
from data_sets import DataSets
import itertools
from experiments import Experiment

import warnings

warnings.filterwarnings('ignore')

# total_features = 1e4
# n_significant_features = 100
# DataSets.save_artificial(
#     *artificial_data.generate(
#         n_samples=300,
#         n_features=total_features,
#         n_significant_features=n_significant_features,
#         feature_distribution=artificial_data.multiple_distribution(
#             distributions=[
#                 artificial_data.multivariate_normal(
#                     mean=artificial_data.constant(0),
#                     cov=artificial_data.uniform(0, 1)
#                 ),
#                 artificial_data.normal(-1, 1)
#             ],
#             shares=[0.5, 0.5]
#         ),
#         insignificant_feature_distribution=artificial_data.multiple_distribution(
#             distributions=[
#                 artificial_data.multivariate_normal(
#                     mean=artificial_data.constant(0),
#                     cov=artificial_data.uniform(0, 1)
#                 ),
#                 artificial_data.normal(0, 1)
#             ],
#             shares=[0.5, 0.5]
#         ),
#         labeling=artificial_data.linear_labeling(weights=np.ones(n_significant_features))
#     )
# )

# data, _ = DataSets.load("artificial")
# cov = np.cov(data[:200])
# plt.imshow(cov, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
# plt.clf()


def foo(p):
    feature_selectors = [
        SymmetricalUncertainty(),
        Relief(),
        SVM_RFE(percentage_features_to_select=p),
        LassoFeatureSelector(),
    ]

    e_methods = [
        ensemble_methods.Mean(feature_selectors),
        ensemble_methods.Influence(feature_selectors),
        ensemble_methods.MeanWithClassifier(feature_selectors, analysis.default_classifiers),
    ]

    fs = feature_selectors + e_methods

    data_sets = ["artificial", "colon", "arcene", "dexter", "gisette"]

    analysis.run(data_sets, fs, jaccard_percentage=p)
    analysis.artificial(fs)

foo(0.7 / 100)


def test():
    fs = [
        SymmetricalUncertainty(),
        Relief(),
        SVM_RFE(),
        LassoFeatureSelector(),
    ]

    e_methods = [
        ensemble_methods.Mean(fs),
        ensemble_methods.Influence(fs),
        ensemble_methods.MeanWithClassifier(fs, analysis.default_classifiers),
    ]

    for comb in itertools.combinations(list(range(4)), 3):
        comb_fs = [fs[i] for i in comb]
        e_methods.extend([
            ensemble_methods.Mean(comb_fs),
            ensemble_methods.Influence(comb_fs),
            ensemble_methods.MeanWithClassifier(comb_fs, analysis.default_classifiers),
        ])

    for comb in itertools.combinations(list(range(4)), 2):
        comb_fs = [fs[i] for i in comb]
        e_methods.extend([
            ensemble_methods.Mean(comb_fs),
            ensemble_methods.Influence(comb_fs),
            ensemble_methods.MeanWithClassifier(comb_fs, analysis.default_classifiers),
        ])

    data_sets = ["colon", "arcene", "dexter", "gisette"]

    analysis.run(data_sets, fs + e_methods, prefix="combinations")


def combination_plot():
    from tabulate import tabulate

    accuracy = np.load('../results/RAW/combinations_jc10_accuracy.npy')
    robustness = np.load('../results/RAW/combinations_jc10_robustness.npy')
    labels = []
    with open('../results/RAW/combinations_jc10_accuracy_1.txt') as f:
        for label in f:
            labels.append(label.strip())

    beta = 2 * accuracy.mean(axis=-1) * robustness.mean(axis=-1) / (robustness.mean(axis=-1) + accuracy.mean(axis=-1))
    m_b = beta.mean(axis=(-1, 0))
    std_b = beta.std(axis=(-1, 0))

    m_a = accuracy.mean(axis=(-1, -2, 0))
    std_a = accuracy.std(axis=(-1, -2, 0))
    m_r = robustness.mean(axis=(-1, -2, 0))
    std_r = robustness.std(axis=(-1, -2, 0))


    def tprint(order):
        order = [o for o in order if o < 4 or (o - 4) % 3 == 0]
        rows = [
            ["accuracy"] + list(map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), m_a[order].tolist(), std_a[order].tolist())),
            ["robustness"] + list(map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), m_r[order].tolist(), std_r[order].tolist())),
            ["beta"] + list(map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), m_b[order].tolist(), std_b[order].tolist())),
        ]
        print(tabulate(rows, ["Measure"] + [labels[i] for i in order], tablefmt='pipe'))
        print()

    print("ACCURACY")
    tprint(np.argsort(m_a)[::-1])
    print("ROBUSTNESS")
    tprint(np.argsort(m_r)[::-1])
    print("BETA")
    tprint(np.argsort(m_b)[::-1])
