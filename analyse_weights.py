import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data_sets import Analysis, Weights
import pandas as pd
import numpy as np
from feature_selector import FeatureSelector
from sklearn.cross_validation import ShuffleSplit
import os
import errno


class AnalyseBenchmarkResults():
    def __init__(self, feature_selector: FeatureSelector = None, feature_selection_method="weight"):
        self.feature_selector = feature_selector

        if not isinstance(feature_selector, list):
            self.feature_selectors = [feature_selector]
        else:
            self.feature_selectors = feature_selector

        self.feature_selection_method = feature_selection_method

    def run(self, data_set, save_to_file=False):
        for i in range(len(self.feature_selectors)):
            analysis = AnalyseFeatureSelection(self.feature_selectors[i], save_to_file)
            analysis.generate(
                data_set,
                self.cv(),
                self.feature_selection_method)

    @staticmethod
    def cv():
        return ShuffleSplit(0)


class AnalyseFeatureSelection:
    def __init__(self, feature_selector: FeatureSelector, save_to_file=False):
        self.feature_selector = feature_selector
        self.save_to_file = save_to_file

    def generate(self, data_set, cv, assessment_method):
        weights = Weights.load(data_set, cv, assessment_method, self.feature_selector)
        stats, fig = AnalyseWeights().analyse_weights(weights.T)

        try:
            os.makedirs(Analysis.dir_name(data_set, cv, assessment_method))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        directory = Analysis.file_name(data_set, cv, assessment_method, self.feature_selector)

        fig.suptitle("Weight analysis for " + self.feature_selector.__name__, fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.9)

        if self.save_to_file:
            fig.savefig(directory + '.png')
            stats.to_csv(directory + '.csv')

        plt.show()
        print(stats)


class AnalyseWeights:
    # shape: (features x samples)
    def analyse_weights(self, weights):
        column_names = ['S' + str(s) for s in range(weights.shape[1])]
        weights_df = pd.DataFrame(weights, columns=column_names)
        weights_mean = weights_df.T.mean()

        stats = self.weights_stats(weights_df, weights_mean)
        fig = self.weights_plot(weights_df, weights_mean)
        return stats, fig

    def weights_stats(self, weights, weights_mean):
        weights_mean_df = pd.DataFrame(weights_mean, columns=['mean'])
        stats_df = pd.concat([weights, weights_mean_df], axis=1)
        stats_matrix = stats_df.as_matrix()
        n_unique_values = [len(np.unique(stats_matrix[:, i])) for i in range(stats_matrix.shape[1])]
        unique_df = pd.DataFrame(n_unique_values, columns=['unique']).T
        unique_df.columns = ['S' + str(s) for s in range(weights.shape[1])] + ['mean']
        stats = stats_df.describe().append(unique_df)

        return stats

    def weights_plot(self, weights, weights_mean):
        fig = plt.figure(figsize=(15, 10))
        sample_size = weights.shape[1]
        gs = GridSpec(round(sample_size / 3 + 0.5), 6)

        ax = fig.add_subplot(gs[:3, 0:3])
        self.plot_boxplot(weights, ax)

        for i in range(sample_size):
            ax = fig.add_subplot(gs[int(i / 3), 3 + (i % 3)])
            self.plot_hist(weights, weights_mean, ax, i)

        fig.tight_layout()
        return fig

    def plot_boxplot(self, weights, ax):
        meanlineprops = dict(linestyle='-', linewidth=1.5, color='purple')
        weights.boxplot(ax=ax, return_type='axes', meanprops=meanlineprops,
                        meanline=True, showmeans=True, notch=True, showfliers=False)

    def plot_hist(self, weights, weights_mean, ax, sample_index):
        n_bins = 50
        max_xticks = 4
        max_yticks = 5

        weights['S' + str(sample_index)].plot.hist(ax=ax, color='green', alpha=0.5, bins=n_bins)
        weights_mean.plot.hist(ax=ax, color='orange', alpha=0.5, bins=n_bins)
        ax.set_ylabel('')
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)
        yloc = plt.MaxNLocator(max_yticks)
        ax.yaxis.set_major_locator(yloc)
