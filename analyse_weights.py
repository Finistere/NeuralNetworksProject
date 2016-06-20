import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data_sets import DataSets
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

    def run(self, data_set):
        for i in range(len(self.feature_selectors)):
            analysis = AnalyseFeatureSelection(self.feature_selectors[i])
            analysis.generate(
                data_set,
                self.cv(),
                self.feature_selection_method)

    @staticmethod
    def cv():
        return ShuffleSplit(0)


class AnalyseFeatureSelection():
    def __init__(self, feature_selector: FeatureSelector, save_to_file = False):
        self.__name__ = feature_selector.__name__
        self.save_to_file = save_to_file

    def generate(self, data_set, cv, method):
        weights = self.load(data_set, cv, method)

        stats, fig = AnalyseWeights().analyse_weights(weights.T)

        try:
            os.makedirs(self.__dir_name_results(data_set, cv, method))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        directory = self.__file_name_results(data_set, cv, method)

        fig.suptitle("Weight analysis for " + self.__name__, fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.9)

        if(self.save_to_file):
            fig.savefig(directory + '.png')
            stats.to_pickle(directory + '.pkl')

        plt.show()
        print(stats)

    def load(self, data_set, cv, method):
        try:
            filename = self.__file_name_weigths(data_set, cv, method) + ".npy"
            return np.load(filename)
        except FileNotFoundError:
            raise "Weight " + filename + " not found"

    def __file_name_weigths(self, data_set, cv, method):
        return self.__dir_name_weigths(data_set, cv, method) + "/" + self.__name__

    @staticmethod
    def __dir_name_weigths(data_set, cv, method):
        return "{root_dir}/feature_{method}s/{data_set}/{cv}".format(
            root_dir=DataSets.root_dir,
            method=method,
            data_set=data_set,
            cv=type(cv).__name__
        )
    def __file_name_results(self, data_set, cv, method):
        return self.__dir_name_results(data_set, cv, method) + "/" + self.__name__

    @staticmethod
    def __dir_name_results(data_set, cv, method):
        return "{root_dir}/{method}s_results/{data_set}/{cv}".format(
            root_dir=DataSets.root_dir,
            method=method,
            data_set=data_set,
            cv=type(cv).__name__
        )


class AnalyseWeights:
    # shape: (features x samples)
    def analyse_weights(self, weights):
        numbering_as_string = [str(s) for s in range(weights.shape[1])]
        column_names = map(lambda number: 'S' + number, numbering_as_string)
        weights_df = pd.DataFrame(weights, columns=column_names)
        weights_mean = weights_df.T.mean()

        stats = self.weights_stats(weights_df, weights_mean)
        fig = self.weights_plot(weights_df, weights_mean)
        return stats, fig

    def weights_stats(self, weights, weights_mean):
        weights_mean_df = pd.DataFrame(weights_mean, columns=['mean'])
        stats = pd.concat([weights, weights_mean_df], axis=1).describe()
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
