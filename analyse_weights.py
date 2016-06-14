import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd


class AnalyseWeights():
    # shape: (features x samples)
    def run_analysis(self, weights):
        numbering_as_string = [str(s) for s in range(weights.shape[1])]
        column_names = map(lambda number : 'S'+ number, numbering_as_string)
        weights_df = pd.DataFrame(weights, columns = column_names)
        weights_mean = weights_df.T.mean()

        self.print_stats(weights_df, weights_mean);
        self.plot(weights_df, weights_mean)

    def print_stats(self, weights, weights_mean):
        weights_mean_df = pd.DataFrame(weights_mean, columns=['mean'])
        stats = pd.concat([weights, weights_mean_df], axis=1).describe()
        print(stats)

    def plot(self, weights, weights_mean):
        fig = plt.figure(figsize=(15,10))
        sample_size = weights.shape[1]
        gs = GridSpec(round(sample_size/3 + 0.5), 6)

        ax = fig.add_subplot(gs[:3, 0:3])
        self.plot_boxplot(weights, ax)

        for i in range(sample_size):
            ax = fig.add_subplot(gs[int(i / 3), 3 + (i % 3)])
            self.plot_hist(weights, weights_mean, ax, i)

        plt.show()

    def plot_boxplot(self, weights, ax):
        meanlineprops = dict(linestyle='-', linewidth=1.5, color='purple')
        weights.boxplot(ax=ax, return_type='axes', meanprops=meanlineprops,
                   meanline=True, showmeans=True, notch=True, showfliers=False)

    def plot_hist(self, weights, weights_mean, ax, sample_index):
        weights['S'+str(sample_index)].plot.hist(ax=ax, color='green', alpha = 0.5)
        weights_mean.plot.hist(ax=ax, color='orange', alpha = 0.5)
        plt.ylabel('')
