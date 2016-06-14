import matplotlib.pyplot as plt
import pandas as pd


class AnalyseWeights():
    # shape: (features x samples)
    def run_analysis(self, weights):
        numbering_as_string = [str(s) for s in range(weights.shape[1])]
        column_names = map(lambda number : 'S'+ number, numbering_as_string)
        weights_df = pd.DataFrame(weights, columns = column_names)
        weights_mean = weights_df.T.mean()

        self.plot_boxplot(weights_df);
        self.plot_hist(weights_df, weights_mean);
        self.print_stats(weights_df, weights_mean);

    def plot_boxplot(self, weights):
        meanlineprops = dict(linestyle='-', linewidth=1.5, color='purple')
        plt.figure()
        weights.boxplot(return_type='axes', meanprops=meanlineprops,
                   meanline=True, showmeans=True, notch=True, showfliers=False)
        plt.show()

    def print_stats(self, weights, weights_mean):
        stats = pd.concat([weights, weights_mean], axis=1).describe()
        print(stats)

    def plot_hist(self, weights, weights_mean):
        plt.figure()
        for i in range(weights.shape[1]):
            plt.subplot(5,2,i+1)
            weights['S'+str(i)].plot.hist(color='green', alpha = 0.5)
            weights_mean.plot.hist(color='orange', alpha = 0.5)
        plt.show()
