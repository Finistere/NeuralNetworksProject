import numpy as np
import scipy as sp
import filters

class RobustnessMeasurement:

    def __init__(self, sample, classes):
        self.sample = np.array(sample)
        self.classes = np.array(classes)

    # apply all similariy
    def calculate_robustness(self, similarity_measurement='spearman', 
                             subsample_size_coeff=0.9, subsample_number=10):
        """Calculate the robustness for a feature selection
        with a similarity measurement

        Parameters
        ----------
        similarity_measurement: str TODO build this in or remove it
            The name of the measurment method for similarity
            Valid names are 'spearman' for Spearman rank correlation
            coefficient
        subsample_size: float
            Value between 0 and 1 which sets the size of a subsample
            by int(sample_size * subsample_size_coeff).
            In the paper the term [xM] is used for this parameter
        subsample_number: int
            The number of subsamples to iterate over
            In the paper the term k is used for this parameter

        Returns
        -------
        out: float
            The robustness coefficient
        """

        sample_size = self.sample.shape[1]
        subsample_size = int(sample_size * subsample_size_coeff)

        if(subsample_size > sample_size):
            raise ValueError('subsample_size is smaller than sample_size')

        features_rank = np.zeros((subsample_number,self.sample.shape[0]))

        print("Start ranking features with %d subsamples of size %d" %(subsample_number, subsample_size))
        for i in range(subsample_number):
            print("Start run number %d"%i)
            # subsampling process
            subsample_indices = np.arange(sample_size)
            np.random.shuffle(subsample_indices)
            subsample = self.sample[:,subsample_indices[0:subsample_size]]
            subclasses = self.classes[subsample_indices[0:subsample_size]]

            # apply feature selection method
            SUF = filters.SUFilter(subsample, subclasses)
            features_rank[i] = SUF.fit(ranked=True)
            print("end run number %d"%i)
            
        print("Finished ranking")
        # apply spearman similarity measures
        spearman_coeffs, _ = sp.stats.spearmanr(features_rank.T)
        # takes the lower triangular matrix to sum over
        S_tot = np.sum(np.tril(spearman_coeffs,-1)) / (subsample_number*(subsample_number-1))
        return S_tot




