import numpy as np
import scipy as sp
from sklearn.metrics import jaccard_similarity_score 
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

        features_weight= np.zeros((subsample_number,self.sample.shape[0]))

        #print("Start ranking features with %d subsamples of size %d" %(subsample_number, subsample_size))
        for i in range(subsample_number):
            #print("Start run number %d"%i)
            subsample, subclasses = self.simple_random_sampling(subsample_size)
            # apply feature selection method
            SUF = filters.SUFilter(subsample, subclasses)
            features_weight[i] = SUF.apply_su_on_data()
            #print("end run number %d"%i)
        
        #TODO Robustness reiceves list of features

        #print("Finished ranking")
        features_rank = self.rank_features(features_weight)
        # apply spearman similarity measures
        spearman_coeffs = self.spearmean_coefficient(features_rank)
        # takes the lower triangular matrix to sum over
        spearman_total = self.mean_of_lower_triangular_of_symmetric_matrix(spearman_coeffs)
        jaccard_indices = jaccard_index(features_rank)
        jaccard_total = self.mean_of_lower_triangular_of_symmetric_matrix(jaccard_indices)
        return spearman_total, jaccard_total

    def simple_random_sampling(self, subsample_size):
        sample_size = self.sample.shape[1]
        subsample_indices = np.arange(sample_size)
        np.random.shuffle(subsample_indices)
        subsample = self.sample[:,subsample_indices[0:subsample_size]]
        subclasses = self.classes[subsample_indices[0:subsample_size]]
        return subsample, subclasses

    def rank_features(self, features_weight):
        features_rank = [sp.stats.rankdata(features_weight[i], method='ordinal') 
                for i in range(features_weight.shape[0])]
        return np.array(features_rank)

    def filter_ranked_features(self, features_rank, perc):
        minimal_rank = int((1-perc)*features_rank.shape[1])
        # set everything below the minimal rank to zero and everything else to 1
        selected_features = (features_rank > minimal_rank).reshape(-1)
        return selected_features
    
    def spearmean_coefficient(self, features_rank):
        spearman_coeffs, _ = sp.stats.spearmanr(features_rank, axis=1)
        return spearman_coeffs

    def mean_of_lower_triangular_of_symmetric_matrix(self, symmetric_matrix):
        numerator = 2*np.sum(np.tril(symmetric_matrix,-1)) 
        number_of_rows = symmetric_matrix.shape[0]
        denominator = number_of_rows*(number_of_rows-1)
        return numerator/denominator



def jaccard_index(features_rank, perc=0.1):
    if(np.any(np.min(features_rank, axis=1) != np.ones(features_rank.shape[0],dtype=np.int))):
        raise ValueError('features_rank raking does not always begin with 1')
    # the minimal rank a feature mast have to be chose
    minimal_rank = int((1-perc)*features_rank.shape[1]) + 1
    features_rank_c = np.copy(features_rank)
    # set everything below the minimal rank to zero and everything else to 1
    features_rank_c[minimal_rank > features_rank_c] = 0
    features_rank_c[0 != features_rank_c] = 1

    k = features_rank.shape[0]
    jaccard_indices = np.zeros((k,k))

    for i in range(k):
        for j in range(k):
            # both features have to be 1
            condition = np.logical_and(features_rank_c[i]==1, features_rank_c[j]==1)
            jaccard_indices[i,j] = np.sum(condition)

    # normalize
    jaccard_indices = jaccard_indices/float(features_rank.shape[1]-(minimal_rank-1))
    return jaccard_indices

