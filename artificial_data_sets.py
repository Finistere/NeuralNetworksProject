import numpy as np
import os
import scipy.stats


class ArtificialSample:
    # own distribution_function has to support argument numpy array (n_features x n_samples)
    #  and return numpy array (n_samples) binary
    def __init__(self, n_features, n_samples, distribution_fun='normal', distribution_args=None):
        self.n_features = n_features
        self.n_samples = n_samples
        self.distribution_fun = distribution_fun
        self.distribution_args = distribution_args
        self.samples = None

    def init_samples(self):
        distribution_function = get_distribution_function(self.distribution_fun, self.distribution_args)
        self.samples = np.array([distribution_function(self.n_samples) for i in range(self.n_features)])


class ArtificialLabeledSample(ArtificialSample):
    def __init__(self, n_features, n_samples, distribution_fun='normal', distribution_args=None,
                 label_fun='linear', label_args=None):
        super().__init__(n_features, n_samples, distribution_fun, distribution_args)
        self.label_fun = label_fun
        self.label_args = label_args
        self.labels = None

    def init_labeled_samples(self):
        if self.samples is None: self.init_samples()
        distribution_function = get_distribution_function(self.distribution_fun, self.distribution_args)
        label_function = get_label_function(self.label_fun, self.label_args)
        self.samples = np.array([distribution_function(self.n_samples) for i in range(self.n_features)])
        self.labels = label_function(self.samples)


class ArtificialNoiseSample(ArtificialSample):
    #type_of_noise function has to support argument sigma
    def __init__(self, n_features, n_samples, type_of_noise='normal', variance=0.1):
        fun_args = {'sigma': variance, 'mean': 0}
        super().__init__(n_features, n_samples, type_of_noise, fun_args)
        self.type_of_noise = type_of_noise
        self.variance = variance


class ArtificialDataSet:
    # sample noise: noise added to each sample
    # noise features: features consist out of noise
    def __init__(self, n_features, n_samples, distribution_fun='normal', distribution_args=None,
                 classification_fun='linear', type_of_noise='normal', noise_variance=0.1, n_noise_features=0,
                 type_of_noise_features='normal', noise_features_variance=0.1):
        self.artificial_labeled_samples = ArtificialLabeledSample(n_features, n_samples, distribution_fun,
                                                                 distribution_args, classification_fun)
        self.artificial_labeled_samples_noise = ArtificialNoiseSample(n_features, n_samples, type_of_noise,
                                                                     noise_variance)
        self.artificial_noise_features = ArtificialNoiseSample(n_noise_features, n_samples, type_of_noise_features,
                                                               noise_features_variance)

    def init_data_set(self):
        self.artificial_labeled_samples.init_labeled_samples()
        self.artificial_labeled_samples_noise.init_samples()
        self.artificial_noise_features.init_samples()

    def obtain_artificial_data_set(self):
        self.init_data_set()
        samples = self.artificial_labeled_samples.samples + self.artificial_labeled_samples_noise.samples
        noise_features = self.artificial_noise_features.samples
        artificial_data_set = np.vstack((samples, noise_features))
        return artificial_data_set, self.artificial_labeled_samples.labels

    def save_data_set(self):
        data_set, labels = self.obtain_artificial_data_set()
        if not os.path.exists('../ARTIFICIAL/ARTIFICIAL/'): os.makedirs('../ARTIFICIAL/ARTIFICIAL/')
        np.savetxt('../ARTIFICIAL/ARTIFICIAL/artificial.data', data_set)
        np.savetxt('../ARTIFICIAL/ARTIFICIAL/artificial.labels', labels)
        os.rename('../ARTIFICIAL/ARTIFICIAL/artificial.data.npy', '../ARTIFICIAL/ARTIFICIAL/artificial.data')
        os.rename('../ARTIFICIAL/ARTIFICIAL/artificial.labels.npy', '../ARTIFICIAL/ARTIFICIAL/artificial.labels')
        return 1


def get_distribution_function(distribution_fun, fun_args):
    if distribution_fun == 'normal':
        distribution_function = _normal(fun_args)
    elif distribution_fun == 'laplace':
        distribution_function = _laplace(fun_args)
    elif distribution_fun == 'uniform':
        distribution_function = _uniform(fun_args)
    elif callable(distribution_fun):
        def distribution_function(**distribution_args):
            return distribution_fun(**distribution_args)

    return distribution_function


def _normal(fun_args=None):
    fun_args = {} if fun_args is None else fun_args
    mean = fun_args.get('mean', 0)
    sigma = fun_args.get('sigma', 1)

    def normal_function(n_samples):
        return scipy.stats.norm.rvs(loc=mean, scale=sigma, size=n_samples)

    return normal_function


def _laplace(fun_args=None):
    fun_args = {} if fun_args is None else fun_args
    mean = fun_args.get('mean', 0)
    sigma = fun_args.get('sigma', 1)

    def laplace_function(n_samples):
        return scipy.stats.laplace.rvs(loc=mean, scale=sigma, size=n_samples)

    return laplace_function


def _uniform(fun_args=None):
    fun_args = {} if fun_args is None else fun_args
    mean = fun_args.get('mean', 0)
    sigma = fun_args.get('sigma', 1)

    def uniform_function(n_samples):
        return scipy.stats.uniform.rvs(loc=mean, scale=sigma, size=n_samples)

    return uniform_function


def get_label_function(label_fun, fun_args):
    if label_fun == 'linear':
        label_function = _linear_aggregation(fun_args)
    elif callable(label_fun):
        def label_function(**fun_args):
            return label_fun(**fun_args)

    return label_function


def _linear_aggregation(fun_args=None):
    fun_args = {} if fun_args is None else fun_args
    weights = fun_args.get('weights', None)

    def linear_aggregation(samples):
        if weights is None:
            weighted_samples = samples
        else:
            assert weights.size == samples.shape[1]
            weighted_samples = np.dot(samples, weights)

        return np.sign(np.sum(weighted_samples, axis=1))

    return linear_aggregation
