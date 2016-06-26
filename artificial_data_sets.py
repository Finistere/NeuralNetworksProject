import numpy as np
import os
import scipy.stats
from abc import ABCMeta, abstractmethod


class ArtificialSample:
    def __init__(self, n_features=0, n_samples=0, distribution_fun='normal', distribution_args=None):
        self.n_features = n_features
        self.n_samples = n_samples
        self.distribution_fun = distribution_fun
        self.distribution_args = distribution_args
        self.data = None

    @abstractmethod
    def init_data(self, n_features=None, n_samples=None):
        pass


class ArtificialUnivariateSample(ArtificialSample):
    # own distribution_function has to support argument numpy array (n_features x n_samples)
    #  and return numpy array (n_samples) binary
    def init_data(self, n_features=None, n_samples=None):
        if n_samples is not None: self.n_samples = n_samples
        if n_features is not None: self.n_features = n_features
        distribution_function = get_distribution_function(self.distribution_fun, self.distribution_args)
        self.data = np.array([distribution_function(self.n_samples) for i in range(self.n_features)])


class ArtificialSampleNoise(ArtificialUnivariateSample):
    # type_of_noise function has to support argument sigma
    def __init__(self, n_features=0, n_samples=0, type_of_noise='normal', variance=0.1):
        fun_args = {'sigma': variance, 'mean': 0}
        super().__init__(n_features, n_samples, type_of_noise, fun_args)
        self.type_of_noise = type_of_noise
        self.variance = variance


class ArtificialMultivariateSample(ArtificialSample):
    # own distribution_function has to support argument numpy array (n_samples)
    # and return numpy array (n_features, n_samples) binary
    def init_data(self, n_features=None, n_samples=None):
        if n_samples is not None: self.n_samples = n_samples
        if n_features is not None: self.n_features = n_features
        distribution_function = get_multivariate_distribution_function(self.distribution_fun, self.distribution_args,
                                                                       self.n_features)
        self.data = distribution_function(self.n_samples)


class ArtificialLabeledSample:
    def __init__(self, samples: ArtificialSample, label_fun='linear', label_args=None):
        samples_list = samples
        if not isinstance(samples, list):
            samples_list = [samples]

        n_samples = samples_list[0].n_samples
        for sample in samples_list:
            if sample.n_samples != n_samples:
                print("Not all samples have same n_sample")
                raise

        self.samples = samples_list
        self.label_fun = label_fun
        self.label_args = label_args
        self.labels = None

    def init_data(self):
        for sample in self.samples:
            if sample.data is None: sample.init_data()

    def init_labeled_samples(self):
        label_function = get_label_function(self.label_fun, self.label_args)
        self.init_data()
        data = self.get_merged_data()
        self.labels = label_function(data)

    def get_merged_data(self):
        return get_merged_samples(self.samples)


class ArtificialDataSet:
    # sample noise: noise added to each sample
    # noise features: features consist out of noise
    def __init__(self, samples: ArtificialSample, noise_samples: ArtificialSampleNoise,
                 noise_features: ArtificialSample, label_fun='linear', label_args=None):
        samples_list = samples
        if not isinstance(samples, list):
            samples_list = [samples]
        noise_features_list = noise_features
        if not isinstance(noise_features, list):
            noise_features_list = [noise_features]

        if samples_list[0].n_samples != noise_samples.n_samples:
            print("noise_sample and samples do not have the same size")
            raise

        self.labeled_samples = ArtificialLabeledSample(samples, label_fun, label_args)
        self.noise_samples = noise_samples
        self.noise_features = noise_features_list

    def init_data_set(self):
        self.labeled_samples.init_labeled_samples()
        self.noise_samples.init_data()
        for noise_feature in self.noise_features:
            if noise_feature.data is None: noise_feature.init_data()

    def obtain_artificial_data_set(self):
        self.init_data_set()
        samples = self.labeled_samples.get_merged_data() + self.noise_samples.data
        noise_features = get_merged_samples(self.noise_features)
        artificial_data_set = np.vstack((samples, noise_features))
        return artificial_data_set, self.labeled_samples.labels

    def save_data_set(self):
        data_set, labels = self.obtain_artificial_data_set()
        if not os.path.exists('../ARTIFICIAL/ARTIFICIAL/'): os.makedirs('../ARTIFICIAL/ARTIFICIAL/')
        np.savetxt('../ARTIFICIAL/ARTIFICIAL/artificial.data', data_set)
        np.savetxt('../ARTIFICIAL/ARTIFICIAL/artificial.labels', labels)
        os.rename('../ARTIFICIAL/ARTIFICIAL/artificial.data.npy', '../ARTIFICIAL/ARTIFICIAL/artificial.data')
        os.rename('../ARTIFICIAL/ARTIFICIAL/artificial.labels.npy', '../ARTIFICIAL/ARTIFICIAL/artificial.labels')
        return 1


def get_merged_samples(sample_list):
    if not (isinstance(sample_list, list) or isinstance(sample_list[0], ArtificialSample)):
        print("Input should be list of ArtificialSample")
        raise

    total_features = 0
    for sample in sample_list:
        total_features += sample.n_features

    data = np.zeros((total_features, sample_list[0].n_samples))
    current_feature = 0
    for sample in sample_list:
        data[current_feature:current_feature + sample.n_features] = sample.data

    return data


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
    elif distribution_fun is None:
        distribution_function = np.zeros

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


def get_multivariate_distribution_function(multivariate_distribution_fun, fun_args, n_features):
    if multivariate_distribution_fun == 'normal':
        multivariate_distribution_function = _multivariate_normal(fun_args, n_features)
    elif callable(multivariate_distribution_fun):
        def distribution_function(**distribution_args):
            return multivariate_distribution_fun(**distribution_args)
    elif multivariate_distribution_fun is None:
        multivariate_distribution_function = np.zeros

    return multivariate_distribution_function


def _multivariate_normal(fun_args, n_features):
    fun_args = {} if fun_args is None else fun_args
    mean = fun_args.get('mean', np.zeros(n_features))
    sigma = fun_args.get('sigma', np.eye(n_features))

    def multivariate_normal(n_samples):
        return np.random.multivariate_normal(mean=mean, cov=sigma, size=n_samples).T

    return multivariate_normal


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

        return np.sign(np.sum(weighted_samples, axis=0))

    return linear_aggregation
