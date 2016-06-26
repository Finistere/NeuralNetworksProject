import artificial_data_sets
import numpy as np


class TestArtificialUnivariateSample:
    n_features = 10
    n_samples = 5
    artificial_sample = artificial_data_sets.ArtificialUnivariateSample(n_features=n_features, n_samples=n_samples)

    def init_samples(self):
        self.artificial_sample.init_samples()
        assert self.artificial_sample.data.shape[0] == self.n_features
        assert self.artificial_sample.data.shape[1] == self.n_samples


class TestArtificialSampleNoise:
    n_features = 10
    n_samples = 5
    artificial_sample = artificial_data_sets.ArtificialSampleNoise(n_features=n_features, n_samples=n_samples)

    def init_samples(self):
        self.artificial_sample.init_samples()
        assert self.artificial_sample.data.shape[0] == self.n_features
        assert self.artificial_sample.data.shape[1] == self.n_samples


class TestArtificialMultivariateSample:
    n_features = 10
    n_samples = 5
    artificial_sample = artificial_data_sets.ArtificialMultivariateSample(n_features=n_features, n_samples=n_samples)

    def init_samples(self):
        self.artificial_sample.init_samples()
        assert self.artificial_sample.data.shape[0] == self.n_features
        assert self.artificial_sample.data.shape[1] == self.n_samples


class TestArtificialLabeledSample:
    n_features = 10
    n_samples = 5
    artificial_sample = artificial_data_sets.ArtificialUnivariateSample(n_features=n_features, n_samples=n_samples)
    artificial_labeled_samples = artificial_data_sets.ArtificialLabeledSample(artificial_sample)

    def test_init_labeled_samples(self):
        self.artificial_labeled_samples.init_labeled_samples()
        assert self.artificial_labeled_sample.samples.data.shape[0] == self.n_features
        assert self.artificial_labeled_sample.samples.data.shape[1] == self.n_samples
        assert self.artificial_labeled_sample.labels.size == self.n_samples


class TestListArtificialDataSet:
    n_features = 10
    n_samples = 5
    n_noise_features = 95
    artificial_sample_univariate = artificial_data_sets.ArtificialUnivariateSample(n_features=n_features,
                                                                                   n_samples=n_samples)
    artificial_sample_multivariate = artificial_data_sets.ArtificialMultivariateSample(n_features=n_features,
                                                                                       n_samples=n_samples)
    artificial_sample_noise = artificial_data_sets.ArtificialSampleNoise(n_features=n_features, n_samples=n_samples)
    artificial_noise_univariate = artificial_data_sets.ArtificialUnivariateSample(n_features=n_noise_features,
                                                                               n_samples=n_samples)
    artificial_noise_multivariate = artificial_data_sets.ArtificialMultivariateSample(n_features=n_features,
                                                                                       n_samples=n_samples)
    artificial_data_set = artificial_data_sets.ArtificialDataSet(
        [artificial_sample_univariate, artificial_sample_multivariate], artificial_sample_noise,
        [artificial_noise_univariate,artificial_noise_multivariate])

    def test_obtain_artificial_data_set(self):
        data_set, labels = self.artificial_data_set.obtain_artificial_data_set()
        assert data_set.shape[0] == (self.n_features + self.n_noise_features)
        assert data_set.shape[1] == self.n_samples
        assert labels.size == self.n_samples


class TestArtificialDataSet:
    n_features = 10
    n_samples = 5
    n_noise_features = 95
    artificial_sample = artificial_data_sets.ArtificialUnivariateSample(n_features=n_features, n_samples=n_samples)
    artificial_sample_noise = artificial_data_sets.ArtificialSampleNoise(n_features=n_features, n_samples=n_samples)
    artificial_noise_feature = artificial_data_sets.ArtificialUnivariateSample(n_features=n_noise_features,
                                                                               n_samples=n_samples)
    artificial_data_set = artificial_data_sets.ArtificialDataSet(artificial_sample, artificial_sample_noise,
                                                                 artificial_noise_feature)

    def test_obtain_artificial_data_set(self):
        data_set, labels = self.artificial_data_set.obtain_artificial_data_set()
        assert data_set.shape[0] == (self.n_features + self.n_noise_features)
        assert data_set.shape[1] == self.n_samples
        assert labels.size == self.n_samples


class TestDistributionFunctions:
    # check if it returns function with paratemers
    def test_normal_with_parameters(self):
        fun_args = {'mean': 1, 'sigma': 2}
        normal_function = artificial_data_sets._normal(fun_args)

        def function(): return 0

        assert type(normal_function) == type(function)

    # check if it returns function without paratemers
    def test_normal_without_parameters(self):
        fun_args = None
        normal_function = artificial_data_sets.data_normal(fun_args)

        def function(): return 0

        assert type(normal_function) == type(function)

    # check if it returns number of sample
    def test_normal_sample_size(self):
        fun_args = {'mean': 1, 'sigma': 2}
        normal_function = artificial_data_sets._normal(fun_args)
        assert len(normal_function(5)) == 5

    # test own function
    def test_get_distribution_function(self):
        def deterministic_function(x, n_samples):
            return [(1 if x > 0 else 0) for i in range(n_samples)]

        distribution_function = self.TestingSet.get_distribution_function(
            deterministic_function=deterministic_function, distribution_args={'x': 1})
        assert np.allclose(distribution_function(2) == np.array([1, 1]))
