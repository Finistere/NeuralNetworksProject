import artificial_data_sets
import numpy as np


class TestArtificialSample:
    n_features = 10
    n_samples = 5
    artificial_sample = artificial_data_sets.ArtificialSample(n_features=n_features, n_samples=n_samples)

    def init_samples(self):
        self.artificial_sample.init_samples()
        assert self.artificial_sample.samples.shape[0] == self.n_features
        assert self.artificial_sample.samples.shape[1] == self.n_samples


class TestArtificialLabeledSample:
    n_features = 10
    n_samples = 5
    artificial_labeled_sample = artificial_data_sets.ArtificialLabeledSample(n_features=n_features, n_samples=n_samples)

    def test_obtain_artificial_data_set(self):
        self.artificial_labeled_sample.init_labeled_samples()
        assert self.artificial_labeled_sample.samples.shape[0] == self.n_features
        assert self.artificial_labeled_sample.samples.shape[1] == self.n_samples
        assert self.artificial_labeled_sample.labels.size == self.n_samples


class TestArtificialNoiseSample:
    n_samples = 5
    n_noise_features = 100
    artificial_noise_sample = artificial_data_sets.ArtificialNoiseSample(n_samples=n_samples,
                                                                         n_features=n_noise_features)

    def test_obtain_artificial_data_set(self):
        self.artificial_noise_sample.init_samples()
        assert self.artificial_noise_sample.samples.shape[0] == self.n_noise_features
        assert self.artificial_noise_sample.samples.shape[1] == self.n_samples


class TestArtificialDataSet:
    n_features = 10
    n_samples = 5
    n_noise_features = 100
    artificial_data_set = artificial_data_sets.ArtificialDataSet(n_features=n_features, n_samples=n_samples,
                                                                 n_noise_features=n_noise_features)

    def test_obtain_artificial_data_set(self):
        data_set = self.artificial_data_set.obtain_artificial_data_set()
        assert data_set.shape[0] == (self.n_features + self.n_noise_features)
        assert data_set.shape[1] == self.n_samples


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
