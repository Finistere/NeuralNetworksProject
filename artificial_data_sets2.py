import numpy as np


class ArtificialData:
    @staticmethod
    def generate(n_samples, n_features, n_significant_features, sample_distribution, noise_distribution=None):
        if n_significant_features <= 0 or n_significant_features > n_samples:
            raise ValueError("significant needs to be positive and inferior to the number of samples")

        if n_significant_features < 1:
            n_significant_features = np.ceil(n_significant_features * n_features)

        shape = (n_features, n_samples)
        samples = sample_distribution(shape)

        if noise_distribution:
            samples += noise_distribution(shape)

        labels = ArtificialData.__generate_labels(samples[:n_significant_features])

        return samples, labels

    @staticmethod
    def __generate_labels(samples, n_iter=10):
        i = 0
        while True:
            i += 1
            weights = np.random.uniform(-1, 1, samples.shape[0])
            labels = np.sign(weights.dot(samples)).T
            # not every label is the same
            if np.abs(labels.sum()) < samples.shape[1] * 0.95:
                return labels
            if i > n_iter:
                raise Exception("Labels could not be generated after {} iterations".format(n_iter))

    @staticmethod
    def normal(n_samples, n_features, significant=0.01, mean=0, variance=1, noise_variance=0):
        if variance <= 0:
            raise ValueError("Variance can't be negative")
        if noise_variance < 0:
            raise ValueError("Noise Variance can't be negative")

        return ArtificialData.generate(
            n_samples,
            n_features,
            n_significant_features=significant,
            sample_distribution=lambda s: np.random.normal(mean, variance, s),
            noise_distribution=None if noise_variance == 0 else lambda s: np.random.normal(0, noise_variance, s)
        )

    def multivariate_normal(self, n_samples, n_features, significant=0.01, mean=0, cov=1, noise_variance=0):
        pass