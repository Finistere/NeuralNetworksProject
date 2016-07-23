import numpy as np


def linear_labels(weights=None):
    def labeling(samples, weights=weights):
        if weights is None:
            weights = np.random.uniform(-1, 1, samples.shape[0])
        labels = weights.dot(samples)
        return np.sign(labels - np.median(labels))

    return labeling


def linear_power_labels():
    def labeling(samples):
        powers = np.full(samples.shape[0], 2)
        samples_power = np.power(samples.T, powers).T
        return linear_labels()(samples_power)

    return labeling


def generate(n_samples, n_features, n_significant_features, feature_distribution,
             noise_distribution=None,
             insignificant_feature_distribution=None,
             labeling=linear_labels()
             ):
    if n_significant_features <= 0 or n_significant_features > n_samples:
        raise ValueError("significant needs to be positive and inferior to the number of samples")

    if n_significant_features < 1:
        n_significant_features = np.ceil(n_significant_features * n_features)

    if insignificant_feature_distribution is None:
        samples = feature_distribution((n_features, n_samples))
    else:
        samples = np.vstack((
            feature_distribution((n_significant_features, n_samples)),
            insignificant_feature_distribution((n_features - n_significant_features, n_samples))
        ))

    if noise_distribution:
        samples += noise_distribution((n_features, n_samples))

    labels = labeling(samples[:n_significant_features])

    feature_labels = np.hstack((np.ones(n_significant_features),-np.ones(n_features-n_significant_features)))
    return samples, labels, feature_labels


def constant(c):
    return lambda s: np.full(s, c)


def uniform(a, b):
    return lambda s: np.random.uniform(a, b, s)


def laplace(mean, variance):
    return lambda s: np.random.laplace(mean, variance, s)


def normal(mean, variance):
    return lambda s: np.random.normal(mean, variance, s)


def multivariate_normal(mean, cov):
    def distribution(s, mean=mean, cov=cov):
        if callable(mean):
            mean = mean(s[0])

        if callable(cov):
            cov = cov((s[0], s[0]))
            # positive semi-definite matrix
            cov = cov.T.dot(cov)

        return np.random.multivariate_normal(mean, cov, s[1]).T

    return distribution


def multiple_distribution(distributions, shares):
    def distribution(s, distributions=distributions, shares=np.array(shares)):
        shares /= shares.sum()
        shares = np.array([int(share * s[0]) for share in shares])
        shares[0] += s[0] - shares.sum()

        samples = []
        for i, share in enumerate(shares):
            samples.append(distributions[i]((share, s[1])))
        return np.vstack(tuple(samples))

    return distribution
