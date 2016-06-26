from artificial_data_sets import ArtificialUnivariateSample, ArtificialMultivariateSample, ArtificialSampleNoise,\
    ArtificialDataSet

n_samples = 200
sample_univariate = ArtificialUnivariateSample(n_features=90, n_samples=n_samples)
sample_multivariate = ArtificialMultivariateSample(n_features=10, n_samples=n_samples)
sample_noise = ArtificialSampleNoise(n_features=100, n_samples=n_samples)
noise_feature_univariate = ArtificialUnivariateSample(n_features=9000, n_samples=n_samples)
noise_feature_multivariate = ArtificialMultivariateSample(n_features=900, n_samples=n_samples)

data_set = ArtificialDataSet(samples=[sample_univariate, sample_multivariate], noise_samples=sample_noise,
                             noise_features=[noise_feature_univariate, noise_feature_multivariate])

# eben so
data, labels = data_set.obtain_artificial_data_set()
print(data.shape, labels.size)
# oder einfach speichern
#data_set.save_data_set()
