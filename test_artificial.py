from artificial_data_sets import ArtificialDataSet

artificial_data_set = ArtificialDataSet(n_features=100, n_samples=500, n_noise_features=10000 - 100)
artificial_data_set.save_data_set()
