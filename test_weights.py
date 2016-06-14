import numpy as np
from analyse_weights import AnalyseWeights


weights = np.load("../feature_weights/arcene/KFold/SymmetricalUncertainty.npy")
aw = AnalyseWeights()
aw.run_analysis(weights.T)