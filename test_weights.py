import numpy as np
import analyse_weights as aw

weights = np.random.normal(0, 1, (100,10))
analyse_weights = aw.AnalyseWeights()
analyse_weights.run_analysis(weights)