from analyse_weights import AnalyseBenchmarkResults
from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector


feature_selectors = [
    SymmetricalUncertainty(),
    Relief(),
    SVM_RFE(),
    LassoFeatureSelector()
]

analysis = AnalyseBenchmarkResults(feature_selectors)
analysis.run("colon")