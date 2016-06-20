from analyse_weights import AnalyseBenchmarkResults
from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE


feature_selectors = [
    SymmetricalUncertainty()]

analysis = AnalyseBenchmarkResults(feature_selectors)
analysis.run("arcene")