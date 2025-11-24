"""
Ensemble classifiers for co-training framework.
Implements Section 3.3 of the paper: XGBoost, Random Forest, SVM
"""

from .feature_extractor import FeatureExtractor
from .ensemble_classifiers import EnsembleClassifiers

__all__ = ['FeatureExtractor', 'EnsembleClassifiers']
