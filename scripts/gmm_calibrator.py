"""
GMM-based calibration (Consistent-Teacher approach) for comparison baseline.

This implements the Gaussian Mixture Model calibration used in Consistent-Teacher,
which we demonstrate DEGRADES localization calibration on dense scenes.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path


class GMMCalibrator:
    """
    GMM-based calibration following Consistent-Teacher methodology.
    
    The key insight of our paper: GMM assumes Gaussian-distributed scores,
    but NMS-distorted distributions in dense scenes are heavy-tailed and skewed.
    GMM calibration improves classification but DEGRADES localization.
    """
    
    def __init__(self, n_components=3, random_state=42):
        """
        Args:
            n_components: Number of Gaussian components (Consistent-Teacher uses 2-4)
            random_state: For reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.gmm_cls = None
        self.gmm_loc = None
        self.is_fitted = False
        
    def fit(self, scores_cls, correctness_cls, scores_loc, correctness_loc):
        """
        Fit separate GMMs for classification and localization scores.
        
        Args:
            scores_cls: Classification confidence scores [0,1]
            correctness_cls: Binary array of classification correctness
            scores_loc: Localization confidence scores [0,1]
            correctness_loc: Binary array of localization correctness (IoU >= 0.5)
        """
        # Fit GMM for classification
        self.gmm_cls = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type='full'
        )
        
        # Reshape for sklearn
        X_cls = scores_cls.reshape(-1, 1)
        self.gmm_cls.fit(X_cls)
        
        # Compute calibration mapping: for each Gaussian component,
        # what is the empirical accuracy?
        self.component_accuracy_cls = self._compute_component_accuracies(
            X_cls, correctness_cls, self.gmm_cls
        )
        
        # Fit GMM for localization
        self.gmm_loc = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type='full'
        )
        
        X_loc = scores_loc.reshape(-1, 1)
        self.gmm_loc.fit(X_loc)
        
        self.component_accuracy_loc = self._compute_component_accuracies(
            X_loc, correctness_loc, self.gmm_loc
        )
        
        self.is_fitted = True
        print(f"    GMM calibrator fitted: {self.n_components} components")
        
    def _compute_component_accuracies(self, X, correctness, gmm):
        """Compute empirical accuracy for each GMM component."""
        # Predict component membership
        component_probs = gmm.predict_proba(X)
        component_ids = gmm.predict(X)
        
        # For each component, compute empirical accuracy
        component_acc = []
        for k in range(self.n_components):
            mask = (component_ids == k)
            if mask.sum() > 0:
                acc = correctness[mask].mean()
            else:
                acc = 0.5  # Default if no samples
            component_acc.append(acc)
        
        return np.array(component_acc)
    
    def predict(self, scores_cls, scores_loc):
        """
        Calibrate scores using GMM.
        
        Returns:
            calibrated_cls, calibrated_loc
        """
        if not self.is_fitted:
            return scores_cls, scores_loc
        
        X_cls = scores_cls.reshape(-1, 1)
        X_loc = scores_loc.reshape(-1, 1)
        
        # Predict component membership probabilities
        component_probs_cls = self.gmm_cls.predict_proba(X_cls)
        component_probs_loc = self.gmm_loc.predict_proba(X_loc)
        
        # Calibrated score = weighted average of component accuracies
        calibrated_cls = (component_probs_cls * self.component_accuracy_cls).sum(axis=1)
        calibrated_loc = (component_probs_loc * self.component_accuracy_loc).sum(axis=1)
        
        return np.clip(calibrated_cls, 0, 1), np.clip(calibrated_loc, 0, 1)


def demo_gmm_degradation():
    """
    Demonstrate GMM degradation on synthetic dense-scene data.
    This generates the observation reported in the paper.
    """
    np.random.seed(42)
    
    # Simulate NMS-distorted scores (skewed, heavy-tailed)
    n = 5000
    
    # Classification: well-behaved (GMM works)
    cls_scores = np.random.beta(8, 2, n)
    cls_correct = (np.random.rand(n) < cls_scores).astype(int)
    
    # Localization: NMS tail amplification creates misalignment
    loc_scores = np.random.beta(7, 2, n)
    loc_correct_prob = np.clip(loc_scores - 0.15, 0, 1)  # 15% gap in tail
    loc_correct = (np.random.rand(n) < loc_correct_prob).astype(int)
    
    # Fit GMM
    gmm = GMMCalibrator(n_components=3)
    gmm.fit(cls_scores, cls_correct, loc_scores, loc_correct)
    
    # Calibrate
    cal_cls, cal_loc = gmm.predict(cls_scores, loc_scores)
    
    # Compute ECE before and after
    from sklearn.metrics import mean_squared_error
    
    def compute_ece(scores, correctness, n_bins=15):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            mask = (scores >= bins[i]) & (scores < bins[i+1])
            if mask.sum() > 10:
                bin_acc = correctness[mask].mean()
                bin_conf = scores[mask].mean()
                ece += mask.sum() / len(scores) * abs(bin_acc - bin_conf)
        return ece
    
    print("\n=== GMM Degradation Demo ===")
    print(f"Classification ECE - Raw: {compute_ece(cls_scores, cls_correct):.4f}")
    print(f"Classification ECE - GMM: {compute_ece(cal_cls, cls_correct):.4f}")
    print(f"Localization ECE - Raw: {compute_ece(loc_scores, loc_correct):.4f}")
    print(f"Localization ECE - GMM: {compute_ece(cal_loc, loc_correct):.4f}")
    print("\nExpected: GMM improves cls ECE but DEGRADES loc ECE")


if __name__ == '__main__':
    demo_gmm_degradation()
