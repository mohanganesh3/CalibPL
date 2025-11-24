"""
Ensemble classifiers: XGBoost, Random Forest, SVM.
Implements Section 3.3 of the paper.

Each detection model has 3 ensemble classifiers trained on its features.
Total: 6 classifiers (3 for FRCNN, 3 for YOLO)
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

logger = logging.getLogger(__name__)


class EnsembleClassifiers:
    """
    Ensemble classifiers for pseudo-label generation.
    Paper Section 3.3: "Each model passes its output to ensemble classifiers"
    
    Classifiers:
    1. XGBoost: Gradient boosting for high accuracy
    2. Random Forest: Bagging ensemble for robustness
    3. SVM: Support Vector Machine for margin-based classification
    """
    
    def __init__(
        self,
        model_type: str = 'frcnn',  # 'frcnn' or 'yolo'
        random_state: int = 42
    ):
        """
        Initialize ensemble classifiers.
        
        Args:
            model_type: Which detection model these classifiers are for
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        
        # Initialize classifiers with paper's hyperparameters
        # These will be optimized in Phase 5
        self.xgboost = xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=6,
            reg_alpha=0.1,
            n_estimators=100,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.random_forest = RandomForestClassifier(
            max_depth=20,
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,  # Important: need probabilities for confidence filtering
            random_state=random_state
        )
        
        self.is_trained = {
            'xgboost': False,
            'random_forest': False,
            'svm': False
        }
    
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        classifiers: List[str] = ['xgboost', 'random_forest', 'svm']
    ):
        """
        Train ensemble classifiers on extracted features.
        
        Paper Section 3.3: "Following feature extraction, each model passes its
        output to a series of ensemble classifiers"
        
        Args:
            features: Extracted features (N, feature_dim)
            labels: Ground truth labels (N,)
            classifiers: Which classifiers to train
        """
        logger.info(f"Training {self.model_type} ensemble classifiers")
        logger.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        
        if 'xgboost' in classifiers:
            logger.info("Training XGBoost...")
            self.xgboost.fit(features, labels)
            self.is_trained['xgboost'] = True
            logger.info("XGBoost training complete")
        
        if 'random_forest' in classifiers:
            logger.info("Training Random Forest...")
            self.random_forest.fit(features, labels)
            self.is_trained['random_forest'] = True
            logger.info("Random Forest training complete")
        
        if 'svm' in classifiers:
            logger.info("Training SVM...")
            self.svm.fit(features, labels)
            self.is_trained['svm'] = True
            logger.info("SVM training complete")
        
        logger.info("All ensemble classifiers trained successfully")
    
    def predict_with_confidence(
        self,
        features: np.ndarray,
        classifiers: List[str] = ['xgboost', 'random_forest', 'svm']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence scores.
        
        Paper Section 3.4: "High-confidence predictions (>0.7) are used as pseudo-labels"
        
        Uses voting ensemble:
        - Each classifier votes with its confidence
        - Final prediction is weighted average
        - Confidence is the agreement level among classifiers
        
        Args:
            features: Input features (N, feature_dim)
            classifiers: Which classifiers to use
            
        Returns:
            predictions: Class predictions (N,)
            confidences: Confidence scores (N,) in [0, 1]
        """
        if not any(self.is_trained[c] for c in classifiers):
            raise ValueError("No classifiers have been trained")
        
        logger.info(f"Generating pseudo-labels with {len(classifiers)} classifiers")
        
        # Collect predictions and probabilities from each classifier
        all_predictions = []
        all_probabilities = []
        
        if 'xgboost' in classifiers and self.is_trained['xgboost']:
            pred = self.xgboost.predict(features)
            proba = self.xgboost.predict_proba(features)
            all_predictions.append(pred)
            all_probabilities.append(proba)
        
        if 'random_forest' in classifiers and self.is_trained['random_forest']:
            pred = self.random_forest.predict(features)
            proba = self.random_forest.predict_proba(features)
            all_predictions.append(pred)
            all_probabilities.append(proba)
        
        if 'svm' in classifiers and self.is_trained['svm']:
            pred = self.svm.predict(features)
            proba = self.svm.predict_proba(features)
            all_predictions.append(pred)
            all_probabilities.append(proba)
        
        # Ensemble voting
        # Average the probability predictions
        avg_probabilities = np.mean(all_probabilities, axis=0)
        
        # Final prediction is the class with highest average probability
        final_predictions = np.argmax(avg_probabilities, axis=1)
        
        # Confidence is the maximum probability (measures agreement)
        confidences = np.max(avg_probabilities, axis=1)
        
        logger.info(f"Generated {len(final_predictions)} pseudo-labels")
        logger.info(f"Mean confidence: {confidences.mean():.3f}")
        logger.info(f"High-confidence (>0.7): {(confidences > 0.7).sum()} / {len(confidences)}")
        
        return final_predictions, confidences
    
    def filter_high_confidence(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter predictions by confidence threshold.
        
        Paper Section 3.4: "Only high-confidence predictions (>0.7) are used
        as pseudo-labels for the co-training process"
        
        Args:
            predictions: Class predictions (N,)
            confidences: Confidence scores (N,)
            threshold: Confidence threshold (default: 0.7 from paper)
            
        Returns:
            filtered_indices: Indices of high-confidence samples
            filtered_predictions: High-confidence predictions
        """
        high_conf_mask = confidences > threshold
        filtered_indices = np.where(high_conf_mask)[0]
        filtered_predictions = predictions[high_conf_mask]
        
        logger.info(f"Filtered {len(filtered_indices)} / {len(predictions)} "
                   f"high-confidence predictions (threshold={threshold})")
        
        return filtered_indices, filtered_predictions
    
    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate ensemble classifiers.
        
        Returns:
            metrics: Dict with accuracy for each classifier
        """
        from sklearn.metrics import accuracy_score
        
        metrics = {}
        
        if self.is_trained['xgboost']:
            pred = self.xgboost.predict(features)
            metrics['xgboost_accuracy'] = accuracy_score(labels, pred)
        
        if self.is_trained['random_forest']:
            pred = self.random_forest.predict(features)
            metrics['random_forest_accuracy'] = accuracy_score(labels, pred)
        
        if self.is_trained['svm']:
            pred = self.svm.predict(features)
            metrics['svm_accuracy'] = accuracy_score(labels, pred)
        
        # Ensemble accuracy
        pred_ensemble, _ = self.predict_with_confidence(features)
        metrics['ensemble_accuracy'] = accuracy_score(labels, pred_ensemble)
        
        logger.info(f"{self.model_type} Ensemble Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return metrics
    
    def save(self, output_dir: str):
        """Save trained classifiers to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.is_trained['xgboost']:
            joblib.dump(self.xgboost, output_dir / f'{self.model_type}_xgboost.pkl')
        if self.is_trained['random_forest']:
            joblib.dump(self.random_forest, output_dir / f'{self.model_type}_random_forest.pkl')
        if self.is_trained['svm']:
            joblib.dump(self.svm, output_dir / f'{self.model_type}_svm.pkl')
        
        logger.info(f"Ensemble classifiers saved to {output_dir}")
    
    def load(self, input_dir: str):
        """Load trained classifiers from disk."""
        input_dir = Path(input_dir)
        
        xgb_path = input_dir / f'{self.model_type}_xgboost.pkl'
        rf_path = input_dir / f'{self.model_type}_random_forest.pkl'
        svm_path = input_dir / f'{self.model_type}_svm.pkl'
        
        if xgb_path.exists():
            self.xgboost = joblib.load(xgb_path)
            self.is_trained['xgboost'] = True
        
        if rf_path.exists():
            self.random_forest = joblib.load(rf_path)
            self.is_trained['random_forest'] = True
        
        if svm_path.exists():
            self.svm = joblib.load(svm_path)
            self.is_trained['svm'] = True
        
        logger.info(f"Ensemble classifiers loaded from {input_dir}")
        logger.info(f"Trained classifiers: {[k for k, v in self.is_trained.items() if v]}")


class CoTrainingEnsemble:
    """
    Combined ensemble for co-training.
    Manages both FRCNN and YOLO ensembles.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize both ensembles."""
        self.frcnn_ensemble = EnsembleClassifiers('frcnn', random_state)
        self.yolo_ensemble = EnsembleClassifiers('yolo', random_state)
    
    def train_all(
        self,
        frcnn_features: np.ndarray,
        yolo_features: np.ndarray,
        labels: np.ndarray
    ):
        """Train both ensembles."""
        logger.info("Training FRCNN ensemble...")
        self.frcnn_ensemble.train(frcnn_features, labels)
        
        logger.info("Training YOLO ensemble...")
        self.yolo_ensemble.train(yolo_features, labels)
        
        logger.info("All ensembles trained successfully")
    
    def generate_pseudo_labels(
        self,
        frcnn_features: np.ndarray,
        yolo_features: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Tuple[Dict, Dict]:
        """
        Generate pseudo-labels from both ensembles.
        
        Returns:
            frcnn_pseudo: Dict with 'indices' and 'labels' for high-confidence FRCNN predictions
            yolo_pseudo: Dict with 'indices' and 'labels' for high-confidence YOLO predictions
        """
        # Generate predictions from FRCNN ensemble
        frcnn_preds, frcnn_confs = self.frcnn_ensemble.predict_with_confidence(frcnn_features)
        frcnn_indices, frcnn_labels = self.frcnn_ensemble.filter_high_confidence(
            frcnn_preds, frcnn_confs, confidence_threshold
        )
        
        # Generate predictions from YOLO ensemble
        yolo_preds, yolo_confs = self.yolo_ensemble.predict_with_confidence(yolo_features)
        yolo_indices, yolo_labels = self.yolo_ensemble.filter_high_confidence(
            yolo_preds, yolo_confs, confidence_threshold
        )
        
        frcnn_pseudo = {'indices': frcnn_indices, 'labels': frcnn_labels}
        yolo_pseudo = {'indices': yolo_indices, 'labels': yolo_labels}
        
        return frcnn_pseudo, yolo_pseudo
    
    def save(self, output_dir: str):
        """Save both ensembles."""
        self.frcnn_ensemble.save(output_dir)
        self.yolo_ensemble.save(output_dir)
    
    def load(self, input_dir: str):
        """Load both ensembles."""
        self.frcnn_ensemble.load(input_dir)
        self.yolo_ensemble.load(input_dir)
