"""
Detection Calibration Benchmark
================================
Measures calibration quality for object detectors on dense retail scenes.

Adapted from classification ECE (Guo et al. 2017) to detection domain
following D-ECE framework (Küppers et al. WACV 2024).

Key difference from classification calibration:
- Each DETECTION BOX has a confidence score
- A detection is "correct" if IoU with ground truth > threshold
- D-ECE measures gap between confidence and precision at box level

Metrics computed:
- D-ECE: Detection Expected Calibration Error
- MCE: Maximum Calibration Error
- Brier Score: Mean squared error of confidences
- Reliability Diagram: Visual calibration curve

Author: Calibrated Dense Detection Paper
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class DetectionCalibrationMetrics:
    """Container for detection calibration metrics."""
    d_ece: float        # Detection Expected Calibration Error
    mce: float          # Maximum Calibration Error
    brier: float        # Brier Score
    accuracy: float     # Overall precision (TP / (TP + FP))
    avg_confidence: float
    num_detections: int
    num_correct: int
    bin_accuracies: list
    bin_confidences: list
    bin_counts: list
    
    def to_dict(self) -> Dict:
        return {
            'd_ece': self.d_ece,
            'mce': self.mce,
            'brier': self.brier,
            'accuracy': self.accuracy,
            'avg_confidence': self.avg_confidence,
            'num_detections': self.num_detections,
            'num_correct': self.num_correct
        }


def compute_iou(box1: list, box2: list) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / (union + 1e-6)


def match_detections_to_gt(
    predictions: List[Dict],
    gt_labels_dir: str,
    images_dir: str,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to ground truth and determine correctness.
    
    For each detection box, determine if it's a True Positive (IoU > threshold
    with a GT box) or False Positive.
    
    Args:
        predictions: List of dicts with 'image', 'box', 'confidence'
        gt_labels_dir: Path to YOLO format labels directory
        images_dir: Path to images directory (for image dimensions)
        iou_threshold: IoU threshold for matching
    
    Returns:
        confidences: array of confidence scores
        correctness: array of 0/1 (1 = TP, 0 = FP)
    """
    from PIL import Image
    
    # Group predictions by image
    preds_by_image = {}
    for pred in predictions:
        img_name = pred['image']
        if img_name not in preds_by_image:
            preds_by_image[img_name] = []
        preds_by_image[img_name].append(pred)
    
    all_confidences = []
    all_correctness = []
    
    for img_name, img_preds in preds_by_image.items():
        # Load ground truth (YOLO format: class cx cy w h)
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(gt_labels_dir, label_name)
        
        gt_boxes = []
        if os.path.exists(label_path):
            # Get image dimensions to convert YOLO format to absolute coords
            img_path = os.path.join(images_dir, img_name)
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception:
                img_w, img_h = 640, 640  # fallback
            
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        # Convert YOLO (normalized cx, cy, w, h) to absolute (x1, y1, x2, y2)
                        x1 = (cx - w/2) * img_w
                        y1 = (cy - h/2) * img_h
                        x2 = (cx + w/2) * img_w
                        y2 = (cy + h/2) * img_h
                        gt_boxes.append([x1, y1, x2, y2])
        
        # Sort predictions by confidence (highest first)
        img_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Track which GT boxes are already matched
        gt_matched = [False] * len(gt_boxes)
        
        for pred in img_preds:
            conf = pred['confidence']
            pred_box = pred['box']
            
            # Find best matching GT box
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                correct = 1
                gt_matched[best_gt_idx] = True
            else:
                correct = 0
            
            all_confidences.append(conf)
            all_correctness.append(correct)
    
    return np.array(all_confidences), np.array(all_correctness)


def compute_detection_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15
) -> DetectionCalibrationMetrics:
    """
    Compute Detection Expected Calibration Error (D-ECE).
    
    D-ECE = Σ_m (|B_m|/n) |precision(B_m) - conf(B_m)|
    
    Where B_m are equal-width confidence bins, precision is TP/(TP+FP)
    within each bin, and conf is the mean confidence in each bin.
    
    Args:
        confidences: Per-detection confidence scores
        correctness: Per-detection correctness labels (0 or 1)
        n_bins: Number of equal-width bins
    
    Returns:
        DetectionCalibrationMetrics
    """
    n = len(confidences)
    if n == 0:
        return DetectionCalibrationMetrics(
            d_ece=0, mce=0, brier=0, accuracy=0,
            avg_confidence=0, num_detections=0, num_correct=0,
            bin_accuracies=[], bin_confidences=[], bin_counts=[]
        )
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    d_ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Get detections in this bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        
        count = in_bin.sum()
        
        if count > 0:
            bin_acc = correctness[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            gap = abs(float(bin_acc) - float(bin_conf))
            
            d_ece += (count / n) * gap
            mce = max(mce, gap)
            
            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
        
        bin_counts.append(int(count))
    
    # Brier score for detection: E[(conf - correct)^2]
    brier = float(np.mean((confidences - correctness) ** 2))
    
    accuracy = float(correctness.mean())
    avg_confidence = float(confidences.mean())
    
    return DetectionCalibrationMetrics(
        d_ece=float(d_ece),
        mce=float(mce),
        brier=brier,
        accuracy=accuracy,
        avg_confidence=avg_confidence,
        num_detections=n,
        num_correct=int(correctness.sum()),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts
    )


def plot_reliability_diagram(
    metrics: DetectionCalibrationMetrics,
    title: str = "Detection Reliability Diagram",
    save_path: Optional[str] = None
):
    """
    Create reliability diagram for object detection calibration.
    
    Shows per-bin precision vs confidence, with gap visualization.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_bins = len(metrics.bin_accuracies)
    bin_centers = np.linspace(0.5/n_bins, 1 - 0.5/n_bins, n_bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # === Reliability Diagram ===
    accs = np.array(metrics.bin_accuracies)
    confs = np.array(metrics.bin_confidences)
    counts = np.array(metrics.bin_counts)
    
    # Only plot bins with data
    mask = counts > 0
    
    ax1.bar(bin_centers, accs, width=1/n_bins, alpha=0.6,
            color='#2196F3', edgecolor='black', label='Precision')
    
    gap = confs - accs
    ax1.bar(bin_centers, gap, bottom=accs, width=1/n_bins, alpha=0.3,
            color='#F44336', edgecolor='black', label='Gap (overconfidence)')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    
    ax1.set_xlabel('Confidence', fontsize=13)
    ax1.set_ylabel('Precision', fontsize=13)
    ax1.set_title(f'{title}\nD-ECE = {metrics.d_ece:.4f} | MCE = {metrics.mce:.4f}', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # === Confidence Distribution ===
    ax2.bar(bin_centers, counts, width=1/n_bins, alpha=0.7,
            color='#4CAF50', edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=13)
    ax2.set_ylabel('Number of Detections', fontsize=13)
    ax2.set_title(f'Confidence Distribution\n'
                  f'Avg conf: {metrics.avg_confidence:.3f} | '
                  f'Precision: {metrics.accuracy:.3f}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close(fig)
    return fig


def apply_temperature_scaling(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, np.ndarray]:
    """
    Find optimal temperature T that minimizes D-ECE.
    
    Applies: scaled_conf = conf^(1/T) / (conf^(1/T) + (1-conf)^(1/T))
    
    This is equivalent to temperature scaling in logit space.
    
    Returns:
        best_T: Optimal temperature
        scaled_confidences: Calibrated confidence scores
    """
    best_ece = float('inf')
    best_T = 1.0
    
    for T in np.arange(0.1, 5.0, 0.05):
        # Apply temperature in logit space
        logits = np.log(confidences / (1 - confidences + 1e-10) + 1e-10)
        scaled_logits = logits / T
        scaled_conf = 1 / (1 + np.exp(-scaled_logits))
        scaled_conf = np.clip(scaled_conf, 1e-6, 1 - 1e-6)
        
        metrics = compute_detection_ece(scaled_conf, correctness, n_bins)
        
        if metrics.d_ece < best_ece:
            best_ece = metrics.d_ece
            best_T = T
    
    # Apply best temperature
    logits = np.log(confidences / (1 - confidences + 1e-10) + 1e-10)
    scaled_logits = logits / best_T
    scaled_conf = 1 / (1 + np.exp(-scaled_logits))
    scaled_conf = np.clip(scaled_conf, 1e-6, 1 - 1e-6)
    
    return best_T, scaled_conf


def apply_platt_scaling(
    confidences: np.ndarray,
    correctness: np.ndarray
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Platt scaling: fit logistic regression on confidences.
    
    Learns a, b such that: calibrated = sigmoid(a * logit(conf) + b)
    """
    from sklearn.linear_model import LogisticRegression
    
    logits = np.log(confidences / (1 - confidences + 1e-10) + 1e-10)
    logits = logits.reshape(-1, 1)
    
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(logits, correctness.astype(int))
    
    scaled_conf = lr.predict_proba(logits)[:, 1]
    scaled_conf = np.clip(scaled_conf, 1e-6, 1 - 1e-6)
    
    return (float(lr.coef_[0][0]), float(lr.intercept_[0])), scaled_conf


def apply_isotonic_regression(
    confidences: np.ndarray,
    correctness: np.ndarray
) -> np.ndarray:
    """
    Isotonic regression: non-parametric calibration.
    
    Fits a monotonically increasing function mapping confidence → calibrated probability.
    """
    from sklearn.isotonic import IsotonicRegression
    
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(confidences, correctness)
    
    scaled_conf = ir.predict(confidences)
    return np.clip(scaled_conf, 1e-6, 1 - 1e-6)


def run_calibration_benchmark(
    predictions_file: str,
    gt_labels_dir: str,
    images_dir: str,
    output_dir: str,
    model_name: str = "detector",
    iou_threshold: float = 0.5,
    n_bins: int = 15
) -> Dict:
    """
    Run the complete calibration benchmark on a detector's predictions.
    
    This is the main entry point for Week 2 experiments.
    
    Args:
        predictions_file: Path to JSON with predictions from train_baselines.py
        gt_labels_dir: Path to YOLO format ground truth labels
        images_dir: Path to test images
        output_dir: Where to save results
        model_name: Name for labeling
        iou_threshold: IoU threshold for TP/FP determination
        n_bins: Number of calibration bins
    
    Returns:
        Dict with all metrics and calibration method results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"CALIBRATION BENCHMARK: {model_name.upper()}")
    print(f"{'='*70}")
    
    # Load predictions
    with open(predictions_file) as f:
        data = json.load(f)
    
    predictions = data['predictions']
    print(f"Loaded {len(predictions)} detections from {data['num_test_images']} images")
    
    # Match to ground truth
    print(f"Matching detections to GT (IoU > {iou_threshold})...")
    confidences, correctness = match_detections_to_gt(
        predictions, gt_labels_dir, images_dir, iou_threshold
    )
    
    print(f"  Total detections: {len(confidences)}")
    print(f"  True positives:   {int(correctness.sum())}")
    print(f"  False positives:  {int((1 - correctness).sum())}")
    print(f"  Precision:        {correctness.mean():.4f}")
    
    # 1. Uncalibrated metrics
    print(f"\n--- Uncalibrated ---")
    uncalib = compute_detection_ece(confidences, correctness, n_bins)
    print(f"  D-ECE: {uncalib.d_ece:.4f}")
    print(f"  MCE:   {uncalib.mce:.4f}")
    print(f"  Brier: {uncalib.brier:.4f}")
    print(f"  Avg confidence: {uncalib.avg_confidence:.4f}")
    print(f"  Precision:      {uncalib.accuracy:.4f}")
    
    plot_reliability_diagram(
        uncalib,
        title=f"{model_name} — Uncalibrated",
        save_path=os.path.join(output_dir, f"{model_name}_uncalibrated.png")
    )
    
    results = {
        'model': model_name,
        'iou_threshold': iou_threshold,
        'num_detections': len(confidences),
        'uncalibrated': uncalib.to_dict(),
        'calibration_methods': {}
    }
    
    # 2. Temperature Scaling
    print(f"\n--- Temperature Scaling ---")
    best_T, temp_conf = apply_temperature_scaling(confidences, correctness, n_bins)
    temp_metrics = compute_detection_ece(temp_conf, correctness, n_bins)
    print(f"  Optimal T: {best_T:.2f}")
    print(f"  D-ECE: {temp_metrics.d_ece:.4f} (Δ = {temp_metrics.d_ece - uncalib.d_ece:+.4f})")
    print(f"  Brier: {temp_metrics.brier:.4f} (Δ = {temp_metrics.brier - uncalib.brier:+.4f})")
    
    plot_reliability_diagram(
        temp_metrics,
        title=f"{model_name} — Temperature Scaling (T={best_T:.2f})",
        save_path=os.path.join(output_dir, f"{model_name}_temp_scaling.png")
    )
    
    results['calibration_methods']['temperature_scaling'] = {
        'temperature': best_T,
        **temp_metrics.to_dict()
    }
    
    # 3. Platt Scaling
    print(f"\n--- Platt Scaling ---")
    try:
        platt_params, platt_conf = apply_platt_scaling(confidences, correctness)
        platt_metrics = compute_detection_ece(platt_conf, correctness, n_bins)
        print(f"  Params: a={platt_params[0]:.4f}, b={platt_params[1]:.4f}")
        print(f"  D-ECE: {platt_metrics.d_ece:.4f} (Δ = {platt_metrics.d_ece - uncalib.d_ece:+.4f})")
        print(f"  Brier: {platt_metrics.brier:.4f} (Δ = {platt_metrics.brier - uncalib.brier:+.4f})")
        
        plot_reliability_diagram(
            platt_metrics,
            title=f"{model_name} — Platt Scaling",
            save_path=os.path.join(output_dir, f"{model_name}_platt_scaling.png")
        )
        
        results['calibration_methods']['platt_scaling'] = {
            'a': platt_params[0], 'b': platt_params[1],
            **platt_metrics.to_dict()
        }
    except Exception as e:
        print(f"  ⚠ Platt scaling failed: {e}")
    
    # 4. Isotonic Regression
    print(f"\n--- Isotonic Regression ---")
    try:
        iso_conf = apply_isotonic_regression(confidences, correctness)
        iso_metrics = compute_detection_ece(iso_conf, correctness, n_bins)
        print(f"  D-ECE: {iso_metrics.d_ece:.4f} (Δ = {iso_metrics.d_ece - uncalib.d_ece:+.4f})")
        print(f"  Brier: {iso_metrics.brier:.4f} (Δ = {iso_metrics.brier - uncalib.brier:+.4f})")
        
        plot_reliability_diagram(
            iso_metrics,
            title=f"{model_name} — Isotonic Regression",
            save_path=os.path.join(output_dir, f"{model_name}_isotonic.png")
        )
        
        results['calibration_methods']['isotonic_regression'] = iso_metrics.to_dict()
    except Exception as e:
        print(f"  ⚠ Isotonic regression failed: {e}")
    
    # Save results
    results_file = os.path.join(output_dir, f"{model_name}_calibration.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_file}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"CALIBRATION SUMMARY: {model_name.upper()}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'D-ECE':>8} {'MCE':>8} {'Brier':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'Uncalibrated':<25} {uncalib.d_ece:>8.4f} {uncalib.mce:>8.4f} {uncalib.brier:>8.4f}")
    for method, m in results['calibration_methods'].items():
        name = method.replace('_', ' ').title()
        print(f"{name:<25} {m['d_ece']:>8.4f} {m['mce']:>8.4f} {m['brier']:>8.4f}")
    
    return results
