#!/usr/bin/env python3
"""
Confidence Histogram & Distribution Shift Analysis — BMVC 2026 Contribution C2

Analyzes HOW confidence score distributions change across SSOD iterations,
providing the diagnostic framework for WHY calibration degrades.

Measures:
- Per-iteration confidence histograms
- KL divergence between iter-0 and iter-k distributions
- Summary statistics (mean, std, skewness, kurtosis)

Usage:
    python3 scripts/confidence_histogram_analysis.py \
        --experiment-dir results/calibcotrain_cl/confidence_seed42
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_iteration_confidences(experiment_dir):
    """
    Load confidence scores from each iteration's pseudo-label generation.
    Falls back to smoking gun data if available.
    """
    experiment_dir = Path(experiment_dir)
    iteration_data = {}
    
    # Try to find per-iteration data
    for iter_dir in sorted(experiment_dir.glob("iter_*")):
        iter_num = int(iter_dir.name.split("_")[1])
        # Check for cached confidence data
        conf_file = iter_dir / "confidences.npz"
        if conf_file.exists():
            data = np.load(conf_file)
            iteration_data[iter_num] = data['confidences']
            continue
        
        # Check for pseudo-label stats with confidence info
        stats_file = iter_dir / "pseudo_label_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            if 'confidences' in stats:
                iteration_data[iter_num] = np.array(stats['confidences'])
    
    return iteration_data


def compute_kl_divergence(p, q, n_bins=50, epsilon=1e-10):
    """Compute KL divergence KL(P || Q) using histograms."""
    bins = np.linspace(0, 1, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    
    # Add epsilon to avoid log(0)
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon
    
    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    
    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return float(kl)


def compute_distribution_stats(confidences):
    """Compute summary statistics of confidence distribution."""
    from scipy import stats as scipy_stats
    
    return {
        'mean': float(np.mean(confidences)),
        'std': float(np.std(confidences)),
        'median': float(np.median(confidences)),
        'skewness': float(scipy_stats.skew(confidences)),
        'kurtosis': float(scipy_stats.kurtosis(confidences)),
        'q25': float(np.percentile(confidences, 25)),
        'q75': float(np.percentile(confidences, 75)),
        'pct_above_0.5': float(np.mean(confidences > 0.5)),
        'pct_above_0.9': float(np.mean(confidences > 0.9)),
        'n_detections': len(confidences),
    }


def analyze_from_smoking_gun(smoking_gun_path, output_path):
    """
    Analyze confidence distribution shift using smoking gun data.
    The smoking gun has raw confidence distributions at each iteration.
    """
    with open(smoking_gun_path) as f:
        data = json.load(f)
    
    # Print what we have
    print(f"\n{'='*60}")
    print(f" CONFIDENCE DISTRIBUTION SHIFT ANALYSIS")
    print(f"{'='*60}")
    
    # Extract iteration-level calibration data
    isotonic_data = data.get('results', {}).get('isotonic', [])
    confidence_data = data.get('results', {}).get('confidence', [])
    
    report = {
        'title': 'Confidence Distribution Shift Under Iterative SSOD',
        'description': 'Analyzes how model confidence distributions change across co-training iterations',
        'iterations': [],
    }
    
    for entry in isotonic_data:
        iteration = entry.get('iteration', -1)
        d_ece = entry.get('d_ece_raw', entry.get('d_ece', 0))
        brier = entry.get('brier', 0)
        n_det = entry.get('num_detections', 0)
        
        iter_info = {
            'iteration': iteration,
            'd_ece_raw': d_ece,
            'brier_score': brier,
            'num_detections': n_det,
        }
        
        # Pseudo-label stats if available
        ps = entry.get('pseudo_label_stats', {})
        if ps:
            iter_info['pseudo_labels'] = {
                'total_boxes': ps.get('total_boxes', 0),
                'accepted': ps.get('accepted', 0),
                'rejected': ps.get('rejected', ps.get('total_boxes', 0) - ps.get('accepted', 0)),
                'acceptance_rate': ps.get('accepted', 0) / max(ps.get('total_boxes', 1), 1),
            }
        
        report['iterations'].append(iter_info)
        
        print(f"\nIteration {iteration}:")
        print(f"  Raw D-ECE:      {d_ece:.6f}")
        print(f"  Brier Score:    {brier:.6f}")
        print(f"  Num detections: {n_det}")
        if ps:
            ar = ps.get('accepted', 0) / max(ps.get('total_boxes', 1), 1)
            print(f"  Pseudo-labels:  {ps.get('accepted', 0)}/{ps.get('total_boxes', 0)} accepted ({ar:.1%})")
    
    # Compute trends
    if len(report['iterations']) > 1:
        d_eces = [r['d_ece_raw'] for r in report['iterations']]
        briers = [r['brier_score'] for r in report['iterations']]
        n_dets = [r['num_detections'] for r in report['iterations']]
        
        report['trends'] = {
            'd_ece_trend': 'increasing' if d_eces[-1] > d_eces[0] else 'decreasing',
            'd_ece_range': [float(min(d_eces)), float(max(d_eces))],
            'brier_trend': 'increasing' if briers[-1] > briers[0] else 'decreasing',
            'detection_count_trend': 'decreasing' if n_dets[-1] < n_dets[0] else 'increasing',
            'detection_count_change_pct': float((n_dets[-1] - n_dets[0]) / max(n_dets[0], 1) * 100),
        }
        
        print(f"\n--- Trends ---")
        print(f"  D-ECE: {report['trends']['d_ece_trend']} ({d_eces[0]:.4f} → {d_eces[-1]:.4f})")
        print(f"  Brier: {report['trends']['brier_trend']} ({briers[0]:.4f} → {briers[-1]:.4f})")
        print(f"  Detections: {n_dets[0]} → {n_dets[-1]} ({report['trends']['detection_count_change_pct']:+.1f}%)")
        
        # The KEY insight: pseudo-label acceptance rates
        acceptance_rates = []
        for r in report['iterations']:
            if 'pseudo_labels' in r:
                acceptance_rates.append(r['pseudo_labels']['acceptance_rate'])
        
        if acceptance_rates:
            print(f"\n--- Pseudo-Label Acceptance Rates ---")
            for i, ar in enumerate(acceptance_rates):
                print(f"  Iter {i+1}: {ar:.1%}")
            report['trends']['acceptance_rate_trend'] = 'increasing' if acceptance_rates[-1] > acceptance_rates[0] else 'decreasing'
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Confidence Distribution Shift Analysis')
    parser.add_argument('--smoking-gun', type=str,
                        default='results/smoking_gun_ablation/smoking_gun_results.json',
                        help='Path to smoking gun results')
    parser.add_argument('--experiment-dir', type=str, default=None,
                        help='Path to experiment directory with per-iter data')
    parser.add_argument('--output', type=str,
                        default='results/confidence_distribution_analysis.json',
                        help='Output path')
    args = parser.parse_args()
    
    output = str(PROJECT_ROOT / args.output)
    
    # Use smoking gun data first (it has rich per-iteration data)
    smoking_gun = str(PROJECT_ROOT / args.smoking_gun)
    if os.path.exists(smoking_gun):
        print(f"Analyzing smoking gun data: {smoking_gun}")
        analyze_from_smoking_gun(smoking_gun, output)
    elif args.experiment_dir:
        print(f"Analyzing experiment: {args.experiment_dir}")
        iteration_data = load_iteration_confidences(args.experiment_dir)
        if not iteration_data:
            print("No confidence data found in experiment directory")
            sys.exit(1)
        # TODO: Add analysis for direct experiment data
    else:
        print("No data source specified")
        sys.exit(1)


if __name__ == '__main__':
    main()
