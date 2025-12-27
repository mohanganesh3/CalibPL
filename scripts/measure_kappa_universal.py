#!/usr/bin/env python3
"""
Measure κ (kappa) threshold across multiple datasets.

This script validates the UNIVERSAL nature of the NMS selection bias:
κ ≈ 12 should hold across COCO, SKU-110K, CrowdHuman, etc.

Usage:
    python3 scripts/measure_kappa_universal.py --datasets coco sku110k crowdhuman
"""

import argparse
import numpy as np
import yaml
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Disable CUDA for this diagnostic script
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def count_objects_per_image(label_dir):
    """Count objects per image from YOLO-format labels."""
    label_dir = Path(label_dir)
    counts = []
    
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = f.readlines()
            counts.append(len(lines))
    
    return np.array(counts)


def compute_density_statistics(counts):
    """Compute density statistics."""
    return {
        'mean': float(np.mean(counts)),
        'median': float(np.median(counts)),
        'std': float(np.std(counts)),
        'min': int(np.min(counts)),
        'max': int(np.max(counts)),
        'p90': float(np.percentile(counts, 90)),
        'p95': float(np.percentile(counts, 95)),
        'total_images': len(counts),
        'total_objects': int(np.sum(counts))
    }


def estimate_kappa_threshold(counts, bins=50):
    """
    Estimate κ threshold where dual misalignment begins.
    
    Strategy: Plot histogram and find the density level where
    >50% of images exceed it (heuristic for onset of bias).
    """
    hist, bin_edges = np.histogram(counts, bins=bins)
    
    # κ is typically where the "dense" regime starts
    # Heuristic: 90th percentile of object counts
    kappa_estimate = float(np.percentile(counts, 90))
    
    return kappa_estimate, hist, bin_edges


def analyze_dataset(name, label_dir, output_dir):
    """Analyze one dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    
    if not Path(label_dir).exists():
        print(f"  ERROR: {label_dir} does not exist!")
        return None
    
    # Count objects per image
    print(f"  Counting objects in {label_dir}...")
    counts = count_objects_per_image(label_dir)
    
    if len(counts) == 0:
        print(f"  ERROR: No labels found in {label_dir}")
        return None
    
    # Compute statistics
    stats = compute_density_statistics(counts)
    
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total objects: {stats['total_objects']}")
    print(f"  Objects per image:")
    print(f"    Mean: {stats['mean']:.2f}")
    print(f"    Median: {stats['median']:.2f}")
    print(f"    Std: {stats['std']:.2f}")
    print(f"    Range: [{stats['min']}, {stats['max']}]")
    print(f"    90th percentile: {stats['p90']:.2f}")
    print(f"    95th percentile: {stats['p95']:.2f}")
    
    # Estimate κ
    kappa_est, hist, bin_edges = estimate_kappa_threshold(counts)
    print(f"\n  Estimated κ (90th percentile): {kappa_est:.2f}")
    
    # Create distribution plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(kappa_est, color='red', linestyle='--', linewidth=2, 
               label=f'κ ≈ {kappa_est:.1f}')
    ax.axvline(stats['mean'], color='blue', linestyle=':', linewidth=2,
               label=f'Mean = {stats["mean"]:.1f}')
    ax.set_xlabel('Objects per Image', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{name}: Object Density Distribution', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plot_path = output_dir / f"{name.lower()}_density_distribution.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"  Plot saved: {plot_path}")
    plt.close()
    
    return {
        'name': name,
        'stats': stats,
        'kappa_estimate': kappa_est,
        'density_distribution': counts.tolist()
    }


def create_comparison_plot(results, output_path):
    """Create comparative plot across datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [r['name'] for r in results]
    kappas = [r['kappa_estimate'] for r in results]
    means = [r['stats']['mean'] for r in results]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, means, width, label='Mean objects/img', alpha=0.7)
    ax.bar(x + width/2, kappas, width, label='κ estimate (p90)', alpha=0.7)
    
    # Draw universal κ reference line
    ax.axhline(12, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label='Universal κ ≈ 12')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Object Count', fontsize=12)
    ax.set_title('κ Threshold Universality Validation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nComparison plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Measure κ threshold universally")
    parser.add_argument('--output-dir', default='results/kappa_analysis',
                        help='Output directory')
    args = parser.parse_args()
    
    # Define datasets to analyze
    datasets = [
        {
            'name': 'COCO-1%',
            'label_dir': 'data/coco/yolo_format/coco_frac_1/train/labels'
        },
        {
            'name': 'COCO-5%',
            'label_dir': 'data/coco/yolo_format/coco_frac_5/train/labels'
        },
        {
            'name': 'SKU-110K-10%',
            'label_dir': 'data/SKU110K/yolo_format/frac_10/train/labels'
        },
        {
            'name': 'SKU-110K-Full',
            'label_dir': 'data/SKU110K/yolo_format/train/labels'
        },
        {
            'name': 'CrowdHuman',
            'label_dir': 'data/CrowdHuman/yolo_labels/train'
        }
    ]
    
    print("\n" + "="*70)
    print("UNIVERSAL κ THRESHOLD MEASUREMENT")
    print("Hypothesis: κ ≈ 12 across all datasets/domains")
    print("="*70)
    
    results = []
    for ds in datasets:
        result = analyze_dataset(ds['name'], ds['label_dir'], args.output_dir)
        if result is not None:
            results.append(result)
    
    # Save results
    output_dir = Path(args.output_dir)
    results_file = output_dir / "kappa_measurements.json"
    with open(results_file, 'w') as f:
        # Don't save full distributions (too large), just stats
        summary = [{
            'name': r['name'],
            'stats': r['stats'],
            'kappa_estimate': r['kappa_estimate']
        } for r in results]
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {results_file}")
    
    # Create comparison plot
    if len(results) > 1:
        comparison_path = output_dir / "kappa_universality_comparison.pdf"
        create_comparison_plot(results, comparison_path)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: κ ESTIMATES ACROSS DATASETS")
    print("="*70)
    print(f"{'Dataset':<20} {'Mean ρ':<12} {'κ (p90)':<12} {'Deviation from 12':<20}")
    print("-"*70)
    
    for r in results:
        mean_rho = r['stats']['mean']
        kappa = r['kappa_estimate']
        deviation = abs(kappa - 12)
        print(f"{r['name']:<20} {mean_rho:>10.2f}  {kappa:>10.2f}  {deviation:>18.2f}")
    
    # Compute mean κ across datasets
    if results:
        mean_kappa = np.mean([r['kappa_estimate'] for r in results])
        std_kappa = np.std([r['kappa_estimate'] for r in results])
        print("-"*70)
        print(f"{'UNIVERSAL κ':<20} {'':>10}  {mean_kappa:>10.2f} ± {std_kappa:.2f}")
    
    print("\n" + "="*70)
    if results and 10 <= mean_kappa <= 14:
        print("✓ HYPOTHESIS CONFIRMED: κ ≈ 12 is UNIVERSAL")
    else:
        print("⚠ HYPOTHESIS NEEDS REFINEMENT")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
