#!/usr/bin/env python3
"""
Week 2 & 3: Run Calibration Benchmark on trained baselines.

Usage:
    python scripts/run_calibration_benchmark.py --model rtdetr
    python scripts/run_calibration_benchmark.py --model yolov12
    python scripts/run_calibration_benchmark.py --model all
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration.detection_calibration import run_calibration_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run calibration benchmark")
    parser.add_argument('--model', choices=['rtdetr', 'yolov12', 'all'], default='all')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--bins', type=int, default=15, help='Number of calibration bins')
    args = parser.parse_args()
    
    results_dir = PROJECT_ROOT / "results" / "week1_baselines"
    calib_dir = PROJECT_ROOT / "results" / "week2_calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    
    gt_labels_dir = str(PROJECT_ROOT / "data" / "SKU110K" / "yolo_format" / "test" / "labels")
    images_dir = str(PROJECT_ROOT / "data" / "SKU110K" / "yolo_format" / "test" / "images")
    
    models = ['rtdetr', 'yolov12'] if args.model == 'all' else [args.model]
    all_results = {}
    
    for model_name in models:
        pred_file = results_dir / f"{model_name}_predictions.json"
        
        if not pred_file.exists():
            print(f"\n⚠ No predictions found for {model_name} at {pred_file}")
            print(f"  Run: python scripts/train_baselines.py --model {model_name}")
            continue
        
        results = run_calibration_benchmark(
            predictions_file=str(pred_file),
            gt_labels_dir=gt_labels_dir,
            images_dir=images_dir,
            output_dir=str(calib_dir),
            model_name=model_name,
            iou_threshold=args.iou,
            n_bins=args.bins
        )
        all_results[model_name] = results
    
    # Cross-model comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"CROSS-MODEL CALIBRATION COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':<15} {'mAP50':>8} {'D-ECE':>8} {'MCE':>8} {'Brier':>8} {'Avg Conf':>10}")
        print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        
        for name, r in all_results.items():
            u = r['uncalibrated']
            print(f"{name:<15} {'N/A':>8} {u['d_ece']:>8.4f} {u['mce']:>8.4f} "
                  f"{u['brier']:>8.4f} {u['avg_confidence']:>10.4f}")
    
    # Save combined results
    combined_file = calib_dir / "calibration_benchmark.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined results: {combined_file}")


if __name__ == '__main__':
    main()
