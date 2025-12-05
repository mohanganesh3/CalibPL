#!/usr/bin/env python3
"""
LaECE: Localization-Aware Expected Calibration Error
=====================================================
Implementation based on Kuzucu et al. (ECCV 2024 Oral):
"On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines"

LaECE measures calibration conditioned on localization quality.
Standard D-ECE only checks: confidence vs classification correctness.
LaECE asks: are well-localized detections better calibrated than poorly-localized ones?

Formula:
    LaECE = (1/K) * Σ_k ECE(detections with IoU in bin k)
    
Where K bins divide IoU into ranges: [0.5-0.6], [0.6-0.7], [0.7-0.8], [0.8-0.9], [0.9-1.0]

Usage:
    python3 scripts/compute_laece.py --model yolov12 --weights path/to/best.pt
    python3 scripts/compute_laece.py --use-cached  # use existing detection results
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))


def compute_ece_from_bins(confidences, correctness, n_bins=15):
    """Compute ECE from arrays of confidence and correctness."""
    if len(confidences) == 0:
        return 0.0, {}
    
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_eces = {}
    ece = 0.0
    total = len(confidences)
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        
        if n_in_bin > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctness[mask].mean()
            bin_ece = abs(avg_conf - avg_acc)
            ece += (n_in_bin / total) * bin_ece
            bin_eces[f"{lo:.2f}-{hi:.2f}"] = {
                'count': int(n_in_bin),
                'avg_confidence': float(avg_conf),
                'avg_accuracy': float(avg_acc),
                'gap': float(bin_ece),
            }
    
    return float(ece), bin_eces


def compute_d_ece(detections, iou_threshold=0.5, n_bins=15):
    """
    Compute Detection ECE (D-ECE).
    
    detections: list of dicts with 'confidence' and 'iou_with_gt' keys
    A detection is "correct" if iou_with_gt >= iou_threshold
    """
    confidences = [d['confidence'] for d in detections]
    correctness = [1.0 if d['iou_with_gt'] >= iou_threshold else 0.0 for d in detections]
    
    return compute_ece_from_bins(confidences, correctness, n_bins)


def compute_laece(detections, iou_threshold=0.5, n_iou_bins=5, n_conf_bins=15):
    """
    Compute Localization-Aware ECE (LaECE).
    
    Partitions detections by IoU quality, then computes ECE within each IoU bin.
    LaECE reveals whether well-localized detections are better calibrated
    than poorly-localized ones.
    
    detections: list of dicts with 'confidence' and 'iou_with_gt' keys
    """
    # IoU bins: [0.5-0.6], [0.6-0.7], [0.7-0.8], [0.8-0.9], [0.9-1.0]
    iou_bin_edges = np.linspace(iou_threshold, 1.0, n_iou_bins + 1)
    
    iou_bin_results = {}
    eces_per_bin = []
    weights_per_bin = []
    total_dets = len(detections)
    
    for i in range(n_iou_bins):
        lo, hi = iou_bin_edges[i], iou_bin_edges[i + 1]
        
        # Get detections in this IoU range (matched detections)
        if i < n_iou_bins - 1:
            bin_dets = [d for d in detections if lo <= d['iou_with_gt'] < hi]
        else:
            bin_dets = [d for d in detections if lo <= d['iou_with_gt'] <= hi]
        
        # Also include false positives (iou < threshold) in the lowest bin
        if i == 0:
            fp_dets = [d for d in detections if d['iou_with_gt'] < iou_threshold]
            # For FPs, they're "incorrect" regardless of confidence
            all_bin_dets = fp_dets + bin_dets
        else:
            all_bin_dets = bin_dets
        
        if len(all_bin_dets) > 0:
            confidences = [d['confidence'] for d in all_bin_dets]
            correctness = [1.0 if d['iou_with_gt'] >= iou_threshold else 0.0 for d in all_bin_dets]
            bin_ece, bin_details = compute_ece_from_bins(confidences, correctness, n_conf_bins)
            
            eces_per_bin.append(bin_ece)
            weights_per_bin.append(len(all_bin_dets))
            
            iou_bin_results[f"IoU_{lo:.1f}-{hi:.1f}"] = {
                'n_detections': len(all_bin_dets),
                'ece': float(bin_ece),
                'avg_confidence': float(np.mean(confidences)),
                'accuracy': float(np.mean(correctness)),
                'bins': bin_details,
            }
        else:
            iou_bin_results[f"IoU_{lo:.1f}-{hi:.1f}"] = {
                'n_detections': 0,
                'ece': 0.0,
            }
    
    # LaECE = weighted average of per-IoU-bin ECEs
    if sum(weights_per_bin) > 0:
        laece = sum(e * w for e, w in zip(eces_per_bin, weights_per_bin)) / sum(weights_per_bin)
    else:
        laece = 0.0
    
    return {
        'laece': float(laece),
        'iou_bins': iou_bin_results,
        'n_total_detections': total_dets,
    }


def load_detections_from_validation(model_path, data_yaml, device='0'):
    """
    Run model validation and extract per-detection confidence + IoU.
    Returns list of dicts: [{'confidence': float, 'iou_with_gt': float}, ...]
    """
    import torch
    torch.backends.cudnn.enabled = False
    
    from ultralytics import YOLO, RTDETR
    
    # Determine model type
    model_name = Path(model_path).stem.lower()
    if 'rtdetr' in model_name:
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)
    
    # Run validation to get per-image results
    results = model.val(data=data_yaml, device=device, verbose=False)
    
    # We need to re-run inference to get per-detection IoUs
    # Use the model's predict + manual IoU calculation
    import yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    
    val_path = Path(data_cfg['path']) / data_cfg['val']
    if not val_path.exists():
        val_path = Path(data_cfg['val'])
    
    label_path = str(val_path).replace('/images', '/labels')
    
    detections = []
    
    for img_file in sorted(os.listdir(val_path)):
        if not img_file.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        img_path = os.path.join(val_path, img_file)
        lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
        lbl_path = os.path.join(label_path, lbl_file)
        
        # Get predictions
        preds = model.predict(img_path, device=device, verbose=False, conf=0.01)[0]
        
        if preds.boxes is None or len(preds.boxes) == 0:
            continue
        
        pred_boxes = preds.boxes.xyxy.cpu().numpy()
        pred_confs = preds.boxes.conf.cpu().numpy()
        
        # Load ground truth
        gt_boxes = []
        if os.path.exists(lbl_path):
            img_h, img_w = preds.orig_shape
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = (cx - w/2) * img_w
                        y1 = (cy - h/2) * img_h
                        x2 = (cx + w/2) * img_w
                        y2 = (cy + h/2) * img_h
                        gt_boxes.append([x1, y1, x2, y2])
        
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
        
        # Compute IoU matrix
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = compute_iou_matrix(pred_boxes, gt_boxes)
            
            # Greedy matching: each pred gets best IoU with any unmatched GT
            matched_gt = set()
            for pred_idx in np.argsort(-pred_confs):
                if len(gt_boxes) > 0:
                    best_gt = -1
                    best_iou = 0.0
                    for gt_idx in range(len(gt_boxes)):
                        if gt_idx not in matched_gt and ious[pred_idx, gt_idx] > best_iou:
                            best_iou = ious[pred_idx, gt_idx]
                            best_gt = gt_idx
                    
                    if best_gt >= 0 and best_iou > 0:
                        matched_gt.add(best_gt)
                    
                    detections.append({
                        'confidence': float(pred_confs[pred_idx]),
                        'iou_with_gt': float(best_iou),
                    })
                else:
                    detections.append({
                        'confidence': float(pred_confs[pred_idx]),
                        'iou_with_gt': 0.0,
                    })
        else:
            for conf in pred_confs:
                detections.append({
                    'confidence': float(conf),
                    'iou_with_gt': 0.0,
                })
    
    return detections


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two sets of boxes."""
    n_a, n_b = len(boxes_a), len(boxes_b)
    ious = np.zeros((n_a, n_b))
    
    for i in range(n_a):
        for j in range(n_b):
            x1 = max(boxes_a[i][0], boxes_b[j][0])
            y1 = max(boxes_a[i][1], boxes_b[j][1])
            x2 = min(boxes_a[i][2], boxes_b[j][2])
            y2 = min(boxes_a[i][3], boxes_b[j][3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (boxes_a[i][2] - boxes_a[i][0]) * (boxes_a[i][3] - boxes_a[i][1])
            area_b = (boxes_b[j][2] - boxes_b[j][0]) * (boxes_b[j][3] - boxes_b[j][1])
            union = area_a + area_b - inter
            
            ious[i, j] = inter / max(union, 1e-6)
    
    return ious


def main():
    parser = argparse.ArgumentParser(description="Compute D-ECE and LaECE")
    parser.add_argument('--model', choices=['yolov12', 'rtdetr', 'both'], default='both')
    parser.add_argument('--device', default='0')
    parser.add_argument('--use-cached', action='store_true', help='Use cached detection results')
    parser.add_argument('--cache-dir', default=str(PROJECT_ROOT / "results" / "laece"))
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    data_yaml = str(PROJECT_ROOT / "data" / "SKU110K" / "sku110k.yaml")
    
    models = {}
    if args.model in ['yolov12', 'both']:
        models['yolov12'] = str(PROJECT_ROOT / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt")
    if args.model in ['rtdetr', 'both']:
        models['rtdetr'] = str(PROJECT_ROOT / "results" / "week1_baselines" / "rtdetr" / "train" / "weights" / "best.pt")
    
    all_results = {}
    
    for model_name, model_path in models.items():
        print(f"\n{'='*60}")
        print(f"Computing LaECE for {model_name}")
        print(f"{'='*60}")
        
        cache_file = cache_dir / f"{model_name}_detections.json"
        
        if args.use_cached and cache_file.exists():
            print(f"Loading cached detections from {cache_file}")
            with open(cache_file) as f:
                detections = json.load(f)
        else:
            print(f"Running inference with {model_path}...")
            if not os.path.exists(model_path):
                print(f"  ⚠ Model not found: {model_path}")
                continue
            detections = load_detections_from_validation(model_path, data_yaml, args.device)
            
            # Cache detections
            with open(cache_file, 'w') as f:
                json.dump(detections, f)
            print(f"  Cached {len(detections)} detections to {cache_file}")
        
        print(f"  Total detections: {len(detections)}")
        
        # Compute D-ECE
        d_ece, d_ece_bins = compute_d_ece(detections)
        print(f"\n  D-ECE: {d_ece:.4f} ({d_ece*100:.1f}%)")
        
        # Compute LaECE
        laece_result = compute_laece(detections)
        print(f"  LaECE: {laece_result['laece']:.4f} ({laece_result['laece']*100:.1f}%)")
        
        print(f"\n  IoU-bin breakdown:")
        for bin_name, bin_data in laece_result['iou_bins'].items():
            if bin_data['n_detections'] > 0:
                print(f"    {bin_name}: ECE={bin_data['ece']:.4f}, n={bin_data['n_detections']}, "
                      f"conf={bin_data['avg_confidence']:.3f}, acc={bin_data['accuracy']:.3f}")
        
        all_results[model_name] = {
            'd_ece': d_ece,
            'laece': laece_result['laece'],
            'laece_details': laece_result,
            'd_ece_bins': d_ece_bins,
            'n_detections': len(detections),
        }
    
    # Save all results
    output_file = cache_dir / "laece_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'D-ECE':<12} {'LaECE':<12} {'Detections'}")
    print("-" * 50)
    for name, r in all_results.items():
        print(f"{name:<12} {r['d_ece']:<12.4f} {r['laece']:<12.4f} {r['n_detections']}")
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
