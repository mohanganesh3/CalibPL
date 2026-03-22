#!/usr/bin/env python3
"""
Dual Uncertainty Validation
===========================
Runs MCDropoutDetector on a subset of the SKU-110K test set.
Matches predictions to Ground Truth to compute TP/FP statuses.
Analyzes if Epistemic and Loc_Epistemic uncertainty are significantly
higher for False Positives compared to True Positives.
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path

# Fix for Tesla K80 cuDNN issue
import torch
torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration.mc_dropout import MCDropoutDetector
from core.calibration.detection_calibration import compute_iou


def load_ground_truth(label_dir: str, image_names: list) -> dict:
    """Loads YOLO format txt ground truth files into dict keyed by image name."""
    gts = {}
    label_path = Path(label_dir)
    for img_name in image_names:
        txt_name = img_name.replace('.jpg', '.txt')
        txt_file = label_path / txt_name
        img_gts = []
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # cx, cy, w, h
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert to normalized x1, y1, x2, y2 (for IoU matching)
                        # We don't have exact image dims here easily, but relative IoU
                        # calculation still works if both sets of boxes are normalized or both absolute.
                        # MCDropout outputs absolute pixels because of Ultralytics predict.
                        # We need image dims to convert GT to absolute. 
                        # We will read it from actual image using cv2.
                        img_gts.append({
                            'cls': cls_id,
                            'cx': cx, 'cy': cy, 'w': w, 'h': h
                        })
        gts[img_name] = img_gts
    return gts


def match_and_analyze(images_dir, labels_dir, mc_results):
    """
    Matches MC Detections to Ground Truth to separate TPs and FPs.
    Computes mean uncertainties for each group.
    """
    import cv2
    
    gt_dict = load_ground_truth(labels_dir, list(mc_results.keys()))
    
    tp_stats = {'conf': [], 'cls_ep': [], 'loc_ep': [], 'passes': []}
    fp_stats = {'conf': [], 'cls_ep': [], 'loc_ep': [], 'passes': []}
    
    for img_name, dets in mc_results.items():
        if not dets:
            continue
            
        # Get image dims to convert GT YOLO format to absolute [x1, y1, x2, y2]
        img_path = Path(images_dir) / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, _ = img.shape
        
        # Convert GTs
        gts = gt_dict.get(img_name, [])
        abs_gts = []
        for gt in gts:
            cx_abs, cy_abs = gt['cx'] * w, gt['cy'] * h
            w_abs, h_abs = gt['w'] * w, gt['h'] * h
            x1 = cx_abs - (w_abs / 2)
            y1 = cy_abs - (h_abs / 2)
            x2 = cx_abs + (w_abs / 2)
            y2 = cy_abs + (h_abs / 2)
            abs_gts.append({'box': [x1, y1, x2, y2], 'cls': gt['cls'], 'matched': False})
            
        # Match Detections
        # Sort by confidence highest first
        dets_sorted = sorted(dets, key=lambda x: x['mean_confidence'], reverse=True)
        
        for det in dets_sorted:
            det_box = det['box']
            det_cls = det['class_id']
            
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(abs_gts):
                if gt['matched'] or gt['cls'] != det_cls:
                    continue
                iou = compute_iou(det_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou > 0.5:
                # True Positive
                abs_gts[best_gt_idx]['matched'] = True
                tp_stats['conf'].append(det['mean_confidence'])
                tp_stats['cls_ep'].append(det['epistemic'])
                tp_stats['loc_ep'].append(det['loc_epistemic'])
                tp_stats['passes'].append(det['num_passes'])
            else:
                # False Positive
                fp_stats['conf'].append(det['mean_confidence'])
                fp_stats['cls_ep'].append(det['epistemic'])
                fp_stats['loc_ep'].append(det['loc_epistemic'])
                fp_stats['passes'].append(det['num_passes'])
                
    # Print Analysis
    def print_group(name, stats):
        n = len(stats['conf'])
        print(f"\\n=== {name} ({n} detections) ===")
        if n == 0:
            return
        print(f"Mean Confidence : {np.mean(stats['conf']):.4f}  (std: {np.std(stats['conf']):.4f})")
        print(f"Mean Cls Epistem: {np.mean(stats['cls_ep']):.5f}  (std: {np.std(stats['cls_ep']):.5f})")
        print(f"Mean Loc Epistem: {np.mean(stats['loc_ep']):.1f}  (std: {np.std(stats['loc_ep']):.1f})")
        print(f"Mean Num Passes : {np.mean(stats['passes']):.2f} / 5")
        
    print_group("True Positives (Correct Detections)", tp_stats)
    print_group("False Positives (Hallucinations)", fp_stats)
    
    print("\n\\n=== CONCLUSION ===")
    if np.mean(fp_stats['cls_ep']) > np.mean(tp_stats['cls_ep']):
         ratio_cls = np.mean(fp_stats['cls_ep']) / (np.mean(tp_stats['cls_ep']) + 1e-9)
         print(f"✓ Hypothesis CONFIRMED: False Positives have {ratio_cls:.1f}x higher Classification Epistemic Uncertainty.")
    if np.mean(fp_stats['loc_ep']) > np.mean(tp_stats['loc_ep']):
         ratio_loc = np.mean(fp_stats['loc_ep']) / (np.mean(tp_stats['loc_ep']) + 1e-9)
         print(f"✓ Hypothesis CONFIRMED: False Positives have {ratio_loc:.1f}x higher Localization Epistemic Uncertainty.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="results/week1_baselines/yolov12/train/weights/best.pt")
    parser.add_argument('--images', type=str, default="data/SKU110K/yolo_format/val/images")
    parser.add_argument('--labels', type=str, default="data/SKU110K/yolo_format/val/labels")
    parser.add_argument('--num-images', type=int, default=100, help="Number of test images to analyze")
    parser.add_argument('--t', type=int, default=5, help="Number of MC Dropout passes")
    args = parser.parse_args()
    
    model_path = str(PROJECT_ROOT / args.model)
    images_dir = str(PROJECT_ROOT / args.images)
    labels_dir = str(PROJECT_ROOT / args.labels)
    
    print(f"Initializing MCDropoutDetector (T={args.t}) on {model_path}...")
    mc = MCDropoutDetector(model_path, T=args.t, device='0,1,2,3')
    
    # Select subset of images
    image_paths = sorted([str(p) for p in Path(images_dir).glob("*.jpg")])[:args.num_images]
    print(f"Running uncertainty estimation on {len(image_paths)} images...")
    
    outs_json = "/tmp/uncertainty_val.json"
    results = mc.predict_batch_with_uncertainty(image_paths, save_path=outs_json)
    
    print("\\nAnalyzing results against Ground Truth...")
    match_and_analyze(images_dir, labels_dir, results)

if __name__ == "__main__":
    main()
