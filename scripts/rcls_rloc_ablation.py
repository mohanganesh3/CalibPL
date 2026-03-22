#!/usr/bin/env python3
"""
Generate Independent rcls vs rloc Ablation Table.
Grid search over rcls in {0.5, 0.6, 0.7} and rloc in {0.5, 0.6, 0.7}.
"""

import os
import sys
import json
import numpy as np
import random
from pathlib import Path
import torch
import cv2

torch.backends.cudnn.enabled = False

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJ))

from scripts.prediction_stability import compute_cgjs_for_image

def compute_precision_recall_at_params(model, val_images, val_lbl_dir, val_img_dir, r_cls, r_loc, n_images=100):
    tp = 0
    fp = 0
    fn = 0
    
    for img_name in val_images[:n_images]:
        lbl_path = val_lbl_dir / img_name.replace('.jpg', '.txt')
        img_path = val_img_dir / img_name
        
        # Load GT
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        gt_boxes.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, cls))
        
        # Predict
        results = model.predict(str(img_path), device=0, conf=0.05, verbose=False, max_det=200)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            fn += len(gt_boxes)
            continue
        
        boxes = results[0].boxes
        
        cgjs_scores = compute_cgjs_for_image(model, str(img_path), results[0], lightweight=True, conf_threshold=0.05)
        
        matched_gt = set()
        
        for j in range(len(boxes)):
            conf = float(boxes.conf[j].item())
            cgjs = float(cgjs_scores[j]) if j < len(cgjs_scores) else conf
            
            # Independent filtering
            if conf >= r_cls and cgjs >= r_loc:
                pred_box = boxes.xywhn[j].cpu().numpy()
                pcx, pcy, pw, ph = pred_box
                px1 = pcx - pw/2; py1 = pcy - ph/2
                px2 = pcx + pw/2; py2 = pcy + ph/2
                
                pred_cls = int(boxes.cls[j].item())
                
                is_tp = False
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gx1, gy1, gx2, gy2, gcls) in enumerate(gt_boxes):
                    if pred_cls != gcls:
                        continue
                    ix1 = max(px1, gx1); iy1 = max(py1, gy1)
                    ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    area_p = (px2-px1) * (py2-py1)
                    area_g = (gx2-gx1) * (gy2-gy1)
                    union = area_p + area_g - inter
                    iou = inter / union if union > 0 else 0
                    if iou >= 0.5 and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                    matched_gt.add(best_gt_idx)
                    tp += 1
                else:
                    fp += 1
                    
        fn += (len(gt_boxes) - len(matched_gt))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    count = tp + fp
    return precision, recall, count

def main():
    from ultralytics import YOLO
    
    # We evaluate on SKU110k validation subset
    model_path = PROJ / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt"
    if not model_path.exists():
        print("Model not found")
        return
    model = YOLO(str(model_path))
    
    val_img_dir = PROJ / "data" / "SKU110K" / "yolo_format" / "test" / "images"
    val_lbl_dir = PROJ / "data" / "SKU110K" / "yolo_format" / "test" / "labels"
    
    val_images = sorted([f.name for f in val_img_dir.glob("*.jpg")])
    random.seed(42)
    random.shuffle(val_images)
    
    thresholds = [0.5, 0.6, 0.7]
    matrix = {}
    
    print("=== Independent rcls vs rloc Ablation ===")
    print(f"{'r_cls':<5} | {'r_loc':<5} | {'Precision':<9} | {'Recall':<9} | {'# Pseudo'}")
    print("-" * 55)
    
    results_dir = PROJ / "results" / "ablations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for r_cls in thresholds:
        matrix[r_cls] = {}
        for r_loc in thresholds:
            prec, rec, count = compute_precision_recall_at_params(model, val_images, val_lbl_dir, val_img_dir, r_cls, r_loc, n_images=50)
            matrix[r_cls][r_loc] = {'precision': prec, 'recall': rec, 'count': count}
            print(f"{r_cls:<5} | {r_loc:<5} | {prec:.4f}    | {rec:.4f}    | {count}")
            
    with open(results_dir / "rcls_rloc_ablation.json", "w") as f:
        json.dump(matrix, f, indent=4)
        
    print(f"\nSaved results to {results_dir / 'rcls_rloc_ablation.json'}")

if __name__ == "__main__":
    main()
