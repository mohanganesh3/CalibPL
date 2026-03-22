#!/usr/bin/env python3
"""
Generate Sensitivity Analysis Table for CalibPL.
Sweeps target reliability r and CGJS threshold β.
"""

import os
import sys
import json
import numpy as np
import random
from pathlib import Path
import torch

torch.backends.cudnn.enabled = False

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJ))

from scripts.prediction_stability import compute_cgjs_for_image

def compute_precision_at_params(model, val_images, val_lbl_dir, r_target, beta, n_images=100):
    """Compute pseudo-label precision for a given (r, β) pair."""
    from sklearn.isotonic import IsotonicRegression
    
    tp = 0
    fp = 0
    
    for img_path in val_images[:n_images]:
        lbl_path = val_lbl_dir / img_path.name.replace('.jpg', '.txt').replace('.png', '.txt')
        
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
            continue
        
        boxes = results[0].boxes
        cgjs_scores = compute_cgjs_for_image(model, str(img_path), results[0], device=0, use_multi_scale=False)
        
        for j in range(len(boxes)):
            conf = float(boxes.conf[j].item())
            cgjs = float(cgjs_scores[j]) if cgjs_scores is not None else 0.0
            
            # CalibPL gate (simplified: use raw conf as proxy for calibrated cls)
            if conf >= r_target and cgjs >= beta:
                # Check if TP
                pred_box = boxes.xyxyn[j].cpu().numpy()
                px1, py1, px2, py2 = pred_box
                pred_cls = int(boxes.cls[j].item())
                
                is_tp = False
                for gx1, gy1, gx2, gy2, gcls in gt_boxes:
                    if pred_cls != gcls:
                        continue
                    ix1 = max(px1, gx1); iy1 = max(py1, gy1)
                    ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    area_p = (px2-px1) * (py2-py1)
                    area_g = (gx2-gx1) * (gy2-gy1)
                    union = area_p + area_g - inter
                    iou = inter / union if union > 0 else 0
                    if iou >= 0.5:
                        is_tp = True
                        break
                
                if is_tp:
                    tp += 1
                else:
                    fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    count = tp + fp
    return precision, count

def main():
    from ultralytics import YOLO
    
    model_path = PROJ / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = PROJ / "models" / "yolo12n.pt"
    model = YOLO(str(model_path))
    
    val_img_dir = PROJ / "data" / "coco" / "yolo_format" / "val" / "images"
    val_lbl_dir = PROJ / "data" / "coco" / "yolo_format" / "val" / "labels"
    
    val_images = sorted(val_img_dir.glob("*.jpg"))
    random.seed(42)
    random.shuffle(val_images)
    
    r_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    beta_values = [0.2, 0.3, 0.5, 0.7]
    
    print("Sensitivity Analysis: Precision × (r, β)")
    print(f"{'r \\\\ β':>8}", end="")
    for beta in beta_values:
        print(f"  β={beta:.1f}", end="")
    print()
    print("-" * 50)
    
    results = {}
    for r in r_values:
        results[r] = {}
        print(f"r={r:.1f}  ", end="", flush=True)
        for beta in beta_values:
            prec, count = compute_precision_at_params(model, val_images, val_lbl_dir, r, beta, n_images=80)
            results[r][beta] = {'precision': prec, 'count': count}
            print(f"  {prec:.3f}", end="", flush=True)
        print()
    
    # Save results
    out_path = PROJ / "results" / "figures" / "sensitivity_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    
    # Generate LaTeX table
    print("\n% LaTeX Table:")
    print("\\begin{tabular}{l" + "c" * len(beta_values) + "}")
    print("\\hline")
    header = "$r$ \\textbackslash{} $\\beta$"
    for beta in beta_values:
        header += f" & {beta:.1f}"
    header += " \\\\"
    print(header)
    print("\\hline")
    for r in r_values:
        row = f"{r:.1f}"
        for beta in beta_values:
            p = results[r][beta]['precision']
            row += f" & {p:.3f}"
        row += " \\\\"
        print(row)
    print("\\hline")
    print("\\end{tabular}")

if __name__ == '__main__':
    main()
