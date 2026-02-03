#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import brier_score_loss
from pathlib import Path
from ultralytics import YOLO
import torch

torch.backends.cudnn.enabled = False

PROJ = Path("/home/mohanganesh/retail-shelf-detection")

def compute_d_ece(confidences, correctness, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    d_ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop = in_bin.sum() / len(confidences)
        if prop > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = correctness[in_bin].mean()
            d_ece += prop * abs(avg_acc - avg_conf)
    return d_ece

def extract_real_scores(model_path, val_img_dir, val_lbl_dir, n_images=500):
    model = YOLO(model_path)
    all_confs = []
    all_corrs = []
    
    img_files = sorted(list(val_img_dir.glob("*.jpg")))[:n_images]
    
    for img_path in img_files:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])
                        gt_boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2, int(parts[0])])
        
        results = model.predict(str(img_path), device=0, conf=0.01, verbose=False, max_det=100)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0: continue
        
        preds = results[0].boxes
        for j in range(len(preds)):
            conf = float(preds.conf[j])
            box = preds.xyxyn[j].cpu().numpy()
            cls = int(preds.cls[j])
            
            is_tp = False
            for gx1, gy1, gx2, gy2, gcls in gt_boxes:
                if cls == gcls:
                    ix1 = max(box[0], gx1); iy1 = max(box[1], gy1)
                    ix2 = min(box[2], gx2); iy2 = min(box[3], gy2)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    area_a = (box[2]-box[0]) * (box[3]-box[1])
                    area_b = (gx2-gx1) * (gy2-gy1)
                    union = area_a + area_b - inter
                    iou = inter / union if union > 0 else 0
                    if iou >= 0.5:
                        is_tp = True
                        break
            
            all_confs.append(conf)
            all_corrs.append(1.0 if is_tp else 0.0)
            
    return np.array(all_confs), np.array(all_corrs)

def run_real_ablation():
    val_img_dir = PROJ / "data" / "coco" / "val2017"
    val_lbl_dir = PROJ / "data" / "coco" / "yolo_format" / "val" / "labels"
    
    iters = [1, 3, 5]
    results = {}
    
    print("=== Empirical Real-Data Ablation: Isotonic vs GMM ===")
    
    for it in iters:
        model_path = PROJ / f"results/calibpl_v3/pseudo_label_seed42_coco1pct/iter_{it}/model/train/weights/best.pt"
        if not model_path.exists():
            continue
            
        print(f"\nEvaluating Real Data for Iteration {it}...")
        scores, labels = extract_real_scores(model_path, val_img_dir, val_lbl_dir, n_images=500)
        if len(scores) == 0: continue
        
        raw_ece = compute_d_ece(scores, labels)
        
        try:
            scores_2d = scores.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='spherical', random_state=42)
            gmm.fit(scores_2d)
            tp_idx = np.argmax(gmm.means_.flatten())
            probas = gmm.predict_proba(scores_2d)
            gmm_calibrated = probas[:, tp_idx]
            gmm_ece = compute_d_ece(gmm_calibrated, labels)
        except Exception as e:
            print(f"  GMM fitting failed: {e}")
            gmm_ece = float('nan')
        
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(scores, labels)
            iso_calibrated = iso.transform(scores)
            iso_ece = compute_d_ece(iso_calibrated, labels)
        except Exception as e:
            print(f"  Isotonic fitting failed: {e}")
            iso_ece = float('nan')
        
        print(f"  Raw D-ECE:      {raw_ece:.4f}")
        print(f"  GMM D-ECE:      {gmm_ece:.4f}")
        print(f"  Isotonic D-ECE: {iso_ece:.4f}")
        
        results[f"iter_{it}"] = {
            'raw_ece': raw_ece,
            'gmm_ece': gmm_ece,
            'iso_ece': iso_ece,
            'n_samples': len(scores)
        }
    
    out_path = PROJ / "results" / "figures" / "real_gmm_isotonic_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical ablation results to {out_path}")

if __name__ == "__main__":
    run_real_ablation()
