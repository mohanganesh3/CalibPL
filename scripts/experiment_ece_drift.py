#!/usr/bin/env python3
"""
Generate New Figure 1: Calibration Drift (Sparse vs Dense, Cls vs Loc)
Tracks the ECE across self-training iterations 0-3 using the generated models.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import random
import torch

# K80 GPU Fix
torch.backends.cudnn.enabled = False

from scripts.prediction_stability import compute_cgjs_for_image

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
DATA_CFG = PROJ / "data" / "coco" / "yolo_format" / "coco_frac_1.yaml"

MODELS = [
    PROJ / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt", # Iter 0 (Baseline)
    PROJ / "results" / "calibpl_v3" / "pseudo_label_seed42_coco1pct" / "iter_1" / "model" / "train" / "weights" / "best.pt",
    PROJ / "results" / "calibpl_v3" / "pseudo_label_seed42_coco1pct" / "iter_2" / "model" / "train" / "weights" / "best.pt",
    PROJ / "results" / "calibpl_v3" / "pseudo_label_seed42_coco1pct" / "iter_3" / "model" / "train" / "weights" / "best.pt",
]

def get_sparse_dense_splits(val_json_path, n_samples=300):
    """Finds images with 1-5 boxes (Sparse) and >=15 boxes (Dense)."""
    with open(val_json_path, 'r') as f:
        coco = json.load(f)
        
    img_box_counts = {img['id']: 0 for img in coco['images']}
    for ann in coco['annotations']:
        img_box_counts[ann['image_id']] += 1
        
    sparse_ids = [idx for idx, count in img_box_counts.items() if 1 <= count <= 5]
    dense_ids = [idx for idx, count in img_box_counts.items() if count >= 12] # lowered to 12 to ensure enough samples
    
    random.seed(42)
    sparse_sample = random.sample(sparse_ids, min(n_samples, len(sparse_ids)))
    dense_sample = random.sample(dense_ids, min(n_samples, len(dense_ids)))
    
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    sparse_files = sorted([id_to_filename[idx] for idx in sparse_sample])
    dense_files = sorted([id_to_filename[idx] for idx in dense_sample])
    
    return sparse_files, dense_files

def compute_ece(confs, corrs, bins=10):
    if len(confs) == 0:
        return 0.0
    bin_limits = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n_total = len(confs)
    
    for i in range(bins):
        bin_mask = (confs >= bin_limits[i]) & (confs < bin_limits[i+1])
        if i == bins - 1:
            bin_mask = (confs >= bin_limits[i]) & (confs <= bin_limits[i+1])
            
        n_bin = np.sum(bin_mask)
        if n_bin > 0:
            bin_acc = np.mean(corrs[bin_mask])
            bin_conf = np.mean(confs[bin_mask])
            ece += (n_bin / n_total) * np.abs(bin_acc - bin_conf)
            
    return ece

def extract_confs_and_corrs(model_path, image_files, val_lbl_dir, val_img_dir):
    """Evaluates the model on subset and returns arrays of confs and actual correctness."""
    model = YOLO(model_path)
    
    all_raw_confs = []
    all_cgjs_scores = []
    all_cls_corr = []
    all_loc_corr = []

    # Use just 100 images for fast generation since TTA is involved
    if len(image_files) > 100:
        image_files = image_files[:100]

    for img_name in image_files:
        img_path = str(val_img_dir / img_name)
        if not os.path.exists(img_path): continue
        
        # Load GT
        gt_boxes = []
        gt_classes = []
        label_path = val_lbl_dir / img_name.replace('.jpg', '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        gt_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                        gt_classes.append(cls_id)
                        
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
        gt_classes = np.array(gt_classes) if gt_classes else np.empty(0)
        
        # Predict
        results = model.predict(img_path, device=0, conf=0.05, verbose=False, max_det=100)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
            
        pred_confs = results[0].boxes.conf.cpu().numpy()
        pred_boxes = results[0].boxes.xyxyn.cpu().numpy()
        pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Get CGJS (Loc Confidence)
        cgjs_scores = compute_cgjs_for_image(model, img_path, results[0], device=0, use_multi_scale=False)
        
        # Match
        n = len(pred_boxes)
        cls_corr = np.zeros(n)
        loc_corr = np.zeros(n)
        
        for i in range(n):
            best_iou = 0
            best_j = -1
            for j in range(len(gt_boxes)):
                box_a = pred_boxes[i]
                box_b = gt_boxes[j]
                x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
                x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0
                
                if pred_classes[i] == gt_classes[j] and iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_j >= 0:
                if best_iou >= 0.1: cls_corr[i] = 1
                if best_iou >= 0.5: loc_corr[i] = 1
                
        all_raw_confs.extend(pred_confs.tolist())
        all_cgjs_scores.extend(cgjs_scores.tolist())
        all_cls_corr.extend(cls_corr.tolist())
        all_loc_corr.extend(loc_corr.tolist())
        
    del model
    torch.cuda.empty_cache()
    
    return np.array(all_raw_confs), np.array(all_cls_corr), np.array(all_cgjs_scores), np.array(all_loc_corr)

def main():
    val_json = PROJ / "data" / "coco" / "annotations" / "instances_val2017.json"
    val_img_dir = PROJ / "data" / "coco" / "val2017"
    val_lbl_dir = PROJ / "data" / "coco" / "yolo_format" / "val" / "labels"
    
    print("Finding Sparse vs Dense splits in COCO val2017...")
    sparse_files, dense_files = get_sparse_dense_splits(val_json, n_samples=100)
    print(f"Sparse images: {len(sparse_files)} | Dense images: {len(dense_files)}")
    
    results = {
        'sparse_cls': [], 'sparse_loc': [],
        'dense_cls': [], 'dense_loc': []
    }
    
    for iter_idx, model_path in enumerate(MODELS):
        print(f"\nEvaluating Iteration {iter_idx} ({model_path.name})")
        if not os.path.exists(model_path):
            print(f"  Missing model {model_path}. Skipping.")
            continue
            
        print("  Evaluating Sparse subset...")
        sp_confs, sp_cls_corr, sp_cgjs, sp_loc_corr = extract_confs_and_corrs(model_path, sparse_files, val_lbl_dir, val_img_dir)
        sp_cls_ece = compute_ece(sp_confs, sp_cls_corr)
        sp_loc_ece = compute_ece(sp_cgjs, sp_loc_corr)
        results['sparse_cls'].append(sp_cls_ece)
        results['sparse_loc'].append(sp_loc_ece)
        
        print("  Evaluating Dense subset...")
        dn_confs, dn_cls_corr, dn_cgjs, dn_loc_corr = extract_confs_and_corrs(model_path, dense_files, val_lbl_dir, val_img_dir)
        dn_cls_ece = compute_ece(dn_confs, dn_cls_corr)
        dn_loc_ece = compute_ece(dn_cgjs, dn_loc_corr)
        results['dense_cls'].append(dn_cls_ece)
        results['dense_loc'].append(dn_loc_ece)
        
        print(f"  Sparse Cls ECE: {sp_cls_ece:.4f} | Sparse Loc ECE: {sp_loc_ece:.4f}")
        print(f"   Dense Cls ECE: {dn_cls_ece:.4f} |  Dense Loc ECE: {dn_loc_ece:.4f}")
        
    # Plotting
    iters = list(range(len(results['sparse_cls'])))
    
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cls ECE Plot
    ax1.plot(iters, results['sparse_cls'], marker='o', label='Sparse (≤5) Cls ECE', color='blue', linestyle='--')
    ax1.plot(iters, results['dense_cls'], marker='s', label='Dense (≥12) Cls ECE', color='darkblue', linewidth=2)
    ax1.set_title("Classification Calibration Drift")
    ax1.set_xlabel("Self-Training Iteration")
    ax1.set_ylabel("Expected Calibration Error (ECE)")
    ax1.set_xticks(iters)
    ax1.legend()
    
    # Loc ECE Plot
    ax2.plot(iters, results['sparse_loc'], marker='o', label='Sparse (≤5) Loc ECE', color='red', linestyle='--')
    ax2.plot(iters, results['dense_loc'], marker='s', label='Dense (≥12) Loc ECE', color='darkred', linewidth=2)
    ax2.set_title("Localization Calibration Drift")
    ax2.set_xlabel("Self-Training Iteration")
    ax2.set_ylabel("Expected Calibration Error (ECE)")
    ax2.set_xticks(iters)
    ax2.legend()
    
    plt.tight_layout()
    out_path = PROJ / "results" / "figures" / "ece_drift_fig1.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved Figure 1 to {out_path}")

if __name__ == '__main__':
    main()
