#!/usr/bin/env python3
"""
Generate New Figure 1 supplemental: Calibration Drift on CrowdHuman
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction_stability import compute_cgjs_for_image
PROJ = Path("/home/mohanganesh/retail-shelf-detection")
CROWDHUMAN_RAW = PROJ / "data" / "CrowdHuman" / "raw" / "CrowdHuman"

MODELS_BASELINE = [
    PROJ / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt", # Iter 0
    PROJ / "results" / "cwpl" / "hard_seed42_frac10" / "iter_1" / "model_a" / "train" / "weights" / "best.pt",
    PROJ / "results" / "cwpl" / "hard_seed42_frac10" / "iter_2" / "model_a" / "train" / "weights" / "best.pt",
    PROJ / "results" / "cwpl" / "hard_seed42_frac10" / "iter_3" / "model_a" / "train" / "weights" / "best.pt",
]

MODELS_CALIBPL = [
    PROJ / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt", # Iter 0
    PROJ / "results" / "cwpl" / "cwpl_seed42_frac10" / "iter_1" / "model_a" / "train" / "weights" / "best.pt",
    PROJ / "results" / "cwpl" / "cwpl_seed42_frac10" / "iter_2" / "model_a" / "train" / "weights" / "best.pt",
    PROJ / "results" / "cwpl" / "cwpl_seed42_frac10" / "iter_3" / "model_a" / "train" / "weights" / "best.pt",
]

def get_sparse_dense_splits_crowdhuman(val_odgt_path, n_samples=300):
    """Finds images with 1-5 boxes (Sparse) and >=15 boxes (Dense) in CrowdHuman format."""
    sparse_ids = []
    dense_ids = []
    id_to_boxes = {}
    
    with open(val_odgt_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            img_id = data["ID"]
            boxes = [b for b in data["gtboxes"] if b["tag"] == "person" and not b.get("extra", {}).get("ignore", 0)]
            id_to_boxes[img_id] = boxes
            count = len(boxes)
            
            if 1 <= count <= 5:
                sparse_ids.append(img_id)
            elif count >= 15:
                dense_ids.append(img_id)
                
    random.seed(42)
    sparse_sample = random.sample(sparse_ids, min(n_samples, len(sparse_ids)))
    dense_sample = random.sample(dense_ids, min(n_samples, len(dense_ids)))
    
    return sparse_sample, dense_sample, id_to_boxes

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

def extract_confs_and_corrs_crowdhuman(model_path, image_ids, all_boxes, val_img_dir):
    """Evaluates the model on subset and returns arrays of confs and actual correctness."""
    model = YOLO(model_path)
    
    all_raw_confs = []
    all_cgjs_scores = []
    all_cls_corr = []
    all_loc_corr = []

    # Use 100 images for fast generation since TTA is involved
    if len(image_ids) > 100:
        image_ids = image_ids[:100]

    for img_id in image_ids:
        img_path = str(val_img_dir / f"{img_id}.jpg")
        if not os.path.exists(img_path): continue
        
        gt_data = all_boxes.get(img_id, [])
        gt_boxes = []
        for gt in gt_data:
            x, y, w, h = gt["fbox"]
            cx = x + w/2
            cy = y + h/2
            gt_boxes.append([0, cx, cy, w, h]) # person is class 0 in COCO
            
        gt_boxes = np.array(gt_boxes)
            
        # Run inference
        results = model.predict(img_path, conf=0.01, verbose=False)
        preds = results[0].boxes
        
        if len(preds) == 0:
            continue
            
        pred_boxes = preds.xywh.cpu().numpy()
        pred_confs = preds.conf.cpu().numpy()
        pred_classes = preds.cls.cpu().numpy()
        
        import cv2
        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        try:
            cgjs_scores = compute_cgjs_for_image(model, image_np, lightweight=True, conf_threshold=0.01)
        except Exception:
            cgjs_scores = pred_confs
            
        H_img, W_img = image_np.shape[:2]
        
        # Determine correctness (IoU > 0.5 and class match)
        for i, (pred_box, pred_conf, pred_cls) in enumerate(zip(pred_boxes, pred_confs, pred_classes)):
            # ONLY CONSIDER PERSON CLASS (0)
            if int(pred_cls) != 0:
                continue
                
            pcx, pcy, pw, ph = pred_box
            matched_cls = False
            matched_loc = False
            
            if len(gt_boxes) > 0:
                # Convert both to xyxy for IoU
                px1, py1 = pcx - pw/2, pcy - ph/2
                px2, py2 = pcx + pw/2, pcy + ph/2
                
                ious = []
                for gt in gt_boxes:
                    gcls, gcx, gcy, gw, gh = gt
                    gx1, gy1 = gcx - gw/2, gcy - gh/2
                    gx2, gy2 = gcx + gw/2, gcy + gh/2
                    
                    inter_x1 = max(px1, gx1)
                    inter_y1 = max(py1, gy1)
                    inter_x2 = min(px2, gx2)
                    inter_y2 = min(py2, gy2)
                    
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    p_area = pw * ph
                    g_area = gw * gh
                    iou = inter_area / (p_area + g_area - inter_area + 1e-6)
                    ious.append((iou, gcls))
                    
                best_iou, best_gcls = max(ious, key=lambda x: x[0])
                if best_iou >= 0.5:
                    matched_loc = True
                    if int(pred_cls) == int(best_gcls):
                        matched_cls = True
            
            all_raw_confs.append(pred_conf)
            all_cgjs_scores.append(cgjs_scores[i] if i < len(cgjs_scores) else pred_conf)
            all_cls_corr.append(1.0 if matched_cls else 0.0)
            all_loc_corr.append(1.0 if matched_loc else 0.0)
            
    return np.array(all_raw_confs), np.array(all_cgjs_scores), np.array(all_cls_corr), np.array(all_loc_corr)

def main():
    val_odgt = CROWDHUMAN_RAW / "annotation_val.odgt"
    val_img_dir = CROWDHUMAN_RAW / "Images"
    
    print("Finding Sparse and Dense Images in CrowdHuman...")
    sparse_files, dense_files, id_to_boxes = get_sparse_dense_splits_crowdhuman(val_odgt)
    print(f"Sampled {len(sparse_files)} sparse, {len(dense_files)} dense images.")
    
    results_dir = PROJ / "results" / "ece_drift"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    iters = [0, 1, 2, 3]
    drift_data = {
        'baseline_sparse_cls': [], 'baseline_sparse_loc': [],
        'baseline_dense_cls': [], 'baseline_dense_loc': [],
        'calibpl_sparse_cls': [], 'calibpl_sparse_loc': [],
        'calibpl_dense_cls': [], 'calibpl_dense_loc': [],
    }
    
    def evaluate_model_set(models_list, prefix):
        for i, model_path in enumerate(models_list):
            print(f"\n--- Evaluating Iteration {iters[i]} ({prefix}) ---")
            if not os.path.exists(model_path):
                print(f"Missing model: {model_path}")
                drift_data[f'{prefix}_sparse_cls'].append(np.nan)
                drift_data[f'{prefix}_sparse_loc'].append(np.nan)
                drift_data[f'{prefix}_dense_cls'].append(np.nan)
                drift_data[f'{prefix}_dense_loc'].append(np.nan)
                continue
                
            print("  Evaluating Sparse subset...")
            sp_confs, sp_cgjs, sp_cls, sp_loc = extract_confs_and_corrs_crowdhuman(model_path, sparse_files, id_to_boxes, val_img_dir)
            sp_ece_cls = compute_ece(sp_confs, sp_cls)
            sp_ece_loc = compute_ece(sp_cgjs, sp_loc)
            drift_data[f'{prefix}_sparse_cls'].append(sp_ece_cls)
            drift_data[f'{prefix}_sparse_loc'].append(sp_ece_loc)
            print(f"    Sparse - Cls ECE: {sp_ece_cls:.3f}, Loc ECE: {sp_ece_loc:.3f}")
            
            print("  Evaluating Dense subset...")
            dn_confs, dn_cgjs, dn_cls, dn_loc = extract_confs_and_corrs_crowdhuman(model_path, dense_files, id_to_boxes, val_img_dir)
            dn_ece_cls = compute_ece(dn_confs, dn_cls)
            dn_ece_loc = compute_ece(dn_cgjs, dn_loc)
            drift_data[f'{prefix}_dense_cls'].append(dn_ece_cls)
            drift_data[f'{prefix}_dense_loc'].append(dn_ece_loc)
            print(f"    Dense - Cls ECE: {dn_ece_cls:.3f}, Loc ECE: {dn_ece_loc:.3f}")

    print("\n>>> EVALUATING BASELINE (Confidence Threshold) <<<")
    evaluate_model_set(MODELS_BASELINE, 'baseline')
    
    print("\n>>> EVALUATING CALIBPL (CWPL Proxy) <<<")
    evaluate_model_set(MODELS_CALIBPL, 'calibpl')

    # Plot lines with markers
    plt.figure(figsize=(12, 6))
    
    # Plot Baseline
    plt.plot(iters, drift_data['baseline_dense_loc'], marker='v', linestyle='-', color='red', linewidth=2, label='Baseline - Dense (Loc ECE)')
    plt.plot(iters, drift_data['baseline_dense_cls'], marker='s', linestyle='-', color='salmon', label='Baseline - Dense (Cls ECE)')
    
    # Plot CalibPL
    plt.plot(iters, drift_data['calibpl_dense_loc'], marker='^', linestyle='-', color='blue', linewidth=2, label='CalibPL - Dense (Loc ECE)')
    plt.plot(iters, drift_data['calibpl_dense_cls'], marker='o', linestyle='-', color='lightblue', label='CalibPL - Dense (Cls ECE)')
    
    plt.xticks(iters, ['Supervised', 'Iter 1', 'Iter 2', 'Iter 3'])
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.title("CrowdHuman Dense Scenes: Baseline Drift vs CalibPL Stability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / "crowdhuman_ece_drift.png", dpi=300)
    plt.close()
    
    # Save raw data
    with open(results_dir / "crowdhuman_ece_drift.json", 'w') as f:
        json.dump(drift_data, f, indent=4)
        
    print(f"\nSaved analysis to {results_dir}")

if __name__ == "__main__":
    main()
