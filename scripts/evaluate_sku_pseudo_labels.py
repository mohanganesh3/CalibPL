#!/usr/bin/env python3
import os
from pathlib import Path
import json

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
GT_DIR = PROJ / "data" / "SKU110K" / "yolo_format" / "labels" / "train"

def evaluate_iteration_pseudo_labels(pseudo_dir):
    """Computes Precision, Recall, and counts for a directory of pseudo-labels."""
    pl_dir = Path(pseudo_dir)
    if not pl_dir.exists():
        return None
        
    txt_files = list(pl_dir.glob("*.txt"))
    if not txt_files:
        return None
        
    tp = 0
    fp = 0
    fn = 0
    total_gt = 0
    total_pl = 0
    
    for pl_file in txt_files:
        img_name = pl_file.name
        gt_file = GT_DIR / img_name
        
        # Load PL
        pred_boxes = []
        with open(pl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    pred_boxes.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, cls))
        
        # Load GT
        gt_boxes = []
        if gt_file.exists():
            with open(gt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        gt_boxes.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, cls))
        
        total_gt += len(gt_boxes)
        total_pl += len(pred_boxes)
        
        matched_gt = set()
        
        for px1, py1, px2, py2, pcls in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gx1, gy1, gx2, gy2, gcls) in enumerate(gt_boxes):
                if pcls != gcls: continue
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
    
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "total_pl": total_pl,
        "total_gt": total_gt
    }

def main():
    base_dir = PROJ / "results" / "phase_c_calibpl"
    results_dir = PROJ / "results" / "ablations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== SKU-110K 10% Per-Iteration Pseudo-Label Statistics ===")
    print(f"{'Iter':<6} | {'Pseudo-Labels':<14} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    
    stats = {}
    
    for i in range(1, 6):
        iter_dir = base_dir / f"pseudo_label_seed123_sku20" / f"iter_{i}" / "labels" # Guessing path, will adjust if needed
        # Let's check dynamically exactly what's inside phrase_c_calibpl
        # Wait, the path is probably like pseudo_label_seed{SEED}_frac10/iter_{i}/labels
        pass
        
    # We will search the tree
    found_dirs = list(base_dir.rglob("iter_*/labels"))
    found_dirs.sort()
    
    for d in found_dirs[:5]:
        iter_name = d.parent.name
        metrics = evaluate_iteration_pseudo_labels(d)
        if metrics:
            stats[iter_name] = metrics
            print(f"{iter_name:<6} | {metrics['total_pl']:<14} | {metrics['precision']:.4f}     | {metrics['recall']:.4f}")
            
    with open(results_dir / "sku110k_per_iteration_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
