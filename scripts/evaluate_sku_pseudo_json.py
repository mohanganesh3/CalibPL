#!/usr/bin/env python3
import os
from pathlib import Path
import json

PROJ = Path("/home/mohanganesh/retail-shelf-detection")

def load_coco_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    img_dict = {img['id']: img for img in data['images']}
    boxes_by_filename = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        filename = Path(img_dict[img_id]['file_name']).name
        boxes_by_filename.setdefault(filename, []).append(ann)
        
    return boxes_by_filename, img_dict

def evaluate_iteration_pseudo_labels(pseudo_json, gt_boxes_by_filename):
    if not pseudo_json.exists():
        return None
        
    with open(pseudo_json) as f:
        pl_data = json.load(f)
        
    img_dict = {img['id']: img for img in pl_data['images']}
    pl_boxes_by_filename = {}
    for ann in pl_data['annotations']:
        img_id = ann['image_id']
        filename = Path(img_dict[img_id]['file_name']).name
        pl_boxes_by_filename.setdefault(filename, []).append(ann)
        
    tp = 0
    fp = 0
    fn = 0
    total_gt = 0
    total_pl = 0
    
    for img in pl_data['images']:
        filename = Path(img['file_name']).name
        
        preds = pl_boxes_by_filename.get(filename, [])
        gts = gt_boxes_by_filename.get(filename, [])
        
        total_gt += len(gts)
        total_pl += len(preds)
        
        # Format boxes as [x1, y1, x2, y2]
        pred_boxes = []
        for p in preds:
            x, y, w, h = p['bbox']
            pred_boxes.append((x, y, x+w, y+h, p['category_id']))
            
        gt_boxes = []
        for g in gts:
            x, y, w, h = g['bbox']
            gt_boxes.append((x, y, x+w, y+h, g['category_id']))
            
        matched_gt = set()
        
        for px1, py1, px2, py2, pcls in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gx1, gy1, gx2, gy2, gcls) in enumerate(gt_boxes):
                if pcls != gcls: continue # Usually category_id is 1 or 0
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
    base_dir = PROJ / "results" / "phase_c_calibpl" / "sku10" / "seed42"
    gt_json = PROJ / "data" / "SKU110K" / "annotations" / "train.json"
    
    if not gt_json.exists():
        print(f"Ground truth not found at {gt_json}")
        return
        
    print("Loading Ground Truth...")
    gt_boxes_by_img, _ = load_coco_annotations(gt_json)
    
    stats = {}
    print("=== SKU-110K 10% Per-Iteration Pseudo-Label Statistics ===")
    print(f"{'Iter':<6} | {'Pseudo-Labels':<14} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    
    for i in range(1, 6):
        iter_name = f"iter_{i}"
        pseudo_json = base_dir / iter_name / "pseudo_train.json"
        
        metrics = evaluate_iteration_pseudo_labels(pseudo_json, gt_boxes_by_img)
        if metrics:
            stats[iter_name] = metrics
            print(f"{iter_name:<6} | {metrics['total_pl']:<14} | {metrics['precision']:.4f}     | {metrics['recall']:.4f}")
            
    results_dir = PROJ / "results" / "ablations"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "sku110k_per_iteration_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
