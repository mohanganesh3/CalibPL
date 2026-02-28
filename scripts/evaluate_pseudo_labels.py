#!/usr/bin/env python3
"""
Pseudo-Label Quality Evaluator
Analyzes the precision, recall, and F1 score of generated pseudo-labels 
against the hidden Ground Truth of the unlabeled set. 
This provides deep theoretical insights into WHY CalibPL + PSS works better.
"""

import os
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

def read_yolo_labels(file_path):
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append({'cls': cls, 'box': [cx-w/2, cy-h/2, cx+w/2, cy+h/2]})
    return boxes

def evaluate_directory(pl_dir, gt_dir, iou_thresh=0.5):
    pl_files = glob.glob(os.path.join(pl_dir, "*.txt"))
    
    total_gt = 0
    total_pl = 0
    true_positives = 0
    
    for pl_file in pl_files:
        filename = os.path.basename(pl_file)
        gt_file = os.path.join(gt_dir, filename)
        
        pl_boxes = read_yolo_labels(pl_file)
        gt_boxes = read_yolo_labels(gt_file)
        
        total_pl += len(pl_boxes)
        total_gt += len(gt_boxes)
        
        gt_matched = set()
        for plb in pl_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gtb in enumerate(gt_boxes):
                if i in gt_matched: continue
                if plb['cls'] != gtb['cls']: continue
                
                iou = compute_iou(plb['box'], gtb['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
            if best_iou >= iou_thresh:
                true_positives += 1
                gt_matched.add(best_gt_idx)
                
    precision = true_positives / max(total_pl, 1)
    recall = true_positives / max(total_gt, 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-6)
    
    return {
        'total_images_labeled': len(pl_files),
        'total_gt_objects': total_gt,
        'total_pseudo_objects': total_pl,
        'true_positives': true_positives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pl-dir', required=True, help="Directory with generated pseudo-labels")
    parser.add_argument('--gt-dir', required=True, help="Original full dataset labels dir for ground truth")
    args = parser.parse_args()
    
    print(f"Evaluating pseudo-labels against Ground Truth...")
    print(f"PL Dir: {args.pl_dir}")
    print(f"GT Dir: {args.gt_dir}")
    
    stats = evaluate_directory(args.pl_dir, args.gt_dir)
    
    print("\n--- RESULTS ---")
    print(f"Images with PLs : {stats['total_images_labeled']}")
    print(f"Total PL Objects: {stats['total_pseudo_objects']}")
    print(f"Total GT Objects: {stats['total_gt_objects']}")
    print(f"Precision       : {stats['precision']:.4f} (How many PLs are correct?)")
    print(f"Recall          : {stats['recall']:.4f} (How many GTs were found?)")
    print(f"F1 Score        : {stats['f1_score']:.4f}")
