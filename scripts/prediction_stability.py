#!/usr/bin/env python3
"""
Class-Geometry Joint Stability (CGJS) implementation for CalibPL.
Computes detection consistency across spatial and color augmentations,
requiring both geometric overlap and class label agreement.
"""

import os
import tempfile
import cv2
import torch
import numpy as np
from typing import Any
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def compute_cgjs_for_image(
    model, 
    image_path: str, 
    base_results: Any, 
    device: str = '0',
    conf_threshold: float = 0.01,
    iou_match_threshold: float = 0.45,
    use_multi_scale: bool = True,
    lightweight: bool = False
) -> np.ndarray:
    """
    Compute Class-Geometry Joint Stability (CGJS) for each base detection.
    Enhanced with Multi-Scale Consistency (MSC) for research-grade robustness.
    """
    if base_results.boxes is None or len(base_results.boxes) == 0:
        return np.array([])
        
    base_boxes = base_results.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
    base_classes = base_results.boxes.cls.cpu().numpy().astype(int)
    n_base = len(base_boxes)
    stability_counts = np.zeros(n_base)
    
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(n_base)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 1. Multi-View Consistency (Augmentations)
    augs = [
        A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])),
        A.Compose([A.RandomBrightnessContrast(p=1.0)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])),
        A.Compose([A.ShiftScaleRotate(p=1.0, shift_limit=0.05, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT)], 
                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    ]
    if lightweight:
        augs = augs[:2]
        use_multi_scale = False
    
    clamped_boxes = []
    for b in base_boxes:
        clamped_boxes.append([max(0, min(w, b[0])), max(0, min(h, b[1])), max(0, min(w, b[2])), max(0, min(h, b[3]))])
    
    valid_indices = [i for i, b in enumerate(clamped_boxes) if b[2] > b[0] and b[3] > b[1]]
    if not valid_indices:
        return np.zeros(n_base)
    
    valid_clamped_boxes = [clamped_boxes[i] for i in valid_indices]
    valid_labels = [int(base_classes[i]) for i in valid_indices]
    
    total_tests = len(augs)
    
    for aug in augs:
        transformed = aug(image=img, bboxes=valid_clamped_boxes, labels=valid_labels)
        aug_img = transformed['image']
        aug_boxes_gt = transformed['bboxes']
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
        cv2.imwrite(temp_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        try:
            aug_results = model.predict(temp_path, device=device, conf=conf_threshold, verbose=False)
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
        
        if len(aug_results) > 0 and aug_results[0].boxes is not None:
            aug_preds = aug_results[0].boxes.xyxy.cpu().numpy()
            aug_pred_classes = aug_results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, gt_box in enumerate(aug_boxes_gt):
                orig_idx = valid_indices[i]
                target_cls = valid_labels[i]
                for pred_box, pred_cls in zip(aug_preds, aug_pred_classes):
                    if pred_cls == target_cls and compute_iou(gt_box, pred_box) >= iou_match_threshold:
                        stability_counts[orig_idx] += 1
                        break

    # 2. Multi-Scale Consistency (MSC)
    if use_multi_scale:
        scales = [480, 800] # Standard imgsz is 640. Test at smaller and larger scales.
        total_tests += len(scales)
        for s in scales:
            results_scale = model.predict(image_path, device=device, conf=conf_threshold, verbose=False, imgsz=s)
            if len(results_scale) > 0 and results_scale[0].boxes is not None:
                scale_preds = results_scale[0].boxes.xyxy.cpu().numpy()
                scale_classes = results_scale[0].boxes.cls.cpu().numpy().astype(int)
                
                for i in range(n_base):
                    target_box = base_boxes[i]
                    target_cls = base_classes[i]
                    for pred_box, pred_cls in zip(scale_preds, scale_classes):
                        if pred_cls == target_cls and compute_iou(target_box, pred_box) >= iou_match_threshold:
                            stability_counts[i] += 1
                            break
                            
    return stability_counts / total_tests
