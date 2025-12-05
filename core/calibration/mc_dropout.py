"""
MC Dropout Uncertainty for YOLO Detectors
==========================================
Implements Monte Carlo Dropout for uncertainty estimation in object detection.

Key insight for our paper:
- Epistemic uncertainty (model doesn't know) → reject pseudo-labels
- Aleatoric uncertainty (data is ambiguous) → downweight pseudo-labels

This is the core of CalibCoTrain (Contribution 2).

Usage:
    from core.calibration.mc_dropout import MCDropoutDetector
    
    mc = MCDropoutDetector("yolo26n.pt", T=10)
    results = mc.predict_with_uncertainty("shelf_image.jpg")
    
    for det in results:
        print(f"Box: {det['box']}, Conf: {det['mean_conf']:.3f}, "
              f"Epistemic: {det['epistemic']:.3f}, "
              f"Aleatoric: {det['aleatoric']:.3f}")
"""

import numpy as np
import json
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class UncertainDetection:
    """A detection with uncertainty estimates."""
    box: list              # Mean [x1, y1, x2, y2]
    mean_confidence: float # Mean conf across T passes (incl penalties)
    epistemic: float       # Variance of confidence across passes (cls epistemic)
    aleatoric: float       # Mean entropy across passes (cls aleatoric)
    loc_epistemic: float   # Mean spatial variance of bounding box coordinates
    box_variance: list     # Variance per coordinate [var_x1, var_y1, var_x2, var_y2]
    total_uncertainty: float # epistemic + aleatoric + loc_epistemic
    class_id: int
    num_passes: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
        elif isinstance(module, torch.nn.Dropout2d):
            module.train()


def add_dropout_to_model(model, dropout_rate=0.1):
    """
    Add dropout layers before detection heads if not present.
    
    For YOLO models, we inject dropout before the final conv layers
    in the detection head.
    """
    import torch.nn as nn
    
    added = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            new_children = []
            for child in module.children():
                new_children.append(child)
                if isinstance(child, (nn.Conv2d, nn.Linear)):
                    new_children.append(nn.Dropout2d(p=dropout_rate))
                    added += 1
            if added > 0:
                for i, child in enumerate(new_children):
                    module._modules[str(i)] = child
    
    return added


class MCDropoutDetector:
    """
    Monte Carlo Dropout wrapper for YOLO detectors.
    
    Performs T stochastic forward passes with dropout enabled,
    then aggregates predictions to compute uncertainty.
    """
    
    def __init__(
        self,
        weights_path: str,
        T: int = 10,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.5,
        dropout_rate: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Args:
            weights_path: Path to YOLO weights
            T: Number of stochastic forward passes
            conf_threshold: Minimum confidence for detection
            iou_threshold: NMS IoU threshold
            dropout_rate: Dropout probability
            device: 'cpu' or 'cuda'
        """
        from ultralytics import YOLO
        
        self.model = YOLO(weights_path)
        self.T = T
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Inject dropout layers into the YOLOv12 architecture (which has none natively)
        print(f"Injecting Dropout2d (p={self.dropout_rate}) into model heads...")
        num_added = add_dropout_to_model(self.model.model, self.dropout_rate)
        print(f"Added {num_added} dropout layers.")
        
    def predict_with_uncertainty(
        self,
        image_path: str,
        iou_merge_threshold: float = 0.5
    ) -> List[UncertainDetection]:
        """
        Run T stochastic forward passes and compute uncertainty.
        
        Steps:
        1. Run inference T times with dropout enabled
        2. Collect all detections across passes
        3. Cluster detections by spatial proximity (IoU)
        4. For each cluster, compute mean confidence, epistemic, and aleatoric uncertainty
        
        Args:
            image_path: Path to input image
            iou_merge_threshold: IoU threshold for merging detections across passes
        
        Returns:
            List of UncertainDetection with uncertainty estimates
        """
        all_pass_detections = []
        
        for t in range(self.T):
            # Enable dropout for stochastic inference
            enable_dropout(self.model.model)
            
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=True,
                workers=0
            )
            
            pass_dets = []
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for j in range(len(r.boxes)):
                        pass_dets.append({
                            'box': r.boxes.xyxy[j].tolist(),
                            'conf': float(r.boxes.conf[j]),
                            'cls': int(r.boxes.cls[j]),
                            'pass': t,
                        })
            all_pass_detections.append(pass_dets)
        
        # Merge detections across passes
        return self._merge_passes(all_pass_detections, iou_merge_threshold)
    
    def _merge_passes(
        self,
        all_pass_detections: List[List[Dict]],
        iou_threshold: float
    ) -> List[UncertainDetection]:
        """
        Merge detections from T passes into clusters.
        
        For each cluster:
        - mean_confidence = mean of confidences across passes
        - epistemic = variance of confidences (model uncertainty)
        - aleatoric = mean of per-pass entropy (data uncertainty)
        """
        from core.calibration.detection_calibration import compute_iou
        
        # Flatten all detections
        all_dets = []
        for pass_dets in all_pass_detections:
            all_dets.extend(pass_dets)
        
        if not all_dets:
            return []
        
        # Greedy clustering by IoU
        used = [False] * len(all_dets)
        clusters = []
        
        for i, det in enumerate(all_dets):
            if used[i]:
                continue
            
            cluster = [det]
            used[i] = True
            
            for j in range(i + 1, len(all_dets)):
                if used[j]:
                    continue
                if compute_iou(det['box'], all_dets[j]['box']) > iou_threshold:
                    cluster.append(all_dets[j])
                    used[j] = True
            
            clusters.append(cluster)
        
        # Compute uncertainty for each cluster
        uncertain_dets = []
        for cluster in clusters:
            original_confs = np.array([d['conf'] for d in cluster])
            
            # Penalize missing passes (flickering predictions)
            # If the box is not detected in all T passes, pad with 1e-5.
            num_missing = self.T - len(cluster)
            if num_missing > 0:
                confs = np.pad(original_confs, (0, num_missing), 'constant', constant_values=1e-5)
            else:
                confs = original_confs
            
            # Mean confidence across ALL T passes
            mean_conf = float(confs.mean())
            
            # Classification Epistemic uncertainty: RELATIVE variance of padded confidences
            # High relative variance = model is unsure
            # We divide by mean_conf so high-magnitude TPs don't artificially look uncertain
            if mean_conf > 0:
                epistemic = float(confs.var() / mean_conf)
            else:
                epistemic = 0.0
            
            # Classification Aleatoric uncertainty: average entropy over padded confidences
            entropies = []
            for c in confs:
                c_clipped = np.clip(c, 1e-10, 1 - 1e-10)
                entropy = -(c_clipped * np.log(c_clipped) + (1 - c_clipped) * np.log(1 - c_clipped))
                entropies.append(entropy)
            aleatoric = float(np.mean(entropies))
            
            # Localization Epistemic uncertainty: spatial standard deviation of coordinates
            boxes = np.array([d['box'] for d in cluster])
            mean_box = boxes.mean(axis=0).tolist()
            
            if len(cluster) > 1:
                # Standard deviation of [x1, y1, x2, y2] across observed passes
                # Using std dev instead of variance keeps the scale linear
                box_variance = boxes.var(axis=0).tolist()
                box_std = boxes.std(axis=0).tolist()
                loc_epistemic = float(np.mean(box_std))
            else:
                # Only appeared in 1 pass; no measurable variance, but classification
                # epistemic is already maximized due to T-1 missing passes.
                box_variance = [0.0, 0.0, 0.0, 0.0]
                loc_epistemic = 0.0
                
            # Total uncertainty combines classification and structural uncertainty
            total = epistemic + aleatoric + loc_epistemic
            
            uncertain_dets.append(UncertainDetection(
                box=mean_box,
                mean_confidence=mean_conf,
                epistemic=epistemic,
                aleatoric=aleatoric,
                loc_epistemic=loc_epistemic,
                box_variance=box_variance,
                total_uncertainty=total,
                class_id=cluster[0]['cls'],
                num_passes=len(cluster),
            ))
        
        # Sort by confidence
        uncertain_dets.sort(key=lambda x: x.mean_confidence, reverse=True)
        
        return uncertain_dets
    
    def predict_batch_with_uncertainty(
        self,
        image_paths: List[str],
        save_path: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Run MC Dropout on a batch of images.
        
        Returns dict mapping image name → list of uncertain detections.
        """
        results = {}
        
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            
            dets = self.predict_with_uncertainty(img_path)
            results[img_name] = [d.to_dict() for d in dets]
            
            if (i + 1) % 10 == 0:
                n_dets = sum(len(v) for v in results.values())
                print(f"  MC Dropout: {i+1}/{len(image_paths)} images "
                      f"({n_dets} detections)")
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump({
                    'T': self.T,
                    'dropout_rate': self.dropout_rate,
                    'num_images': len(results),
                    'predictions': results,
                }, f, indent=2)
            print(f"  ✓ MC Dropout results saved: {save_path}")
        
        return results
