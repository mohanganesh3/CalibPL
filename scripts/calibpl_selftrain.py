#!/usr/bin/env python3
"""
CalibPL: Calibration-Informed Pseudo-Label Selection for SSOD
=============================================================
BMVC 2026

Core Self-Training Implementation featuring:
1. Calibration-Informed Adaptive Threshold (CalibPL)
2. Prediction Stability Score (PSS)

Usage:
    python3 scripts/calibpl_selftrain.py --data-yaml data/... --unlabeled-dir ...
"""

import argparse
import os
import sys

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# Parse --train-device early to get GPU assignment
def _early_gpu_setup():
    """Parse --train-device and set CUDA_VISIBLE_DEVICES before torch import."""
    for i, arg in enumerate(sys.argv):
        if arg == '--train-device' and i + 1 < len(sys.argv):
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
            return sys.argv[i + 1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # default
    return '0'

_PHYSICAL_GPU = _early_gpu_setup()

# Now safe to import torch
import json
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Disable CuDNN to prevent K80 crashes with bf16
import torch
torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prediction_stability import compute_cgjs_for_image

def _setup_gpu(physical_gpu: str):
    """Set CUDA_VISIBLE_DEVICES so physical GPU appears as device 0."""
    global _PHYSICAL_GPU
    _PHYSICAL_GPU = str(physical_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = _PHYSICAL_GPU

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# CALIBRATOR
# ============================================================
class DetectionCalibrator:
    """Fits dual isotonic regression calibrators (Cls + Loc) with stratified bootstrapping."""
    
    def __init__(self, bootstrap_iters: int = 5, use_gmm: bool = False):
        self.use_gmm = use_gmm
        self.bootstrap_iters = bootstrap_iters
        self.is_fitted = False
        self.num_samples = 0
        self.empirical_accuracy_cls = None
        self.empirical_accuracy_loc = None
        self.conf_space = np.linspace(0.01, 0.99, 100)
        self.prob_space_cls = None
        self.prob_space_loc = None
        
        if use_gmm:
            # GMM calibration (Consistent-Teacher baseline)
            from scripts.gmm_calibrator import GMMCalibrator
            self.calibrator_cls = GMMCalibrator(n_components=3)
            self.calibrator_loc = None  # GMM uses same calibrator for both
        else:
            # Isotonic regression (CalibPL)
            from sklearn.isotonic import IsotonicRegression
            self.calibrator_cls = IsotonicRegression(out_of_bounds='clip')
            self.calibrator_loc = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, model_path: str, val_yaml: str, iou_threshold: float = 0.5):
        from ultralytics import YOLO
        import yaml
        from scripts.prediction_stability import compute_cgjs_for_image
        
        print(f"    Fitting Dual-Calibrator (Cls + Loc, Bootstrapped iters={self.bootstrap_iters})...")
        
        with open(val_yaml) as f:
            data_cfg = yaml.safe_load(f)
        
        val_field = data_cfg.get('val', 'val/images')
        base_path = Path(data_cfg.get('path', ''))
        val_path = Path(val_field)
        if not val_path.is_absolute():
            val_path = base_path / val_field
            
        if val_path.name == "images":
            val_img_dir = val_path
            val_lbl_dir = val_path.parent / "labels"
        else:
            val_img_dir = val_path / "images"
            val_lbl_dir = val_path / "labels"
            
        model = YOLO(model_path)
        all_confs = []
        all_cgjs = []
        all_cls_corr = []
        all_loc_corr = []
        all_pred_classes = []
        
        val_images = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
        
        # Limit validation to 300 images for speed if dense
        if len(val_images) > 300:
            random.seed(42)
            val_images = random.sample(val_images, 300)
        
        for i, img_path in enumerate(val_images):
            label_path = val_lbl_dir / img_path.with_suffix('.txt').name
            gt_boxes = []
            gt_classes = []
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            gt_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                            gt_classes.append(cls_id)
            gt_boxes = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
            gt_classes = np.array(gt_classes) if gt_classes else np.empty(0)
            
            results = model.predict(str(img_path), device=0, conf=0.01, verbose=False, max_det=100)
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                pred_confs = results[0].boxes.conf.cpu().numpy()
                pred_boxes = results[0].boxes.xyxyn.cpu().numpy()
                pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # We need CGJS as the localization quality signal
                cgjs_scores = compute_cgjs_for_image(model, str(img_path), results[0], device=0, use_multi_scale=False)
                
                cls_corr, loc_corr = self._match_dual(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold)
                
                all_confs.extend(pred_confs.tolist())
                all_cgjs.extend(cgjs_scores.tolist())
                all_cls_corr.extend(cls_corr.tolist())
                all_loc_corr.extend(loc_corr.tolist())
                all_pred_classes.extend(pred_classes.tolist())
                
        if len(all_confs) > 20:
            all_confs = np.array(all_confs)
            all_cgjs = np.array(all_cgjs)
            all_cls_corr = np.array(all_cls_corr)
            all_loc_corr = np.array(all_loc_corr)
            all_pred_classes = np.array(all_pred_classes)
            
            # Stratified Bootstrapping (balance by predicted class)
            from sklearn.isotonic import IsotonicRegression
            unique_classes = np.unique(all_pred_classes)
            class_indices = {cls: np.where(all_pred_classes == cls)[0] for cls in unique_classes}
            
            bootstrap_preds_cls = []
            bootstrap_preds_loc = []
            
            for _ in range(self.bootstrap_iters):
                indices = []
                # Stratified sampling: equal number of samples per class
                target_samples_per_class = max(10, len(all_confs) // (len(unique_classes) * 2))
                for cls, idxs in class_indices.items():
                    if len(idxs) > 0:
                        sampled = np.random.choice(idxs, size=min(len(idxs), target_samples_per_class), replace=True)
                        indices.extend(sampled)
                        
                indices = np.array(indices)
                if len(indices) < 10:
                    indices = np.random.choice(len(all_confs), len(all_confs), replace=True)
                
                ir_cls = IsotonicRegression(out_of_bounds='clip')
                ir_cls.fit(all_confs[indices], all_cls_corr[indices])
                bootstrap_preds_cls.append(ir_cls.predict(self.conf_space))
                
                ir_loc = IsotonicRegression(out_of_bounds='clip')
                ir_loc.fit(all_cgjs[indices], all_loc_corr[indices])
                bootstrap_preds_loc.append(ir_loc.predict(self.conf_space))
            
            self.prob_space_cls = np.mean(bootstrap_preds_cls, axis=0)
            self.prob_space_loc = np.mean(bootstrap_preds_loc, axis=0)
            
            if self.use_gmm:
                # GMM calibration (Consistent-Teacher baseline)
                print(f"    Fitting GMM calibrator (Consistent-Teacher baseline)...")
                scores = np.array(all_confs)
                self.calibrator_cls.fit(scores, np.array(all_cls_corr), scores, np.array(all_loc_corr))
            else:
                # Isotonic regression (CalibPL)
                self.calibrator_cls.fit(all_confs, all_cls_corr)
                self.calibrator_loc.fit(all_cgjs, all_loc_corr)
            
            self.is_fitted = True
            self.num_samples = int(len(all_confs))
            self.empirical_accuracy_cls = float(np.mean(all_cls_corr))
            self.empirical_accuracy_loc = float(np.mean(all_loc_corr))
            
            print(f"    Dual-Calibrator fitted. Global Acc: Cls={self.empirical_accuracy_cls*100:.1f}%, Loc={self.empirical_accuracy_loc*100:.1f}%")
        else:
            print(f"    Warning: Insufficient detections ({len(all_confs)}) for dual calibration.")
            
        del model
        torch.cuda.empty_cache()

    def get_adaptive_threshold(self, target_reliability: float = 0.5) -> Tuple[float, float]:
        """Returns (tau_cls, tau_loc)."""
        if not self.is_fitted:
            return (0.5, 0.5)
            
        tau_cls = 0.9
        valid_cls = np.where(self.prob_space_cls >= target_reliability)[0]
        if len(valid_cls) > 0:
            tau_cls = max(0.1, float(self.conf_space[valid_cls[0]]))
            
        tau_loc = 0.9
        valid_loc = np.where(self.prob_space_loc >= target_reliability)[0]
        if len(valid_loc) > 0:
            tau_loc = max(0.1, float(self.conf_space[valid_loc[0]]))
            
        return (tau_cls, tau_loc)

    def calibrate(self, raw_confidence: float, cgjs_score: float) -> Tuple[float, float]:
        """Returns (cal_cls, cal_loc)."""
        if not self.is_fitted:
            return (float(raw_confidence), float(cgjs_score))
        
        if self.use_gmm:
            # GMM returns calibrated scores
            cal_cls, cal_loc = self.calibrator_cls.predict(
                np.array([raw_confidence]), 
                np.array([raw_confidence])
            )
            return (float(cal_cls[0]), float(cal_loc[0]))
        else:
            # Isotonic regression
            return (
                float(self.calibrator_cls.predict([raw_confidence])[0]),
                float(self.calibrator_loc.predict([cgjs_score])[0])
            )

    def _match_dual(self, pred_boxes, pred_classes, gt_boxes, gt_classes, strict_iou=0.5):
        n = len(pred_boxes)
        if n == 0: return np.array([]), np.array([])
        if len(gt_boxes) == 0: return np.zeros(n), np.zeros(n)
        
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
                if best_iou >= 0.1: # Lenient localization for classification correctness
                    cls_corr[i] = 1
                if best_iou >= strict_iou: # Strict localization
                    loc_corr[i] = 1
                    
        return cls_corr, loc_corr



# ============================================================
# PSEUDO-LABEL GENERATION
# ============================================================
def generate_pseudo_labels(
    model_path: str,
    unlabeled_dir: str,
    output_dir: str,
    method: str, # 'pseudo_label' (fixed), 'calibpl' (adaptive), 'calibpl_cgjs'
    raw_threshold: float,
    score_threshold: float,
    cgjs_threshold: float = 0.5,
    calibrator: Optional[DetectionCalibrator] = None,
    alpha: float = 0.5, # Weight between cal_conf and CGJS
    max_images: int = 0
):
    from ultralytics import YOLO
    model = YOLO(model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(unlabeled_dir) if f.endswith(('.jpg', '.png'))])
    if max_images > 0:
        image_files = random.sample(image_files, min(max_images, len(image_files)))
        
    stats = {
        'total_images': 0,
        'images_with_labels': 0,
        'total_boxes_kept': 0,
        'total_boxes_rejected': 0,
        'raw_threshold': float(raw_threshold),
        'score_threshold': float(score_threshold),
    }
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(unlabeled_dir, img_name)
        
        # Base prediction (low conf to allow PSS to boost or filter)
        base_conf = 0.05 if method == 'calibpl_pss' else 0.1
        results = model.predict(img_path, device=0, conf=base_conf, verbose=False, max_det=300)
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Compute CGJS if needed (Multiplicative Gating in CalibPL or Score Fusion in calibpl_cgjs)
            cgjs_scores = None
            if method in ['calibpl', 'calibpl_cgjs']:
                cgjs_scores = compute_cgjs_for_image(model, img_path, results[0], device=0, use_multi_scale=False)
            
            label_lines = []
            for j in range(len(boxes)):
                cls = int(boxes.cls[j].item())
                raw_conf = float(boxes.conf[j].item())
                cx, cy, w, h = boxes.xywhn[j].cpu().numpy()
                cgjs = float(cgjs_scores[j]) if cgjs_scores is not None else 0.0
                
                if calibrator:
                    cal_cls, cal_loc = calibrator.calibrate(raw_conf, cgjs)
                else:
                    cal_cls, cal_loc = raw_conf, 1.0
                
                decision_threshold = score_threshold # Either target_reliability or fixed_threshold

                if method == 'calibpl':
                    # V5 Dual-Calibrator Acceptance Gate
                    # Requires reliability in both classification AND localization probability spaces
                    # PLUS raw geometric stability
                    if cal_cls >= decision_threshold and cal_loc >= decision_threshold and cgjs >= cgjs_threshold:
                        label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        stats['total_boxes_kept'] += 1
                    else:
                        stats['total_boxes_rejected'] += 1
                elif method == 'calibpl_cgjs':
                    score = alpha * cal_cls + (1 - alpha) * cgjs
                    if score >= decision_threshold:
                        label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        stats['total_boxes_kept'] += 1
                    else:
                        stats['total_boxes_rejected'] += 1
                else: # Original pseudo_label method
                    if raw_conf >= raw_threshold:
                        label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        stats['total_boxes_kept'] += 1
                    else:
                        stats['total_boxes_rejected'] += 1
            
            if label_lines:
                label_path = output_path / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
                stats['images_with_labels'] += 1
                
        stats['total_images'] += 1
        if (i + 1) % 50 == 0:
            print(f"    Pseudo-labels: {i+1}/{len(image_files)} images, {stats['total_boxes_kept']} kept")
            
    del model
    torch.cuda.empty_cache()
    return stats


# ============================================================
# DATASET PREP AND TRAINING
# ============================================================
def prepare_dataset(original_yaml, pseudo_dir, unlabeled_dir, output_dir):
    import yaml
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    
    with open(original_yaml) as f:
        orig = yaml.safe_load(f)
    orig_path = Path(orig['path'])
    
    # 1. Symlink original labeled data
    count_orig = 0
    for subdir in ["images", "labels"]:
        src_dir = orig_path / "train" / subdir
        dst_dir = output_dir / "train" / subdir
        if src_dir.exists():
            for f in src_dir.iterdir():
                dst = dst_dir / f.name
                if not dst.exists():
                    os.symlink(f.resolve(), dst)
                    if subdir == "images": count_orig += 1
                    
    # 2. Add pseudo labels
    count_pseudo = 0
    pseudo_dir = Path(pseudo_dir)
    if pseudo_dir.exists():
        for label_file in pseudo_dir.glob("*.txt"):
            dst_label = train_labels / label_file.name
            if not dst_label.exists():
                os.symlink(label_file.resolve(), dst_label)

            src_img = None
            for ext in ('.jpg', '.png', '.jpeg'):
                candidate = Path(unlabeled_dir) / f"{label_file.stem}{ext}"
                if candidate.exists():
                    src_img = candidate
                    break

            if src_img is not None:
                dst_img = train_images / src_img.name
                if not dst_img.exists():
                    os.symlink(src_img.resolve(), dst_img)
                    count_pseudo += 1
                
    # 3. Copy val
    val_dir = output_dir / "val"
    if not val_dir.exists():
        val_images = val_dir / "images"
        val_labels = val_dir / "labels"
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)
        
        val_field = Path(orig.get('val', 'val/images'))
        if not val_field.is_absolute():
            val_field = orig_path / val_field
            
        src_images = val_field if val_field.name == "images" else val_field / "images"
        src_labels = val_field.parent / "labels" if val_field.name == "images" else val_field / "labels"
        
        for subdir, src_d in [("images", src_images), ("labels", src_labels)]:
            if src_d.exists():
                for f in src_d.iterdir():
                    dst = val_dir / subdir / f.name
                    if not dst.exists():
                        os.symlink(f.resolve(), dst)
                        
    # 4. Write YAML
    yaml_content = {
        'path': str(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': orig.get('nc', 1),
        'names': orig.get('names', {0: 'product'})
    }
    with open(output_dir / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)
        
    print(f"    Dataset: {count_orig} labeled + {count_pseudo} pseudo-labeled")
    return output_dir / "dataset.yaml"

def train_iteration(model_weights, dataset_yaml, output_dir, epochs=10, batch_size=16):
    from ultralytics import YOLO
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    model = YOLO(model_weights)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=0,
        optimizer='SGD',
        project=output_dir,
        name='train',
        exist_ok=True,
        amp=False,
        half=False,
        workers=0,
        verbose=True
    )
    
    best_weights = str(Path(output_dir) / 'train' / 'weights' / 'best.pt')
    if getattr(results, 'box', None) is not None:
        map50 = float(results.box.map50)
    else:
        val_model = YOLO(best_weights)
        val_res = val_model.val(data=str(dataset_yaml), device=0, split='val', verbose=False)
        map50 = float(val_res.box.map50)
        del val_model
        
    del model
    torch.cuda.empty_cache()
    return best_weights, map50


# ============================================================
# MAIN SSOD LOOP
# ============================================================
def run_ssod(args):
    # Set GPU before anything else
    _setup_gpu(args.train_device)
    set_seed(args.seed)
    
    # Base model
    model_weights = str(PROJECT_ROOT / "models" / "yolo12n.pt")
    
    # If running on SKU-110K, baseline is in week1_baselines. For COCO, we will train a supervised baseline on iter 0.
    if "sku" in args.data_yaml.lower() and os.path.exists(PROJECT_ROOT / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt"):
        model_weights = str(PROJECT_ROOT / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt")
        print(f"Using pretrained baseline: {model_weights}")
    
    output_dir = PROJECT_ROOT / "results" / "calibpl_v3" / f"{args.method}_seed{args.seed}_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f" CalibPL SSOD Framework")
    print(f" Method: {args.method}")
    print(f" Dataset: {args.data_yaml}")
    print(f"{'='*70}")
    
    results_log = {'method': args.method, 'tag': args.tag, 'seed': args.seed, 'iterations': []}
    
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f" ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")
        
        iter_dir = output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Calibrator & Threshold
        calibrator = None
        threshold_raw = args.fixed_threshold
        threshold_score = args.fixed_threshold
        if args.method in ['calibpl', 'calibpl_cgjs']:
            print("\n  [Step 1] Fitting Dual Calibrator (Cls + Loc)...")
            calibrator = DetectionCalibrator(use_gmm=args.use_gmm)
            calibrator.fit(model_weights, args.data_yaml)
            
            tau_cls, tau_loc = calibrator.get_adaptive_threshold(args.target_reliability)
            threshold_raw = tau_cls # Just for logging purposes to keep backwards compatibility
            threshold_score = args.target_reliability if calibrator.is_fitted else args.fixed_threshold
            print(f"    → CalibPL derived raw thresholds: τ_cls* = {tau_cls:.3f}, τ_loc* = {tau_loc:.3f}")
            print(f"    → Reliability-space decision threshold = {threshold_score:.3f}")
        else:
            print(f"\n  [Step 1] Using Fixed Threshold τ = {threshold_raw:.3f}")
            
        # 2. Pseudo-Label Generation
        print(f"\n  [Step 2] Generating Pseudo-Labels ({args.method})...")
        pseudo_dir = iter_dir / "pseudo_labels"
        stats = generate_pseudo_labels(
            model_weights, args.unlabeled_dir, str(pseudo_dir),
            method=args.method, raw_threshold=threshold_raw,
            score_threshold=threshold_score, 
            cgjs_threshold=args.cgjs_threshold,
            calibrator=calibrator,
            alpha=args.cgjs_alpha, max_images=args.max_pseudo
        )
        print(f"    → Kept {stats['total_boxes_kept']} boxes, rejected {stats['total_boxes_rejected']}")
        
        # 3. Prepare Dataset
        print(f"\n  [Step 3] Preparing SSOD Dataset...")
        combo_dir = iter_dir / "dataset"
        ds_yaml = prepare_dataset(args.data_yaml, pseudo_dir, args.unlabeled_dir, combo_dir)
        
        # 4. Train Model
        print(f"\n  [Step 4] Training YOLOv12 for {args.epochs} epochs...")
        new_weights, map50 = train_iteration(
            model_weights, ds_yaml, str(iter_dir / "model"), 
            epochs=args.epochs, batch_size=args.batch_size
        )
        print(f"    → Iteration {iteration} mAP50: {map50:.4f}")
        
        model_weights = new_weights
        
        # Log
        results_log['iterations'].append({
            'iteration': iteration,
            'threshold_raw': threshold_raw,
            'threshold_score': threshold_score,
            'calibrator_fitted': bool(calibrator and calibrator.is_fitted),
            'calibrator_samples': int(calibrator.num_samples) if calibrator else 0,
            'calibrator_empirical_accuracy_cls': calibrator.empirical_accuracy_cls if calibrator else None,
            'calibrator_empirical_accuracy_loc': calibrator.empirical_accuracy_loc if calibrator else None,
            'pseudo_stats': stats,
            'map50': map50
        })
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(results_log, f, indent=2)

    print(f"\n{'='*70}")
    print(f" EXPERIMENT COMPLETE")
    maps = [r['map50'] for r in results_log['iterations']]
    print(f" mAP50 Progress: {[float(f'{m:.4f}') for m in maps]}")
    print(f" BEST mAP50: {max(maps):.4f}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-legacy', choices=['pseudo_label', 'calibpl', 'calibpl_pss'], required=False)
    parser.add_argument('--data-yaml', required=True)
    parser.add_argument('--unlabeled-dir', required=True)
    parser.add_argument('--tag', required=True)
    
    parser.add_argument('--train-device', default='0')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-pseudo', type=int, default=0) # 0 = all
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--method", type=str, default='calibpl', choices=['pseudo_label', 'calibpl', 'calibpl_cgjs', 'calibpl_gmm'])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--target-reliability", type=float, default=0.6)
    parser.add_argument("--cgjs-threshold", type=float, default=0.5, help="Gate for CGJS stability")
    parser.add_argument("--cgjs-alpha", type=float, default=0.5, help="Weight for score fusion")
    parser.add_argument("--use-gmm", action='store_true', help="Use GMM calibration (Consistent-Teacher baseline)")
    args = parser.parse_args()
    run_ssod(args)

if __name__ == '__main__':
    main()
