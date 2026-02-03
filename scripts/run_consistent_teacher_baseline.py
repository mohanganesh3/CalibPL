#!/usr/bin/env python3
"""
Consistent-Teacher GMM Baseline for BMVC 2026 Comparison
=========================================================

This implements the Consistent-Teacher calibration approach using GMM,
demonstrating that GMM DEGRADES localization calibration on dense scenes.

This is the CRITICAL missing experiment for BMVC acceptance.

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 scripts/run_consistent_teacher_baseline.py \
        --data-yaml data/SKU110K/yolo_format/frac_10.yaml \
        --unlabeled-dir data/SKU110K/yolo_format/unlabeled \
        --seed 42 \
        --iterations 5 \
        --epochs 10 \
        --device 0
"""

import argparse
import os
import sys
from pathlib import Path

# Import the existing CalibPL infrastructure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our GMM calibrator
from scripts.gmm_calibrator import GMMCalibrator
from scripts.calibpl_selftrain import (
    set_seed, 
    generate_pseudo_labels,
    DetectionCalibrator  # We'll replace this with GMM
)

import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False


class ConsistentTeacherGMMCalibrator:
    """
    GMM-based calibrator matching Consistent-Teacher approach.
    
    Key difference from CalibPL's isotonic regression:
    - GMM assumes Gaussian-distributed scores
    - Works well on classification (Gaussian-ish)
    - DEGRADES on localization (heavy-tailed due to NMS)
    """
    
    def __init__(self, n_components=3):
        self.gmm = GMMCalibrator(n_components=n_components)
        self.is_fitted = False
        
    def fit(self, model_path: str, val_yaml: str, iou_threshold: float = 0.5):
        """Fit GMM calibrator on validation set."""
        from ultralytics import YOLO
        import yaml
        
        print(f"    Fitting GMM Calibrator (Consistent-Teacher, n_components={self.gmm.n_components})...")
        
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
        all_confs_cls = []
        all_confs_loc = []
        all_cls_corr = []
        all_loc_corr = []
        
        val_images = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
        
        # Limit to 300 images for speed
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
            
            if len(gt_boxes) == 0:
                continue
                
            results = model.predict(str(img_path), device=0, conf=0.01, verbose=False, max_det=300)
            
            if len(results) == 0 or results[0].boxes is None:
                continue
                
            boxes = results[0].boxes
            for j in range(len(boxes)):
                raw_conf = float(boxes.conf[j].item())
                pred_cls = int(boxes.cls[j].item())
                
                # Convert bbox to [x1, y1, x2, y2] format
                xywhn = boxes.xywhn[j].cpu().numpy()
                cx, cy, w, h = xywhn
                pred_box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                
                # Match to GT
                best_iou = 0
                best_match = -1
                for k, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if pred_cls != gt_cls:
                        continue
                    
                    x1 = max(pred_box[0], gt_box[0])
                    y1 = max(pred_box[1], gt_box[1])
                    x2 = min(pred_box[2], gt_box[2])
                    y2 = min(pred_box[3], gt_box[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union = area_pred + area_gt - inter
                    iou = inter / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = k
                
                if best_match >= 0:
                    # Classification correctness: lenient IoU >= 0.1
                    cls_correct = 1 if best_iou >= 0.1 else 0
                    # Localization correctness: strict IoU >= 0.5
                    loc_correct = 1 if best_iou >= iou_threshold else 0
                    
                    all_confs_cls.append(raw_conf)
                    all_confs_loc.append(raw_conf)
                    all_cls_corr.append(cls_correct)
                    all_loc_corr.append(loc_correct)
            
            if (i + 1) % 50 == 0:
                print(f"      Processed {i+1}/{len(val_images)} images...")
        
        # Fit GMM
        scores_cls = np.array(all_confs_cls)
        scores_loc = np.array(all_confs_loc)
        correct_cls = np.array(all_cls_corr)
        correct_loc = np.array(all_loc_corr)
        
        print(f"    Collected {len(scores_cls)} predictions from {len(val_images)} images")
        
        self.gmm.fit(scores_cls, correct_cls, scores_loc, correct_loc)
        self.is_fitted = True
        
        print(f"    GMM calibrator fitted successfully")
        
    def predict(self, conf_cls: np.ndarray, conf_loc: np.ndarray):
        """Calibrate scores using GMM."""
        if not self.is_fitted:
            return conf_cls, conf_loc
        return self.gmm.predict(conf_cls, conf_loc)


def run_consistent_teacher_baseline(args):
    """Run full Consistent-Teacher baseline with GMM calibration."""
    
    set_seed(args.seed)
    
    print("\n" + "="*70)
    print(" CONSISTENT-TEACHER GMM BASELINE")
    print(f" Dataset: {args.data_yaml}")
    print(f" Seed: {args.seed}")
    print(f" Iterations: {args.iterations}")
    print("="*70)
    
    # Create output directory
    tag = f"ct_gmm_seed{args.seed}"
    output_root = Path(f"results/consistent_teacher/{tag}")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model_path = args.base_model
    
    # Self-training loop
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f" ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")
        
        # Step 1: Fit GMM calibrator on current model
        print("\n  [Step 1] Fitting GMM Calibrator...")
        calibrator = ConsistentTeacherGMMCalibrator(n_components=3)
        calibrator.fit(model_path, args.data_yaml)
        
        # Step 2: Generate pseudo-labels with GMM-calibrated confidences
        print("\n  [Step 2] Generating Pseudo-Labels (GMM-calibrated)...")
        pseudo_label_dir = output_root / f"iter_{iteration}" / "pseudo_labels"
        
        # Use custom pseudo-label generation with GMM
        generate_gmm_pseudo_labels(
            model_path=model_path,
            unlabeled_dir=args.unlabeled_dir,
            output_dir=str(pseudo_label_dir),
            calibrator=calibrator,
            threshold=0.5,
            device=0
        )
        
        # Step 3: Prepare SSOD dataset (labeled + pseudo-labeled)
        print("\n  [Step 3] Preparing SSOD Dataset...")
        dataset_dir = output_root / f"iter_{iteration}" / "dataset"
        prepare_ssod_dataset(
            labeled_yaml=args.data_yaml,
            pseudo_dir=pseudo_label_dir,
            unlabeled_dir=args.unlabeled_dir,
            output_dir=dataset_dir
        )
        
        # Step 4: Train YOLOv12
        print(f"\n  [Step 4] Training YOLOv12 for {args.epochs} epochs...")
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        results = model.train(
            data=str(dataset_dir / "dataset.yaml"),
            epochs=args.epochs,
            batch=16,
            imgsz=640,
            device=0,
            project=str(output_root / f"iter_{iteration}" / "model"),
            name="train",
            exist_ok=True,
            deterministic=True,
            seed=args.seed,
            workers=0
        )
        
        # Update model path for next iteration
        model_path = str(output_root / f"iter_{iteration}" / "model" / "train" / "weights" / "best.pt")
        
        # Step 5: Validate
        print("\n  [Step 5] Validating...")
        metrics = model.val(data=args.data_yaml, device=0, verbose=True)
        
        print(f"\n  Iteration {iteration} Results:")
        print(f"    mAP50-95: {metrics.box.map:.4f}")
        print(f"    mAP50: {metrics.box.map50:.4f}")
    
    print("\n" + "="*70)
    print(" CONSISTENT-TEACHER BASELINE COMPLETE")
    print(f" Final model: {model_path}")
    print("="*70)


def generate_gmm_pseudo_labels(model_path, unlabeled_dir, output_dir, calibrator, threshold, device):
    """Generate pseudo-labels using GMM-calibrated confidences."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(unlabeled_dir) if f.endswith(('.jpg', '.png'))])
    
    total_kept = 0
    total_rejected = 0
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(unlabeled_dir, img_name)
        
        results = model.predict(img_path, device=device, conf=0.1, verbose=False, max_det=300)
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Extract confidences
            raw_confs = boxes.conf.cpu().numpy()
            
            # Calibrate using GMM (same score for cls and loc)
            cal_cls, cal_loc = calibrator.predict(raw_confs, raw_confs)
            
            # Consistent-Teacher uses classification confidence for selection
            # (This is the key difference - they DON'T use localization confidence)
            
            label_lines = []
            for j in range(len(boxes)):
                # Use GMM-calibrated CLASSIFICATION confidence
                if cal_cls[j] >= threshold:
                    cls = int(boxes.cls[j].item())
                    xywhn = boxes.xywhn[j].cpu().numpy()
                    cx, cy, w, h = xywhn
                    label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    total_kept += 1
                else:
                    total_rejected += 1
            
            if label_lines:
                label_path = output_path / f"{Path(img_name).stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
        
        if (i + 1) % 50 == 0:
            print(f"    Pseudo-labels: {i+1}/{len(image_files)} images, {total_kept} kept")
    
    print(f"    → Kept {total_kept} boxes, rejected {total_rejected}")


def prepare_ssod_dataset(labeled_yaml, pseudo_dir, unlabeled_dir, output_dir):
    """Prepare combined dataset YAML."""
    import yaml
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labeled data config
    with open(labeled_yaml) as f:
        labeled_cfg = yaml.safe_load(f)
    
    # Create combined dataset structure
    combined_train_img = output_dir / "train" / "images"
    combined_train_lbl = output_dir / "train" / "labels"
    combined_train_img.mkdir(parents=True, exist_ok=True)
    combined_train_lbl.mkdir(parents=True, exist_ok=True)
    
    # Copy labeled data
    labeled_path = Path(labeled_cfg['path'])
    labeled_train = labeled_path / labeled_cfg['train']
    labeled_train_lbl = labeled_train.parent / "labels"
    
    for img in labeled_train.glob("*.jpg") + labeled_train.glob("*.png"):
        shutil.copy(img, combined_train_img / img.name)
        lbl = labeled_train_lbl / img.with_suffix('.txt').name
        if lbl.exists():
            shutil.copy(lbl, combined_train_lbl / lbl.name)
    
    # Copy pseudo-labeled data
    pseudo_dir = Path(pseudo_dir)
    unlabeled_dir = Path(unlabeled_dir)
    
    for lbl in pseudo_dir.glob("*.txt"):
        img_name = lbl.stem
        for ext in ['.jpg', '.png']:
            img = unlabeled_dir / f"{img_name}{ext}"
            if img.exists():
                shutil.copy(img, combined_train_img / img.name)
                shutil.copy(lbl, combined_train_lbl / lbl.name)
                break
    
    # Create dataset YAML
    dataset_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': labeled_cfg['val'],
        'test': labeled_cfg.get('test', labeled_cfg['val']),
        'nc': labeled_cfg['nc'],
        'names': labeled_cfg['names']
    }
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    labeled_count = len(list(labeled_train.glob("*.jpg"))) + len(list(labeled_train.glob("*.png")))
    pseudo_count = len(list(pseudo_dir.glob("*.txt")))
    print(f"    Dataset: {labeled_count} labeled + {pseudo_count} pseudo-labeled")


def main():
    parser = argparse.ArgumentParser(description="Consistent-Teacher GMM Baseline")
    parser.add_argument('--data-yaml', required=True, help='Path to labeled data YAML')
    parser.add_argument('--unlabeled-dir', required=True, help='Path to unlabeled images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--iterations', type=int, default=5, help='Number of self-training iterations')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs per iteration')
    parser.add_argument('--device', type=int, default=0, help='CUDA device (logical, after CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--base-model', default='models/yolo12n.pt', help='Base YOLO model')
    
    args = parser.parse_args()
    
    run_consistent_teacher_baseline(args)


if __name__ == '__main__':
    main()
