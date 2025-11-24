#!/usr/bin/env python3
"""
Week 1: Train RT-DETRv2 and YOLOv12 baselines on SKU-110K.
Also creates SSOD label-fraction splits (0.1%, 1%, 5%, 10%).
Saves all predictions with confidence scores + raw logits for calibration analysis.

Usage:
    # Create SSOD splits first
    python scripts/train_baselines.py --create-splits

    # Train models
    python scripts/train_baselines.py --model rtdetr
    python scripts/train_baselines.py --model yolov12
    python scripts/train_baselines.py --model all

    # Evaluate only (after training)
    python scripts/train_baselines.py --model rtdetr --eval-only
"""

import argparse
import json
import os
import sys
import random
import shutil
from pathlib import Path
from datetime import datetime

# Use all 4 GPUs (Tesla K80 with CUDA 11.4)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')

# Bypass NCCL initialization crashes on K80 (driver mismatch) by forcing Gloo communication
os.environ['ULTRALYTICS_DDP_BACKEND'] = 'gloo'

# K80 fix: disable cuDNN (sm_37 not supported by cuDNN 8.x)
# Uses direct CUDA conv kernels instead — slightly slower but works
import torch
torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
DATA_YAML = PROJECT_ROOT / "data" / "SKU110K" / "sku110k.yaml"
YOLO_DIR = PROJECT_ROOT / "data" / "SKU110K" / "yolo_format"
RESULTS_DIR = PROJECT_ROOT / "results" / "week1_baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
MODEL_CONFIG = {
    'rtdetr': {
        'weights': str(MODELS_DIR / 'rtdetr-l.pt'),
        'display_name': 'RT-DETRv2-l',
        'type': 'rtdetr',
    },
    'yolov12': {
        'weights': str(MODELS_DIR / 'yolo12n.pt'),
        'display_name': 'YOLOv12n',
        'type': 'yolo',
    },
}


def create_ssod_splits(seed=42):
    """
    Create standardized SSOD label-fraction splits for SKU-110K.
    
    Splits: 0.1%, 1%, 5%, 10% of the 1,400 training images.
    This is a sub-contribution of the paper — reproducible SSOD splits
    for dense retail benchmarking.
    """
    random.seed(seed)
    
    train_images = sorted(os.listdir(YOLO_DIR / "train" / "images"))
    n_total = len(train_images)
    
    print(f"\n{'='*70}")
    print(f"CREATING SSOD LABEL-FRACTION SPLITS")
    print(f"{'='*70}")
    print(f"Total training images: {n_total}")
    
    splits_dir = PROJECT_ROOT / "data" / "SKU110K" / "ssod_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle indices
    indices = list(range(n_total))
    random.shuffle(indices)
    
    fractions = {
        '0.1pct': max(1, int(n_total * 0.001)),  # ~1-2 images (extreme sparse)
        '1pct': max(5, int(n_total * 0.01)),      # ~14 images
        '5pct': int(n_total * 0.05),              # ~70 images
        '10pct': int(n_total * 0.10),             # ~140 images
    }
    
    split_info = {}
    for name, n_labeled in fractions.items():
        labeled_indices = indices[:n_labeled]
        unlabeled_indices = indices[n_labeled:]
        
        labeled_images = [train_images[i] for i in labeled_indices]
        unlabeled_images = [train_images[i] for i in unlabeled_indices]
        
        # Save split files
        split_file = splits_dir / f"{name}_labeled.txt"
        with open(split_file, 'w') as f:
            for img in labeled_images:
                f.write(f"train/images/{img}\n")
        
        unlabeled_file = splits_dir / f"{name}_unlabeled.txt"
        with open(unlabeled_file, 'w') as f:
            for img in unlabeled_images:
                f.write(f"train/images/{img}\n")
        
        split_info[name] = {
            'n_labeled': n_labeled,
            'n_unlabeled': len(unlabeled_images),
            'labeled_file': str(split_file),
            'unlabeled_file': str(unlabeled_file),
        }
        
        print(f"  {name}: {n_labeled} labeled / {len(unlabeled_images)} unlabeled")
    
    # Save split metadata
    meta_file = splits_dir / "split_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump({
            'seed': seed,
            'total_train': n_total,
            'splits': split_info,
            'created': datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n✓ SSOD splits saved to: {splits_dir}")
    print(f"✓ Metadata: {meta_file}")
    return split_info


def train_model(model_name: str, epochs: int = 50):
    """Train a model on SKU-110K."""
    config = MODEL_CONFIG[model_name]
    
    print(f"\n{'='*70}")
    print(f"TRAINING {config['display_name']} BASELINE")
    print(f"{'='*70}")
    print(f"Weights: {config['weights']}")
    print(f"Dataset: SKU-110K (1,400 train / 200 val)")
    print(f"Epochs:  {epochs}")
    # Detect device
    import torch
    device = ','.join(str(i) for i in range(torch.cuda.device_count())) if torch.cuda.is_available() else 'cpu'
    print(f"Device:  {'GPU ' + device if device != 'cpu' else 'CPU'}")
    print(f"{'='*70}\n")
    
    if config['type'] == 'rtdetr':
        from ultralytics import RTDETR
        model = RTDETR(config['weights'])
    else:
        from ultralytics import YOLO
        model = YOLO(config['weights'])
    
    output_dir = RESULTS_DIR / model_name
    
    # Training hyperparameters — GPU-optimized for 4x K80 (11GB VRAM each)
    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=8 if config['type'] == 'rtdetr' else 64,  # DDP on 4 GPUs (2 per GPU for RT-DETR)
        imgsz=640,
        project=str(output_dir),
        name='train',
        device=device,
        optimizer='AdamW' if config['type'] == 'rtdetr' else 'SGD',
        lr0=0.0001 if config['type'] == 'rtdetr' else 0.01,
        weight_decay=0.0001 if config['type'] == 'rtdetr' else 0.0005,
        warmup_epochs=3,
        patience=20,
        save=True,
        save_period=10,
        verbose=True,
        amp=False,           # Disabled for K80 stability
        workers=4,
        exist_ok=True,
    )
    
    # RT-DETR doesn't support momentum param
    if config['type'] != 'rtdetr':
        train_kwargs['momentum'] = 0.937
    
    results = model.train(**train_kwargs)
    
    best_weights = output_dir / "train" / "weights" / "best.pt"
    print(f"\n✓ {config['display_name']} training complete")
    print(f"  Best weights: {best_weights}")
    
    return best_weights


def evaluate_model(model_name: str, weights_path: str):
    """Evaluate model and save ALL predictions with confidence scores."""
    config = MODEL_CONFIG[model_name]
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {config['display_name']}")
    print(f"{'='*70}")
    
    if config['type'] == 'rtdetr':
        from ultralytics import RTDETR
        model = RTDETR(str(weights_path))
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
    
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Evaluate on val set
    val_results = model.val(
        data=str(DATA_YAML),
        batch=8 if config['type'] == 'rtdetr' else 64,
        imgsz=640,
        device=device,
        verbose=True,
        split='val',
    )
    
    val_metrics = {
        'mAP50': float(val_results.box.map50),
        'mAP75': float(val_results.box.map75),
        'mAP50-95': float(val_results.box.map),
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
    }
    
    print(f"\nVal Results:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Run inference on test set — save ALL detections for calibration
    test_txt = YOLO_DIR / "test.txt"
    with open(test_txt) as f:
        test_paths = [line.strip() for line in f if line.strip()]
    
    print(f"\nRunning inference on {len(test_paths)} test images...")
    print(f"  Conf threshold: 0.01 (capture all detections for LaECE)")
    
    all_predictions = []
    for i, img_path in enumerate(test_paths):
        if not os.path.isabs(img_path):
            img_path = str(YOLO_DIR / img_path)
        
        if not os.path.exists(img_path):
            continue
        
        results = model.predict(
            source=img_path,
            conf=0.01,    # Very low — we need ALL detections for calibration metrics
            iou=0.5,
            device=device,
            verbose=False,
        )
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for j in range(len(boxes)):
                    pred = {
                        'image': os.path.basename(img_path),
                        'box': boxes.xyxy[j].tolist(),
                        'confidence': float(boxes.conf[j]),
                        'class': int(boxes.cls[j]),
                    }
                    all_predictions.append(pred)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_paths)} images ({len(all_predictions)} detections)")
    
    print(f"\n✓ Total detections on test set: {len(all_predictions)}")
    
    # Save predictions
    pred_file = RESULTS_DIR / f"{model_name}_predictions.json"
    with open(pred_file, 'w') as f:
        json.dump({
            'model': config['display_name'],
            'model_key': model_name,
            'timestamp': datetime.now().isoformat(),
            'val_metrics': val_metrics,
            'num_test_images': len(test_paths),
            'num_detections': len(all_predictions),
            'predictions': all_predictions,
        }, f, indent=2)
    
    print(f"✓ Predictions saved: {pred_file}")
    
    # Save metrics summary
    metrics_file = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'model': config['display_name'],
            'model_key': model_name,
            'weights': str(weights_path),
            'val_metrics': val_metrics,
            'num_detections': len(all_predictions),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"✓ Metrics saved: {metrics_file}")
    return val_metrics


def main():
    parser = argparse.ArgumentParser(description="Train baselines for calibration paper")
    parser.add_argument('--model', choices=['rtdetr', 'yolov12', 'all'], default='rtdetr',
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--eval-only', action='store_true',
                       help='Skip training, only evaluate with existing weights')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create SSOD label-fraction splits and exit')
    args = parser.parse_args()
    
    if args.create_splits:
        create_ssod_splits()
        return
    
    models_to_run = list(MODEL_CONFIG.keys()) if args.model == 'all' else [args.model]
    
    all_metrics = {}
    for model_name in models_to_run:
        config = MODEL_CONFIG[model_name]
        
        if not args.eval_only:
            best_weights = train_model(model_name, epochs=args.epochs)
        else:
            best_weights = RESULTS_DIR / model_name / "train" / "weights" / "best.pt"
            if not best_weights.exists():
                # Fall back to pretrained weights for quick evaluation
                best_weights = Path(config['weights'])
                if not best_weights.exists():
                    print(f"✗ No weights found for {model_name}")
                    continue
                print(f"  Using pretrained weights: {best_weights}")
        
        metrics = evaluate_model(model_name, best_weights)
        all_metrics[model_name] = metrics
    
    # Print summary
    print(f"\n{'='*70}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'mAP50':>8} {'mAP75':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for model_name, metrics in all_metrics.items():
        print(f"{MODEL_CONFIG[model_name]['display_name']:<20} "
              f"{metrics['mAP50']:>8.4f} {metrics['mAP75']:>8.4f} "
              f"{metrics['mAP50-95']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>8.4f}")
    
    # Save combined summary
    summary_file = RESULTS_DIR / "baseline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models': all_metrics,
        }, f, indent=2)
    print(f"\n✓ Summary saved: {summary_file}")


if __name__ == '__main__':
    main()
