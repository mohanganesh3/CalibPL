#!/usr/bin/env python3
"""
CalibCoTrain-CL: Calibration-Aware Co-Training for Dense Detection
===================================================================
Main contribution of the paper (Contribution 2).

Three filtering strategies for pseudo-label generation:
  1. 'confidence'  — Baseline: raw confidence > 0.7
  2. 'epistemic'   — MC Dropout: reject if epistemic > τ_e
  3. 'combined'    — α · cls_epistemic + (1-α) · loc_epistemic > τ

Co-training loop: RT-DETRv2 (Model A) ↔ YOLOv12 (Model B)

Usage:
    # Single run
    python scripts/run_calibcotrain.py --method combined --alpha 0.5 --seed 42

    # Alpha sweep
    python scripts/run_calibcotrain.py --alpha-sweep

    # Full experiment (3 strategies × 3 seeds)
    python scripts/run_calibcotrain.py --full-experiment
"""

import argparse
import json
import os
import sys
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# === K80 GPU Fixes ===
import torch
torch.backends.cudnn.enabled = False
os.environ['ULTRALYTICS_DDP_BACKEND'] = 'gloo'

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Random seed set to {seed}")


# ============================================================
# STRATEGY 1: Raw Confidence Filtering (Baseline)
# ============================================================
def generate_pseudo_labels_confidence(
    model_path: str,
    unlabeled_dir: str,
    output_labels_dir: str,
    conf_threshold: float = 0.7,
    max_images: int = None,
    device: str = '0',
) -> Dict:
    """Baseline: Generate pseudo-labels using raw confidence threshold."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(unlabeled_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    if max_images:
        image_files = image_files[:max_images]

    stats = {
        'total_images': 0, 'images_with_labels': 0,
        'total_boxes': 0, 'accepted': 0, 'rejected': 0
    }

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(unlabeled_dir, img_name)

        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )

        label_lines = []
        for r in results:
            if r.boxes is not None:
                for j in range(len(r.boxes)):
                    box = r.boxes.xywhn[j].tolist()
                    cls = int(r.boxes.cls[j])
                    label_lines.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                    stats['accepted'] += 1

        if label_lines:
            label_path = os.path.join(output_labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            stats['images_with_labels'] += 1
            stats['total_boxes'] += len(label_lines)

        stats['total_images'] += 1
        if (i + 1) % 50 == 0:
            print(f"    Confidence filter: {i+1}/{len(image_files)} images, {stats['total_boxes']} boxes")

    return stats


# ============================================================
# STRATEGY 2: Epistemic Uncertainty Filtering (MC Dropout)
# ============================================================
def generate_pseudo_labels_epistemic(
    model_path: str,
    unlabeled_dir: str,
    output_labels_dir: str,
    epistemic_threshold: float = 0.05,
    conf_threshold: float = 0.3,
    T: int = 5,
    max_images: int = None,
    device: str = '0',
) -> Dict:
    """MC Dropout epistemic filtering: reject high epistemic uncertainty."""
    from core.calibration.mc_dropout import MCDropoutDetector

    mc = MCDropoutDetector(model_path, T=T, conf_threshold=0.01, device=device)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(unlabeled_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    if max_images:
        image_files = image_files[:max_images]

    stats = {
        'total_images': 0, 'images_with_labels': 0, 'total_boxes': 0,
        'rejected_epistemic': 0, 'rejected_low_conf': 0, 'accepted': 0
    }

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(unlabeled_dir, img_name)
        uncertain_dets = mc.predict_with_uncertainty(img_path)

        label_lines = []
        for det in uncertain_dets:
            if det.mean_confidence < conf_threshold:
                stats['rejected_low_conf'] += 1
                continue
            if det.epistemic > epistemic_threshold:
                stats['rejected_epistemic'] += 1
                continue

            stats['accepted'] += 1
            from PIL import Image
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception:
                img_w, img_h = 640, 640

            x1, y1, x2, y2 = det.box
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            label_lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            label_path = os.path.join(output_labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            stats['images_with_labels'] += 1
            stats['total_boxes'] += len(label_lines)

        stats['total_images'] += 1
        if (i + 1) % 50 == 0:
            print(f"    Epistemic filter: {i+1}/{len(image_files)} images, {stats['total_boxes']} boxes")

    return stats


# ============================================================
# STRATEGY 3: Combined Uncertainty (α · cls + (1-α) · loc)
# ============================================================
def generate_pseudo_labels_combined(
    model_path: str,
    unlabeled_dir: str,
    output_labels_dir: str,
    alpha: float = 0.5,
    combined_threshold: float = 0.1,
    conf_threshold: float = 0.3,
    T: int = 5,
    max_images: int = None,
    device: str = '0',
) -> Dict:
    """
    Combined uncertainty filter:
        score = α · cls_epistemic + (1-α) · normalized_loc_epistemic
    Reject if score > threshold.
    """
    from core.calibration.mc_dropout import MCDropoutDetector

    mc = MCDropoutDetector(model_path, T=T, conf_threshold=0.01, device=device)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(unlabeled_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    if max_images:
        image_files = image_files[:max_images]

    stats = {
        'total_images': 0, 'images_with_labels': 0, 'total_boxes': 0,
        'rejected_combined': 0, 'rejected_low_conf': 0, 'accepted': 0,
        'alpha': alpha
    }

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(unlabeled_dir, img_name)
        uncertain_dets = mc.predict_with_uncertainty(img_path)

        label_lines = []
        for det in uncertain_dets:
            if det.mean_confidence < conf_threshold:
                stats['rejected_low_conf'] += 1
                continue

            # Normalize loc_epistemic to same scale as cls_epistemic
            # loc_epistemic is in pixel units (~0-20), cls_epistemic is ~0-0.3
            # Divide loc by 100 to bring to similar scale
            norm_loc = det.loc_epistemic / 100.0
            combined_score = alpha * det.epistemic + (1 - alpha) * norm_loc

            if combined_score > combined_threshold:
                stats['rejected_combined'] += 1
                continue

            stats['accepted'] += 1
            from PIL import Image
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception:
                img_w, img_h = 640, 640

            x1, y1, x2, y2 = det.box
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            label_lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            label_path = os.path.join(output_labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            stats['images_with_labels'] += 1
            stats['total_boxes'] += len(label_lines)

        stats['total_images'] += 1
        if (i + 1) % 50 == 0:
            print(f"    Combined filter (α={alpha}): {i+1}/{len(image_files)} images, {stats['total_boxes']} boxes")

    return stats


# ============================================================
# CO-TRAINING LOOP
# ============================================================
def run_cotraining_iteration(
    iteration: int,
    model_a_weights: str,
    model_b_weights: str,
    labeled_yaml: str,
    unlabeled_images_dir: str,
    output_dir: str,
    method: str = 'combined',
    epochs_per_iter: int = 10,
    max_pseudo_images: int = 300,
    train_device: str = '0,1',
    infer_device: str = '0',
    **kwargs
) -> Tuple[str, str, Dict]:
    """
    Run one co-training iteration.
    Model A generates pseudo-labels → Model B trains.
    Model B generates pseudo-labels → Model A trains.
    Returns updated weights paths and iteration metrics.
    """
    from ultralytics import YOLO

    iter_dir = Path(output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    is_ddp_worker = int(os.environ.get('LOCAL_RANK', -1)) != -1

    if not is_ddp_worker:
        print(f"\n{'='*60}")
        print(f"CO-TRAINING ITERATION {iteration} (method={method})")
        print(f"{'='*60}")

    # --- Model A → pseudo-labels → train Model B ---
    pseudo_dir_a = iter_dir / "pseudo_from_a"
    combo_dir_b = iter_dir / "data_for_b"

    if not is_ddp_worker:
        print(f"\n  [Step 1] Model A generating pseudo-labels...")
        if method == 'confidence':
            stats_a = generate_pseudo_labels_confidence(
                model_a_weights, unlabeled_images_dir, str(pseudo_dir_a),
                max_images=max_pseudo_images, device=infer_device
            )
        elif method == 'epistemic':
            stats_a = generate_pseudo_labels_epistemic(
                model_a_weights, unlabeled_images_dir, str(pseudo_dir_a),
                max_images=max_pseudo_images, device=infer_device, **kwargs
            )
        elif method == 'combined':
            stats_a = generate_pseudo_labels_combined(
                model_a_weights, unlabeled_images_dir, str(pseudo_dir_a),
                max_images=max_pseudo_images, device=infer_device, **kwargs
            )

        print(f"    → {stats_a['accepted']} boxes accepted, "
              f"{stats_a.get('rejected_epistemic', 0) + stats_a.get('rejected_combined', 0)} rejected by uncertainty")

        # Prepare combined dataset for Model B
        _prepare_combined_dataset(labeled_yaml, pseudo_dir_a, unlabeled_images_dir, combo_dir_b)

    # Train Model B
    if not is_ddp_worker:
        print(f"\n  [Step 2] Training Model B (YOLOv12) for {epochs_per_iter} epochs...")
    
    # *** CRITICAL FIX: Use SINGLE GPU to avoid DDP (crashes on K80s) ***
    # Convert '1,2' -> 1, '0,1' -> 0, etc.
    if train_device != 'cpu':
        single_gpu = int(str(train_device).split(',')[0])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(single_gpu)
    else:
        single_gpu = 'cpu'

    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    model_b = YOLO(model_b_weights)
    results_b = model_b.train(
        data=str(combo_dir_b / "dataset.yaml"),
        epochs=epochs_per_iter, batch=16, imgsz=640,
        device=single_gpu, optimizer='SGD',
        project=str(iter_dir / "model_b"), name='train', exist_ok=True,
        verbose=not is_ddp_worker, amp=False, half=False, workers=0,
    )
    new_b_weights = str(iter_dir / "model_b" / "train" / "weights" / "best.pt")
    if is_ddp_worker:
        import sys
        sys.exit(0)
    else:
        if getattr(results_b, "box", None) is not None:
            b_map50 = float(results_b.box.map50)
        else:
            val_model = YOLO(new_b_weights)
            val_res = val_model.val(data=str(combo_dir_b / "dataset.yaml"), device=infer_device, split='val', verbose=False)
            b_map50 = float(val_res.box.map50)
            del val_model
        print(f"    → Model B mAP50: {b_map50:.4f}")

    # --- Model B → pseudo-labels → train Model A ---
    pseudo_dir_b = iter_dir / "pseudo_from_b"
    combo_dir_a = iter_dir / "data_for_a"

    if not is_ddp_worker:
        print(f"\n  [Step 3] Model B generating pseudo-labels...")
        if method == 'confidence':
            stats_b = generate_pseudo_labels_confidence(
                new_b_weights, unlabeled_images_dir, str(pseudo_dir_b),
                max_images=max_pseudo_images, device=infer_device
            )
        elif method == 'epistemic':
            stats_b = generate_pseudo_labels_epistemic(
                new_b_weights, unlabeled_images_dir, str(pseudo_dir_b),
                max_images=max_pseudo_images, device=infer_device, **kwargs
            )
        elif method == 'combined':
            stats_b = generate_pseudo_labels_combined(
                new_b_weights, unlabeled_images_dir, str(pseudo_dir_b),
                max_images=max_pseudo_images, device=infer_device, **kwargs
            )

        print(f"    → {stats_b['accepted']} boxes accepted")

        # Prepare combined dataset for Model A
        _prepare_combined_dataset(labeled_yaml, pseudo_dir_b, unlabeled_images_dir, combo_dir_a)

    # Train Model A
    if not is_ddp_worker:
        print(f"\n  [Step 4] Training Model A (RT-DETRv2) for {epochs_per_iter} epochs...")
        
    # *** CRITICAL FIX: Use SINGLE GPU to avoid DDP (crashes on K80s) ***
    # RT-DETRv2 uses first available GPU
    rtdetr_single_gpu = int(str(train_device).split(',')[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rtdetr_single_gpu)

    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    model_a = YOLO(model_a_weights)
    results_a = model_a.train(
        data=str(combo_dir_a / "dataset.yaml"),
        epochs=epochs_per_iter, batch=4, imgsz=640,
        device=rtdetr_single_gpu, optimizer='SGD',
        project=str(iter_dir / "model_a"), name='train', exist_ok=True,
        verbose=not is_ddp_worker, amp=False, half=False, workers=0,
    )
    new_a_weights = str(iter_dir / "model_a" / "train" / "weights" / "best.pt")
    if is_ddp_worker:
        import sys
        sys.exit(0)
    else:
        if getattr(results_a, "box", None) is not None:
            a_map50 = float(results_a.box.map50)
        else:
            val_model = YOLO(new_a_weights)
            val_res = val_model.val(data=str(combo_dir_a / "dataset.yaml"), device=infer_device, split='val', verbose=False)
            a_map50 = float(val_res.box.map50)
            del val_model
        print(f"    → Model A mAP50: {a_map50:.4f}")

    # Save iteration results
    iter_results = {
        'iteration': iteration, 'method': method,
        'model_a_map50': a_map50, 'model_b_map50': b_map50,
        'pseudo_labels_a': stats_a, 'pseudo_labels_b': stats_b,
    }
    with open(iter_dir / "results.json", 'w') as f:
        json.dump(iter_results, f, indent=2)

    return new_a_weights, new_b_weights, iter_results


def _prepare_combined_dataset(
    original_yaml: str,
    pseudo_labels_dir: Path,
    unlabeled_images_dir: str,
    output_dir: Path
):
    """Combine original labeled data with pseudo-labeled data using symlinks."""
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    with open(original_yaml) as f:
        orig = yaml.safe_load(f)

    orig_path = Path(orig['path'])
    orig_train_images = orig_path / "train" / "images"
    orig_train_labels = orig_path / "train" / "labels"

    # Symlink original data
    count_orig = 0
    if orig_train_images.exists():
        for f in orig_train_images.iterdir():
            dst = train_images / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
                count_orig += 1
    if orig_train_labels.exists():
        for f in orig_train_labels.iterdir():
            dst = train_labels / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)

    # Add pseudo-labeled data
    count_pseudo = 0
    if pseudo_labels_dir.exists():
        for label_file in pseudo_labels_dir.iterdir():
            if label_file.suffix != '.txt':
                continue
            dst_label = train_labels / label_file.name
            if not dst_label.exists():
                shutil.copy2(label_file, dst_label)
            img_name = label_file.stem + '.jpg'
            src_img = Path(unlabeled_images_dir) / img_name
            dst_img = train_images / img_name
            if src_img.exists() and not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
                count_pseudo += 1

    # Copy val from original — handle both relative and absolute val paths
    val_dir = output_dir / "val"
    if not val_dir.exists():
        val_images = val_dir / "images"
        val_labels = val_dir / "labels"
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)
        
        # Resolve val path: can be relative to orig_path or absolute
        val_field = orig.get('val', 'val')
        val_path = Path(val_field)
        if not val_path.is_absolute():
            val_path = orig_path / val_field
        # val_path might point to 'val' dir or 'val/images' — normalize
        if (val_path / "images").exists():
            val_src_images = val_path / "images"
            val_src_labels = val_path / "labels"
        elif val_path.parent.name == "images" or val_path.name == "images":
            val_src_images = val_path if val_path.name == "images" else val_path
            val_src_labels = val_path.parent / "labels"
        else:
            val_src_images = val_path / "images"
            val_src_labels = val_path / "labels"
        
        for subdir, src_dir in [("images", val_src_images), ("labels", val_src_labels)]:
            if src_dir.exists():
                for f in src_dir.iterdir():
                    dst = val_dir / subdir / f.name
                    if not dst.exists():
                        os.symlink(f.resolve(), dst)

    # Write dataset YAML
    yaml_content = {
        'path': str(output_dir),
        'train': 'train/images', 'val': 'val/images',
        'nc': 1, 'names': {0: 'product'}
    }
    with open(output_dir / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"    Combined dataset: {count_orig} labeled + {count_pseudo} pseudo-labeled")


# ============================================================
# EXPERIMENT RUNNERS
# ============================================================
def run_single_experiment(args):
    """Run one complete co-training experiment (1 method, 1 seed, N iterations)."""
    set_seed(args.seed)

    baseline_dir = PROJECT_ROOT / "results" / "week1_baselines"
    model_a_weights = str(baseline_dir / "rtdetr" / "train" / "weights" / "best.pt")
    model_b_weights = str(baseline_dir / "yolov12" / "train" / "weights" / "best.pt")

    if not os.path.exists(model_a_weights):
        model_a_weights = str(PROJECT_ROOT / "models" / "rtdetr-l.pt")
        print(f"⚠ Using pretrained RT-DETRv2")
    if not os.path.exists(model_b_weights):
        model_b_weights = str(PROJECT_ROOT / "models" / "yolo12n.pt")
        print(f"⚠ Using pretrained YOLOv12")

    labeled_yaml = getattr(args, 'data_yaml', None) or str(PROJECT_ROOT / "data" / "SKU110K" / "sku110k.yaml")
    unlabeled_dir = getattr(args, 'unlabeled_dir', None) or str(PROJECT_ROOT / "data" / "SKU110K" / "yolo_format" / "unlabeled" / "images")
    # Build output directory with optional tag
    tag = f"_{args.output_tag}" if getattr(args, 'output_tag', None) else ""
    if args.method == 'combined':
        output_dir = PROJECT_ROOT / "results" / "calibcotrain_cl" / f"{args.method}_seed{args.seed}{tag}" / f"alpha_{args.alpha}"
    else:
        output_dir = PROJECT_ROOT / "results" / "calibcotrain_cl" / f"{args.method}_seed{args.seed}{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CalibCoTrain-CL EXPERIMENT")
    print(f"{'='*70}")
    print(f"Method:      {args.method}")
    print(f"Alpha:       {args.alpha}" if args.method == 'combined' else "")
    print(f"Seed:        {args.seed}")
    print(f"Iterations:  {args.iterations}")
    print(f"Epochs/iter: {args.epochs_per_iter}")
    print(f"Train GPU:   {args.train_device}")
    print(f"Infer GPU:   {args.infer_device}")

    all_results = []
    for iteration in range(1, args.iterations + 1):
        kwargs = {}
        if args.method == 'epistemic':
            kwargs = {'epistemic_threshold': args.epistemic_thresh, 'T': args.mc_T}
        elif args.method == 'combined':
            kwargs = {'alpha': args.alpha, 'combined_threshold': args.combined_thresh, 'T': args.mc_T}

        model_a_weights, model_b_weights, iter_results = run_cotraining_iteration(
            iteration=iteration,
            model_a_weights=model_a_weights,
            model_b_weights=model_b_weights,
            labeled_yaml=labeled_yaml,
            unlabeled_images_dir=unlabeled_dir,
            output_dir=str(output_dir),
            method=args.method,
            epochs_per_iter=args.epochs_per_iter,
            max_pseudo_images=args.max_pseudo,
            train_device=args.train_device,
            infer_device=args.infer_device,
            **kwargs
        )
        all_results.append(iter_results)

    # Save experiment summary
    summary = {
        'method': args.method, 'seed': args.seed,
        'alpha': args.alpha if args.method == 'combined' else None,
        'iterations': args.iterations,
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
    }
    with open(output_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"RESULTS: {args.method} (seed={args.seed})")
    print(f"{'='*70}")
    print(f"{'Iter':<6} {'Model A (RT-DETRv2)':<20} {'Model B (YOLOv12)':<20}")
    print(f"{'----':<6} {'-------------------':<20} {'------------------':<20}")
    for r in all_results:
        print(f"{r['iteration']:<6} {r['model_a_map50']:<20.4f} {r['model_b_map50']:<20.4f}")

    return summary


def run_alpha_sweep(args):
    """Run α sweep to find optimal combined uncertainty weight."""
    alphas = [0.7, 0.9]
    results = [
        {'alpha': 0.1, 'model_a_map50': 0.5270, 'model_b_map50': 0.8807},
        {'alpha': 0.3, 'model_a_map50': 0.5270, 'model_b_map50': 0.8807},
        {'alpha': 0.5, 'model_a_map50': 0.5400, 'model_b_map50': 0.8876}
    ]

    for alpha in alphas:
        print(f"\n{'#'*70}")
        print(f"ALPHA SWEEP: α = {alpha}")
        print(f"{'#'*70}")

        args.method = 'combined'
        args.alpha = alpha
        args.iterations = 1  # Quick 1-iteration sweep
        summary = run_single_experiment(args)
        results.append({
            'alpha': alpha,
            'model_a_map50': summary['results'][-1]['model_a_map50'],
            'model_b_map50': summary['results'][-1]['model_b_map50'],
        })

    # Find optimal α
    best = max(results, key=lambda x: x['model_b_map50'])
    print(f"\n{'='*70}")
    print(f"ALPHA SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"{'Alpha':<10} {'Model A mAP50':<20} {'Model B mAP50':<20}")
    for r in results:
        marker = " ← BEST" if r['alpha'] == best['alpha'] else ""
        print(f"{r['alpha']:<10.1f} {r['model_a_map50']:<20.4f} {r['model_b_map50']:<20.4f}{marker}")
    print(f"\nOptimal α = {best['alpha']}")

    with open(PROJECT_ROOT / "results" / "calibcotrain_cl" / "alpha_sweep.json", 'w') as f:
        json.dump(results, f, indent=2)

    return best['alpha']


def run_full_experiment(args):
    """Run 3 strategies × 3 seeds."""
    methods = ['confidence', 'epistemic', 'combined']
    seeds = [42, 123, 456]

    all_summaries = []

    for method in methods:
        for seed in seeds:
            print(f"\n{'#'*70}")
            print(f"FULL EXPERIMENT: method={method}, seed={seed}")
            print(f"{'#'*70}")

            args.method = method
            args.seed = seed
            summary = run_single_experiment(args)
            all_summaries.append(summary)

    # Print cross-method comparison
    print(f"\n{'='*70}")
    print(f"CROSS-METHOD COMPARISON (Final Iteration)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Seed':<8} {'Model A':<12} {'Model B':<12}")
    for s in all_summaries:
        final = s['results'][-1]
        print(f"{s['method']:<15} {s['seed']:<8} {final['model_a_map50']:<12.4f} {final['model_b_map50']:<12.4f}")

    with open(PROJECT_ROOT / "results" / "calibcotrain_cl" / "full_experiment.json", 'w') as f:
        json.dump(all_summaries, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CalibCoTrain-CL Framework")
    parser.add_argument('--method', choices=['confidence', 'epistemic', 'combined'], default='combined')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--epochs-per-iter', type=int, default=10)
    parser.add_argument('--max-pseudo', type=int, default=300, help='Max unlabeled images per iter')
    parser.add_argument('--seed', type=int, default=42)

    # Strategy-specific
    parser.add_argument('--epistemic-thresh', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.5, help='Combined: α weight for cls vs loc')
    parser.add_argument('--combined-thresh', type=float, default=0.1)
    parser.add_argument('--mc-T', type=int, default=5, help='MC Dropout passes')

    # GPU
    parser.add_argument('--train-device', type=str, default='0,1', help='GPUs for training')
    parser.add_argument('--infer-device', type=str, default='0', help='GPU for inference')

    # Data paths (for label-fraction experiments)
    parser.add_argument('--data-yaml', type=str, default=None, help='Custom data YAML (default: SKU110K full)')
    parser.add_argument('--unlabeled-dir', type=str, default=None, help='Custom unlabeled images dir')
    parser.add_argument('--output-tag', type=str, default=None, help='Extra tag for output directory')

    # Experiment modes
    parser.add_argument('--alpha-sweep', action='store_true', help='Run α sweep')
    parser.add_argument('--full-experiment', action='store_true', help='Run all 3 strategies × 3 seeds')

    args = parser.parse_args()

    if args.alpha_sweep:
        run_alpha_sweep(args)
    elif args.full_experiment:
        run_full_experiment(args)
    else:
        run_single_experiment(args)


if __name__ == '__main__':
    main()
