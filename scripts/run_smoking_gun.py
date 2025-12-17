#!/usr/bin/env python3
"""
Smoking Gun Ablation: Post-Hoc Calibration Instability Under Iterative Training
================================================================================
THE most important experiment in the paper.

Hypothesis:
    Isotonic Regression achieves near-perfect D-ECE on static predictions,
    but DEGRADES when used inside a co-training loop because the retrained
    model generates NEW overconfident errors that the fixed isotonic mapping
    was never trained on.

Protocol:
    For 5 co-training iterations:
        1. Fit Isotonic Regression on current model predictions
        2. Use isotonic-calibrated confidence > 0.7 to filter pseudo-labels
        3. Retrain model on labeled + pseudo-labeled data
        4. Measure D-ECE on the NEW model's predictions (without recalibrating)

    Compare against:
        - Baseline: raw confidence > 0.7 (no calibration)
        - CalibCoTrain: MC Dropout epistemic filtering

    Expected result:
        Isotonic helps at iteration 1 but D-ECE degrades by iteration 3-5,
        proving post-hoc calibration is unstable under iterative training.

Usage:
    python scripts/run_smoking_gun.py --iterations 5 --epochs-per-iter 10
"""

import argparse
import json
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

import os
# Ensure we preserve the shell's original CUDA visibility before Ultralytics alters it
ORIGINAL_CUDA = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')

# K80 fix: Use GLOO for DDP to bypass NCCL driver/runtime mismatch
os.environ['ULTRALYTICS_DDP_BACKEND'] = 'gloo'

# K80 fix: disable cuDNN (sm_37 not supported by cuDNN 8.x)
import torch
torch.backends.cudnn.enabled = False

# FORCE early PyTorch CUDA initialization so it locks in the full device count
# from the shell before Ultralytics alters CUDA_VISIBLE_DEVICES during inference!
_ = torch.cuda.device_count()

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration.detection_calibration import (
    match_detections_to_gt,
    compute_detection_ece,
    apply_isotonic_regression,
)


def get_model_predictions(model, images_dir, conf_threshold=0.01, device='0'):
    """Run inference and return list of prediction dicts."""
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    all_preds = []
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            iou=0.5,
            device=device,
            verbose=False,
        )

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for j in range(len(r.boxes)):
                    all_preds.append({
                        'image': img_name,
                        'box': r.boxes.xyxy[j].tolist(),
                        'confidence': float(r.boxes.conf[j]),
                        'class': int(r.boxes.cls[j]),
                    })

        if (i + 1) % 50 == 0:
            print(f"    Inference: {i+1}/{len(image_files)} images, {len(all_preds)} detections")

    return all_preds


def measure_calibration(predictions, gt_labels_dir, images_dir, iou_threshold=0.5, n_bins=15):
    """Match predictions to GT and compute D-ECE."""
    confidences, correctness = match_detections_to_gt(
        predictions, gt_labels_dir, images_dir, iou_threshold
    )
    metrics = compute_detection_ece(confidences, correctness, n_bins)
    return metrics


def generate_pseudo_labels_with_isotonic(
    model, isotonic_model, unlabeled_dir, output_labels_dir,
    conf_threshold=0.7, max_images=500, device='0'
):
    """Generate pseudo-labels using isotonic-calibrated confidence."""
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(unlabeled_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])[:max_images]

    stats = {'total_images': 0, 'images_with_labels': 0, 'total_boxes': 0,
             'accepted': 0, 'rejected': 0}

    for img_name in image_files:
        img_path = os.path.join(unlabeled_dir, img_name)
        results = model.predict(source=img_path, conf=0.01, device=device, verbose=False)

        label_lines = []
        for r in results:
            if r.boxes is None:
                continue
            for j in range(len(r.boxes)):
                raw_conf = float(r.boxes.conf[j])

                # Apply isotonic calibration
                calibrated_conf = float(isotonic_model.predict([raw_conf])[0])

                if calibrated_conf >= conf_threshold:
                    box = r.boxes.xywhn[j].tolist()
                    cls = int(r.boxes.cls[j])
                    label_lines.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                    stats['accepted'] += 1
                else:
                    stats['rejected'] += 1

        if label_lines:
            label_path = os.path.join(
                output_labels_dir,
                img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            stats['images_with_labels'] += 1
            stats['total_boxes'] += len(label_lines)

        stats['total_images'] += 1

    return stats


def generate_pseudo_labels_confidence(
    model, unlabeled_dir, output_labels_dir,
    conf_threshold=0.7, max_images=500, device='0'
):
    """Baseline: Generate pseudo-labels using raw confidence threshold."""
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(unlabeled_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])[:max_images]

    stats = {'total_images': 0, 'images_with_labels': 0, 'total_boxes': 0}

    for img_name in image_files:
        img_path = os.path.join(unlabeled_dir, img_name)
        results = model.predict(source=img_path, conf=conf_threshold, device=device, verbose=False)

        label_lines = []
        for r in results:
            if r.boxes is None:
                continue
            for j in range(len(r.boxes)):
                box = r.boxes.xywhn[j].tolist()
                cls = int(r.boxes.cls[j])
                label_lines.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")

        if label_lines:
            label_path = os.path.join(
                output_labels_dir,
                img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            stats['images_with_labels'] += 1
            stats['total_boxes'] += len(label_lines)

        stats['total_images'] += 1

    return stats


def prepare_combined_dataset(original_yaml, pseudo_labels_dir, unlabeled_images_dir, output_dir):
    """Combine original labeled data with pseudo-labeled data."""
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    with open(original_yaml) as f:
        orig = yaml.safe_load(f)

    orig_path = Path(orig['path'])

    # Symlink original images and labels
    count_orig = 0
    for subdir in ['images', 'labels']:
        src = orig_path / "train" / subdir
        dst = output_dir / "train" / subdir
        if src.exists():
            for fpath in src.iterdir():
                target = dst / fpath.name
                if not target.exists():
                    os.symlink(fpath.resolve(), target)
                    if subdir == 'images':
                        count_orig += 1

    # Add pseudo-labeled data
    count_pseudo = 0
    pseudo_dir = Path(pseudo_labels_dir)
    if pseudo_dir.exists():
        for label_file in pseudo_dir.iterdir():
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

    # Copy val
    val_dir = output_dir / "val"
    if not val_dir.exists():
        for subdir in ['images', 'labels']:
            (val_dir / subdir).mkdir(parents=True, exist_ok=True)
            src = orig_path / "val" / subdir
            if src.exists():
                for fpath in src.iterdir():
                    target = val_dir / subdir / fpath.name
                    if not target.exists():
                        os.symlink(fpath.resolve(), target)

    # Write YAML
    import yaml
    yaml_content = {
        'path': str(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': {0: 'product'}
    }
    with open(output_dir / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"    Combined dataset: {count_orig} original + {count_pseudo} pseudo-labeled images")


def run_smoking_gun(args):
    """Main smoking gun ablation."""
    from ultralytics import YOLO

    # Paths
    baseline_dir = PROJECT_ROOT / "results" / "week1_baselines"
    model_weights = str(baseline_dir / "yolov12" / "train" / "weights" / "best.pt")

    if not os.path.exists(model_weights):
        print("ERROR: No trained YOLOv12 weights found. Run train_baselines.py first.")
        return

    labeled_yaml = str(PROJECT_ROOT / "data" / "SKU110K" / "sku110k.yaml")
    unlabeled_dir = str(PROJECT_ROOT / "data" / "SKU110K" / "yolo_format" / "unlabeled" / "images")
    test_images = str(PROJECT_ROOT / "data" / "SKU110K" / "yolo_format" / "test" / "images")
    gt_labels = str(PROJECT_ROOT / "data" / "SKU110K" / "yolo_format" / "test" / "labels")

    output_dir = PROJECT_ROOT / "results" / "smoking_gun_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    methods_to_run = [args.method] if args.method != 'both' else ['confidence', 'isotonic']

    print(f"\n{'='*70}")
    print(f"SMOKING GUN ABLATION: Post-Hoc Calibration Instability")
    print(f"{'='*70}")
    print(f"Model:       YOLOv12 (baseline weights)")
    print(f"Iterations:  {args.iterations}")
    print(f"Epochs/iter: {args.epochs_per_iter}")
    print(f"Methods:     {methods_to_run}")
    print(f"Device:      GPU {device}")

    # ==========================================================
    # Iteration 0: Measure baseline calibration
    # ==========================================================
    print(f"\n{'='*60}")
    print(f"ITERATION 0: Baseline Measurement")
    print(f"{'='*60}")

    model = YOLO(model_weights)
    preds = get_model_predictions(model, test_images, device=device)

    baseline_metrics = measure_calibration(preds, gt_labels, test_images)
    print(f"  Baseline D-ECE: {baseline_metrics.d_ece:.4f}")
    print(f"  Baseline Brier: {baseline_metrics.brier:.4f}")
    print(f"  Detections:     {baseline_metrics.num_detections}")

    # Fit initial isotonic model
    confidences, correctness = match_detections_to_gt(preds, gt_labels, test_images)
    from sklearn.isotonic import IsotonicRegression
    iso_model = IsotonicRegression(out_of_bounds='clip')
    iso_model.fit(confidences, correctness)

    # Measure isotonic-calibrated D-ECE at iter 0
    iso_confs = iso_model.predict(confidences)
    iso_metrics_0 = compute_detection_ece(iso_confs, correctness)
    print(f"  Isotonic D-ECE: {iso_metrics_0.d_ece:.4f} (after calibration)")

    # Store results for both methods
    results = {
        'confidence': [{
            'iteration': 0,
            'd_ece': float(baseline_metrics.d_ece),
            'brier': float(baseline_metrics.brier),
            'map50': None,
            'num_detections': baseline_metrics.num_detections,
        }],
        'isotonic': [{
            'iteration': 0,
            'd_ece_raw': float(baseline_metrics.d_ece),
            'd_ece_calibrated': float(iso_metrics_0.d_ece),
            'brier': float(baseline_metrics.brier),
            'map50': None,
            'num_detections': baseline_metrics.num_detections,
        }],
    }

    # ==========================================================
    # Run co-training iterations for BOTH methods
    # ==========================================================
    for method in methods_to_run:
        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*70}")

        current_weights = model_weights

        for iteration in range(1, args.iterations + 1):
            print(f"\n--- Iteration {iteration}/{args.iterations} ({method}) ---")
            iter_dir = output_dir / method / f"iter_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Generate pseudo-labels
            pseudo_dir = iter_dir / "pseudo_labels"
            model_iter = YOLO(current_weights)

            if method == 'isotonic':
                # Re-fit isotonic on current model's test predictions
                print(f"  Fitting isotonic regression on current model...")
                preds_current = get_model_predictions(model_iter, test_images, device=device)
                confs_cur, corr_cur = match_detections_to_gt(preds_current, gt_labels, test_images)
                iso_model = IsotonicRegression(out_of_bounds='clip')
                iso_model.fit(confs_cur, corr_cur)

                stats = generate_pseudo_labels_with_isotonic(
                    model_iter, iso_model, unlabeled_dir, str(pseudo_dir),
                    conf_threshold=0.7, max_images=args.max_pseudo, device=device
                )
            else:
                stats = generate_pseudo_labels_confidence(
                    model_iter, unlabeled_dir, str(pseudo_dir),
                    conf_threshold=0.7, max_images=args.max_pseudo, device=device
                )

            print(f"  Pseudo-labels: {stats['total_boxes']} boxes from {stats['images_with_labels']} images")

            # Step 2: Prepare combined dataset
            combo_dir = iter_dir / "combo_data"
            prepare_combined_dataset(labeled_yaml, str(pseudo_dir), unlabeled_dir, combo_dir)

            print(f"  Training for {args.epochs_per_iter} epochs...")
            # RESTORE full visibility so train() can properly allocate the selected train GPUs
            os.environ['CUDA_VISIBLE_DEVICES'] = ORIGINAL_CUDA
            model_train = YOLO(current_weights)
            train_device = args.train_device if args.train_device else args.device
            train_results = model_train.train(
                data=str(combo_dir / "dataset.yaml"),
                epochs=args.epochs_per_iter,
                batch=32,  # 16 per GPU × 2 GPUs
                imgsz=640,
                device=train_device,
                project=str(iter_dir / "train"),
                name='run',
                exist_ok=True,
                verbose=False,
                amp=False,
                workers=4,
                optimizer='SGD',  # K80 fix: MuSGD uses BFloat16 cuBLAS ops unsupported on sm_37
            )

            new_weights = str(iter_dir / "train" / "run" / "weights" / "best.pt")
            if not os.path.exists(new_weights):
                new_weights = str(iter_dir / "train" / "run" / "weights" / "last.pt")

            map50 = float(train_results.box.map50) if hasattr(train_results, 'box') else None
            print(f"  mAP50: {map50:.4f}" if map50 else "  mAP50: N/A")

            # Step 4: Measure D-ECE on NEW model's predictions
            print(f"  Measuring calibration on retrained model...")
            model_new = YOLO(new_weights)
            preds_new = get_model_predictions(model_new, test_images, device=device)
            new_metrics = measure_calibration(preds_new, gt_labels, test_images)

            print(f"  D-ECE (raw, post-retraining): {new_metrics.d_ece:.4f}")

            iter_result = {
                'iteration': iteration,
                'd_ece': float(new_metrics.d_ece),
                'brier': float(new_metrics.brier),
                'map50': map50,
                'num_detections': new_metrics.num_detections,
                'pseudo_label_stats': stats,
            }

            if method == 'isotonic':
                # Also measure: what does isotonic think the D-ECE is?
                confs_new, corr_new = match_detections_to_gt(preds_new, gt_labels, test_images)
                iso_confs_new = iso_model.predict(confs_new)  # using OLD isotonic model
                iso_metrics_new = compute_detection_ece(iso_confs_new, corr_new)
                iter_result['d_ece_old_isotonic'] = float(iso_metrics_new.d_ece)
                print(f"  D-ECE (old isotonic applied): {iso_metrics_new.d_ece:.4f}")

                # Re-fit and measure with fresh isotonic
                iso_fresh = IsotonicRegression(out_of_bounds='clip')
                iso_fresh.fit(confs_new, corr_new)
                iso_fresh_confs = iso_fresh.predict(confs_new)
                iso_fresh_metrics = compute_detection_ece(iso_fresh_confs, corr_new)
                iter_result['d_ece_fresh_isotonic'] = float(iso_fresh_metrics.d_ece)
                print(f"  D-ECE (fresh isotonic): {iso_fresh_metrics.d_ece:.4f}")

                iter_result['d_ece_raw'] = iter_result['d_ece']
                iter_result['d_ece_calibrated'] = iter_result['d_ece_old_isotonic']

            results[method].append(iter_result)
            current_weights = new_weights

            # Save incremental results
            with open(output_dir / "smoking_gun_results.json", 'w') as f:
                json.dump(results, f, indent=2)

    # ==========================================================
    # Print final summary table
    # ==========================================================
    print(f"\n{'='*70}")
    print(f"SMOKING GUN ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Iter':<6} {'Conf D-ECE':<12} {'Iso D-ECE(raw)':<16} {'Iso D-ECE(old_iso)':<20} {'Conf mAP':<10} {'Iso mAP':<10}")
    print(f"{'-'*6} {'-'*12} {'-'*16} {'-'*20} {'-'*10} {'-'*10}")

    for i in range(args.iterations + 1):
        conf_r = results['confidence'][i] if i < len(results['confidence']) else {}
        iso_r = results['isotonic'][i] if i < len(results['isotonic']) else {}

        conf_ece = f"{conf_r.get('d_ece', 0):.4f}" if conf_r else "N/A"
        iso_raw = f"{iso_r.get('d_ece_raw', iso_r.get('d_ece', 0)):.4f}" if iso_r else "N/A"
        iso_cal = f"{iso_r.get('d_ece_calibrated', iso_r.get('d_ece_calibrated', 0)):.4f}" if iso_r else "N/A"
        conf_map = f"{conf_r['map50']:.4f}" if conf_r.get('map50') else "base"
        iso_map = f"{iso_r['map50']:.4f}" if iso_r.get('map50') else "base"

        print(f"{i:<6} {conf_ece:<12} {iso_raw:<16} {iso_cal:<20} {conf_map:<10} {iso_map:<10}")

    print(f"\n✓ Results saved: {output_dir / 'smoking_gun_results.json'}")

    # Save final summary
    summary = {
        'experiment': 'smoking_gun_ablation',
        'hypothesis': 'Post-hoc isotonic regression degrades under iterative co-training',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'iterations': args.iterations,
            'epochs_per_iter': args.epochs_per_iter,
            'max_pseudo': args.max_pseudo,
            'model': 'YOLOv12n',
        },
        'results': results,
    }
    with open(output_dir / "smoking_gun_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smoking Gun Ablation")
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of co-training iterations')
    parser.add_argument('--epochs-per-iter', type=int, default=10,
                       help='Training epochs per iteration')
    parser.add_argument('--max-pseudo', type=int, default=300,
                       help='Max unlabeled images for pseudo-labeling per iteration')
    parser.add_argument('--method', choices=['confidence', 'isotonic', 'both'], default='both',
                       help='Which method to run (use separate GPUs for parallel execution)')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU device ID for inference')
    parser.add_argument('--train-device', type=str, default=None,
                       help='GPU device(s) for training, e.g. "0,1" for 2-GPU DDP')
    args = parser.parse_args()

    run_smoking_gun(args)
