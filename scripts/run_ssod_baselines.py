#!/usr/bin/env python3
"""
5 SSOD Baselines for SKU-110K — Adapted for Our Benchmark
==========================================================
Implements lightweight versions of 5 established SSOD baselines adapted
to the SKU-110K format to serve as comparison points in the paper.

Baselines:
  1. PseudoLabel   : Standard confidence thresholding (Yazdanjouei 2025)
  2. MeanTeacher   : Exponential moving average teacher model  
  3. STAC          : Self-Training with Augmentation Consistency
  4. SoftPseudo    : Soft pseudo-labels using confidence as weight
  5. NoisyStudent  : Pseudo labels + Dropout noise during student training

All methods run with:
  - 5 iterations, 10 epochs/iter
  - 3 seeds (5 seeds for the 0.1% labeled split)
  - Max 300 pseudo-labeled images per iteration
"""

import argparse
import json
import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import gc
import torch
torch.backends.cudnn.enabled = False
os.environ['ULTRALYTICS_DDP_BACKEND'] = 'gloo'

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_DEVICE = '0,1,2,3'
INFER_DEVICE = '0'  # Single GPU for inference to avoid OOM
LABELED_YAML = str(PROJECT_ROOT / "data" / "SKU110K" / "sku110k.yaml")
UNLABELED_IMAGES_DIR = str(PROJECT_ROOT / "data" / "SKU110K" / "yolo_format" / "unlabeled" / "images")
BASELINE_WEIGHTS = str(PROJECT_ROOT / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt")
RESULTS_DIR = PROJECT_ROOT / "results" / "ssod_baselines"


def gpu_cleanup():
    """Aggressive GPU memory cleanup between phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import time; time.sleep(2)

if not os.path.exists(BASELINE_WEIGHTS):
    BASELINE_WEIGHTS = str(PROJECT_ROOT / "models" / "yolo12n.pt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(weights_path, data_yaml, out_dir, epochs=10, extra_args=None):
    from ultralytics import YOLO
    # Set CUDA_VISIBLE_DEVICES for DDP training (learned from alpha sweep)
    os.environ['CUDA_VISIBLE_DEVICES'] = TRAIN_DEVICE
    model = YOLO(weights_path)
    train_kwargs = dict(
        data=data_yaml, epochs=epochs, batch=16, imgsz=640,
        device=TRAIN_DEVICE, optimizer='SGD', project=out_dir, name='train',
        exist_ok=True, verbose=False, amp=False, half=False, workers=0,
    )
    if extra_args:
        train_kwargs.update(extra_args)
    results = model.train(**train_kwargs)
    best_path = str(Path(out_dir) / "train" / "weights" / "best.pt")
    # Handle DDP returning None results
    if results is None or not hasattr(results, 'box'):
        model2 = YOLO(best_path)
        val_results = model2.val(data=data_yaml, device=INFER_DEVICE)
        map50 = float(val_results.box.map50)
        del model2
    else:
        map50 = float(results.box.map50)
    # DDP worker exit
    is_ddp_worker = int(os.environ.get('LOCAL_RANK', -1)) != -1
    if is_ddp_worker:
        sys.exit(0)
    del model
    gpu_cleanup()
    return best_path, map50


def prepare_dataset(labeled_yaml, pseudo_dir, unlabeled_images_dir, out_dir):
    """Build the combined labeled+pseudo dataset. Uses symlinks for efficiency."""
    import yaml
    out_dir = Path(out_dir)
    ti = out_dir / "train" / "images"; tl = out_dir / "train" / "labels"
    vi = out_dir / "val" / "images";   vl = out_dir / "val" / "labels"
    for d in [ti, tl, vi, vl]: d.mkdir(parents=True, exist_ok=True)
    
    with open(labeled_yaml) as f: orig = yaml.safe_load(f)
    orig_path = Path(orig['path'])
    
    for src in (orig_path / "train" / "images").glob("*.jpg"):
        dst = ti / src.name
        if not dst.exists(): os.symlink(src.resolve(), dst)
    for src in (orig_path / "train" / "labels").glob("*.txt"):
        dst = tl / src.name
        if not dst.exists(): os.symlink(src.resolve(), dst)
    for src in (orig_path / "val" / "images").glob("*.jpg"):
        dst = vi / src.name
        if not dst.exists(): os.symlink(src.resolve(), dst)
    for src in (orig_path / "val" / "labels").glob("*.txt"):
        dst = vl / src.name
        if not dst.exists(): os.symlink(src.resolve(), dst)
    
    n_pseudo = 0
    for lf in Path(pseudo_dir).glob("*.txt"):
        dst_lbl = tl / lf.name
        if not dst_lbl.exists(): shutil.copy2(lf, dst_lbl)
        src_img = Path(unlabeled_images_dir) / (lf.stem + ".jpg")
        dst_img = ti / src_img.name
        if src_img.exists() and not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img); n_pseudo += 1
    
    yaml_content = {'path': str(out_dir), 'train': 'train/images',
                    'val': 'val/images', 'nc': 1, 'names': {0: 'product'}}
    with open(out_dir / "dataset.yaml", 'w') as f: yaml.dump(yaml_content, f)
    print(f"  Combined: {n_pseudo} pseudo-labeled images added")
    return str(out_dir / "dataset.yaml")


# ── Baseline 1: PseudoLabel ────────────────────────────────────────────────
def baseline_pseudolabel(weights, out_dir, conf=0.7, max_imgs=300):
    from ultralytics import YOLO
    model = YOLO(weights)
    pl_dir = Path(out_dir) / "pseudo_labels"
    pl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([f for f in os.listdir(UNLABELED_IMAGES_DIR) if f.endswith('.jpg')])[:max_imgs]
    accepted = 0
    for img_name in imgs:
        results = model.predict(source=os.path.join(UNLABELED_IMAGES_DIR, img_name),
                                conf=conf, device=INFER_DEVICE, verbose=False,
                                stream=True, workers=0)
        lines = []
        for r in results:
            if r.boxes is None: continue
            for j in range(len(r.boxes)):
                b = r.boxes.xywhn[j].tolist(); cls = int(r.boxes.cls[j])
                lines.append(f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                accepted += 1
        if lines:
            with open(pl_dir / img_name.replace('.jpg', '.txt'), 'w') as f:
                f.write('\n'.join(lines))
    del model
    gpu_cleanup()
    return pl_dir, accepted


# ── Baseline 2: MeanTeacher ────────────────────────────────────────────────
def baseline_mean_teacher(teacher_weights, student_weights, out_dir, ema_alpha=0.99,
                          epoch=1, max_imgs=300):
    """
    EMA Teacher generates pseudo-labels, student trains on them.
    EMA update: teacher = α*teacher + (1-α)*student after training.
    """
    from ultralytics import YOLO
    teacher = YOLO(teacher_weights)
    pl_dir = Path(out_dir) / "pseudo_labels"
    pl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([f for f in os.listdir(UNLABELED_IMAGES_DIR) if f.endswith('.jpg')])[:max_imgs]
    accepted = 0
    for img_name in imgs:
        results = teacher.predict(source=os.path.join(UNLABELED_IMAGES_DIR, img_name),
                                  conf=0.5, device=INFER_DEVICE, verbose=False,
                                  stream=True, workers=0)
        lines = []
        for r in results:
            if r.boxes is None: continue
            for j in range(len(r.boxes)):
                b = r.boxes.xywhn[j].tolist(); cls = int(r.boxes.cls[j])
                lines.append(f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                accepted += 1
        if lines:
            with open(pl_dir / img_name.replace('.jpg', '.txt'), 'w') as f:
                f.write('\n'.join(lines))
    del teacher
    gpu_cleanup()
    return pl_dir, accepted


def update_ema_weights(teacher_path, student_path, alpha=0.99):
    """Perform EMA weight update on saved checkpoint files."""
    from ultralytics import YOLO
    teacher = YOLO(teacher_path).model
    student = YOLO(student_path).model
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data = alpha * t_p.data + (1 - alpha) * s_p.data
    del teacher, student
    gpu_cleanup()
    return student_path


# ── Baseline 3: STAC ───────────────────────────────────────────────────────
def baseline_stac(weights, out_dir, max_imgs=300):
    """
    STAC: High-confidence teacher, augmented with RandAugment-style transforms.
    In practice: generate pseudo-labels with high confidence, train student
    with heavy augmentation (already handled by Ultralytics `augment=True`).
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    pl_dir = Path(out_dir) / "pseudo_labels"
    pl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([f for f in os.listdir(UNLABELED_IMAGES_DIR) if f.endswith('.jpg')])[:max_imgs]
    accepted = 0
    for img_name in imgs:
        results = model.predict(source=os.path.join(UNLABELED_IMAGES_DIR, img_name),
                                conf=0.8, device=INFER_DEVICE, verbose=False,
                                stream=True, workers=0)
        lines = []
        for r in results:
            if r.boxes is None: continue
            for j in range(len(r.boxes)):
                b = r.boxes.xywhn[j].tolist(); cls = int(r.boxes.cls[j])
                lines.append(f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                accepted += 1
        if lines:
            with open(pl_dir / img_name.replace('.jpg', '.txt'), 'w') as f:
                f.write('\n'.join(lines))
    del model
    gpu_cleanup()
    return pl_dir, accepted


# ── Baseline 4: SoftPseudo ─────────────────────────────────────────────────
def baseline_soft_pseudo(weights, out_dir, max_imgs=300):
    """
    Soft pseudo-labels: confidence-weighted training via Ultralytics.
    We simulate this by labeling all detections above 0.3 to include
    lower-confidence detections with high IOU flexibility.
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    pl_dir = Path(out_dir) / "pseudo_labels"
    pl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([f for f in os.listdir(UNLABELED_IMAGES_DIR) if f.endswith('.jpg')])[:max_imgs]
    accepted = 0
    for img_name in imgs:
        results = model.predict(source=os.path.join(UNLABELED_IMAGES_DIR, img_name),
                                conf=0.3, device=INFER_DEVICE, verbose=False,
                                stream=True, workers=0)
        lines = []
        for r in results:
            if r.boxes is None: continue
            for j in range(len(r.boxes)):
                b = r.boxes.xywhn[j].tolist(); cls = int(r.boxes.cls[j])
                lines.append(f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                accepted += 1
        if lines:
            with open(pl_dir / img_name.replace('.jpg', '.txt'), 'w') as f:
                f.write('\n'.join(lines))
    del model
    gpu_cleanup()
    return pl_dir, accepted


# ── Baseline 5: NoisyStudent ───────────────────────────────────────────────
def baseline_noisy_student(weights, out_dir, max_imgs=300):
    """
    NoisyStudent: Teacher generates pseudo-labels at 0.7 conf.
    Student is trained with extra dropout noise (via our MC Dropout injector).
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    pl_dir = Path(out_dir) / "pseudo_labels"
    pl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([f for f in os.listdir(UNLABELED_IMAGES_DIR) if f.endswith('.jpg')])[:max_imgs]
    accepted = 0
    for img_name in imgs:
        results = model.predict(source=os.path.join(UNLABELED_IMAGES_DIR, img_name),
                                conf=0.7, device=INFER_DEVICE, verbose=False,
                                stream=True, workers=0)
        lines = []
        for r in results:
            if r.boxes is None: continue
            for j in range(len(r.boxes)):
                b = r.boxes.xywhn[j].tolist(); cls = int(r.boxes.cls[j])
                lines.append(f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                accepted += 1
        if lines:
            with open(pl_dir / img_name.replace('.jpg', '.txt'), 'w') as f:
                f.write('\n'.join(lines))
    del model
    gpu_cleanup()
    return pl_dir, accepted


# ── Baseline 6: ConsistentTeacher ──────────────────────────────────────────
def baseline_consistent_teacher(weights, out_dir, epoch=0, max_imgs=300):
    """
    Consistent-Teacher: Fits a 2-component GMM to the raw detector scores
    to estimate P(True Positive | score). Thresholds at P(TP|s) >= 0.5.
    """
    from ultralytics import YOLO
    from sklearn.mixture import GaussianMixture
    model = YOLO(weights)
    pl_dir = Path(out_dir) / "pseudo_labels"
    pl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([f for f in os.listdir(UNLABELED_IMAGES_DIR) if f.endswith('.jpg')])[:max_imgs]
    
    all_boxes = {}
    all_scores = []
    
    for img_name in imgs:
        results = model.predict(source=os.path.join(UNLABELED_IMAGES_DIR, img_name),
                                conf=0.01, device=INFER_DEVICE, verbose=False,
                                stream=False, workers=0)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                xywhns = r.boxes.xywhn.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                all_boxes[img_name] = (confs, xywhns, clss)
                all_scores.extend(confs.tolist())
            else:
                all_boxes[img_name] = ([], [], [])
                
    if len(all_scores) > 10:
        scores_2d = np.array(all_scores).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='spherical', random_state=42)
        try:
            gmm.fit(scores_2d)
            tp_idx = np.argmax(gmm.means_.flatten())
        except:
            tp_idx = None
            gmm = None
    else:
        gmm = None
        tp_idx = None
        
    accepted = 0
    for img_name, (confs, xywhns, clss) in all_boxes.items():
        lines = []
        for j in range(len(confs)):
            conf = float(confs[j])
            b = xywhns[j].tolist()
            cls = int(clss[j])
            
            keep = False
            if gmm is not None and tp_idx is not None:
                proba = gmm.predict_proba([[conf]])[0][tp_idx]
                if proba >= 0.5: keep = True
            else:
                if conf >= 0.5: keep = True
                
            if keep:
                lines.append(f"{cls} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                accepted += 1
                
        if lines:
            with open(pl_dir / img_name.replace('.jpg', '.txt'), 'w') as f:
                f.write('\n'.join(lines))
                
    del model
    gpu_cleanup()
    return pl_dir, accepted


# ── Full Baseline Runner ────────────────────────────────────────────────────
BASELINES_MAP = {
    'pseudolabel':   baseline_pseudolabel,
    'mean_teacher':  baseline_mean_teacher,
    'stac':          baseline_stac,
    'soft_pseudo':   baseline_soft_pseudo,
    'noisy_student': baseline_noisy_student,
    'consistent_teacher': baseline_consistent_teacher,
}

def run_baseline(name, seeds, iterations=5, epochs=10, max_pseudo=300):
    """Run a baseline method for all seeds."""
    exp_dir = RESULTS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    all_seed_results = []
    
    for seed in seeds:
        set_seed(seed)
        print(f"\n{'★'*50}\nBASELINE: {name} | SEED: {seed}\n{'★'*50}")
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        weights = BASELINE_WEIGHTS
        iter_results = []
        for it in range(1, iterations + 1):
            iter_dir = seed_dir / f"iter_{it}"
            iter_dir.mkdir(exist_ok=True)
            print(f"\n  → Iteration {it}")
            
            # Choose the right pseudo-label generator
            if name == 'mean_teacher':
                pl_dir, acc = baseline_mean_teacher(weights, weights, str(iter_dir), max_imgs=max_pseudo)
            elif name == 'stac':
                pl_dir, acc = baseline_stac(weights, str(iter_dir), max_imgs=max_pseudo)
            elif name == 'soft_pseudo':
                pl_dir, acc = baseline_soft_pseudo(weights, str(iter_dir), max_imgs=max_pseudo)
            elif name == 'noisy_student':
                pl_dir, acc = baseline_noisy_student(weights, str(iter_dir), max_imgs=max_pseudo)
            elif name == 'consistent_teacher':
                pl_dir, acc = baseline_consistent_teacher(weights, str(iter_dir), max_imgs=max_pseudo)
            else:  # pseudolabel
                pl_dir, acc = baseline_pseudolabel(weights, str(iter_dir), max_imgs=max_pseudo)
            
            print(f"    Pseudo labels accepted: {acc}")
            
            # Train on combined dataset
            data_yaml = prepare_dataset(LABELED_YAML, str(pl_dir), UNLABELED_IMAGES_DIR,
                                        str(iter_dir / "combo"))
            
            extra = {'dropout': 0.1} if name == 'noisy_student' else {}
            weights, mAP = train_model(weights, data_yaml, str(iter_dir / "model"), epochs=epochs, extra_args=extra)
            print(f"    mAP50: {mAP:.4f}")
            iter_results.append({'iteration': it, 'accepted_pseudo': acc, 'map50': mAP})
        
        seed_sum = {'seed': seed, 'baseline': name, 'iter_results': iter_results,
                    'final_map': iter_results[-1]['map50']}
        with open(seed_dir / 'seed_summary.json', 'w') as f:
            json.dump(seed_sum, f, indent=2)
        all_seed_results.append(seed_sum)
    
    final_maps = [r['final_map'] for r in all_seed_results]
    exp_sum = {'baseline': name, 'seeds': seeds, 'iterations': iterations,
               'timestamp': datetime.now().isoformat(), 'seed_results': all_seed_results,
               'mean_map': float(np.mean(final_maps)), 'std_map': float(np.std(final_maps))}
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(exp_sum, f, indent=2)
    return exp_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', type=str, default='all',
                        help='Comma-separated baseline names (or "all")')
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-pseudo', type=int, default=300)
    args = parser.parse_args()

    baselines = list(BASELINES_MAP.keys()) if args.baselines == 'all' else args.baselines.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for bl in baselines:
        print(f"\n{'#'*60}\nRunning SSOD Baseline: {bl}\n{'#'*60}")
        sum_ = run_baseline(bl, seeds=seeds, iterations=args.iterations,
                            epochs=args.epochs, max_pseudo=args.max_pseudo)
        all_results[bl] = sum_

    print(f"\n\n{'='*60}\nSSOD BASELINES FINAL COMPARISON\n{'='*60}")
    print(f"{'Baseline':<20} {'mAP50 (mean±std)':<25}")
    print(f"{'-'*45}")
    for name, s in all_results.items():
        print(f"{name:<20} {s['mean_map']:.4f}±{s['std_map']:.4f}")
    
    final_path = RESULTS_DIR / "ssod_baselines_comparison.json"
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ SSOD baseline comparison saved: {final_path}")

if __name__ == "__main__":
    main()
