#!/usr/bin/env python3
"""
FlexMatch SSOD Baseline (Curriculum Pseudo-Labeling)
===================================================
BMVC 2026 Comparison Method

Implements curriculum pseudo-labeling where the threshold for each class
adapts based on the model's learning status for that specific class.
"""

import argparse
import json
import os
import sys
import random
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path

# Disable CuDNN to prevent K80 crashes with bf16
import torch
torch.backends.cudnn.enabled = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse helpers
from scripts.calibpl_selftrain import _setup_gpu, set_seed, prepare_dataset, train_iteration

class FlexMatchThresholds:
    """Manages dynamic per-class thresholds based on learning status."""
    def __init__(self, num_classes, base_threshold=0.5):
        self.nc = num_classes
        self.base_tau = base_threshold
        # Learning status: estimated probability of correct prediction per class
        # Initialized to 0 (no learning)
        self.learning_status = np.zeros(num_classes)
        # Beta maps learning status to threshold scaling
        self.beta = np.zeros(num_classes)
        # Global learning effect
        self.global_status = 0.0
        
    def update(self, val_results_box):
        """Update learning status based on validation metrics (e.g., class-wise mAP50)."""
        # Note: YOLO results.box contains maps for each class present in val set
        if val_results_box is None:
            return
            
        maps = val_results_box.maps # This is mAP50-95 per class usually, or need to extract mAP50
        classes = val_results_box.ap_class_index if hasattr(val_results_box, 'ap_class_index') else np.arange(self.nc)
        
        # We use mAP50 as a proxy for "learning status"
        # Since not all classes might be in val, we only update present classes
        if hasattr(val_results_box, 'map50'):
            # If maps is an array aligned with ap_class_index
            if isinstance(maps, np.ndarray) and len(maps) == len(classes):
                for i, c in enumerate(classes):
                    if c < self.nc:
                        # Smooth update
                        self.learning_status[c] = 0.5 * self.learning_status[c] + 0.5 * float(maps[i])
        
        # Max learning status across any class
        max_status = np.max(self.learning_status)
        if max_status > 0:
            # Normalize so the best class gets full base_tau
            self.beta = self.learning_status / max_status
        else:
            self.beta = np.ones(self.nc)
            
    def get_threshold(self, cls_id):
        """Get the adaptive threshold for a specific class."""
        if cls_id >= self.nc:
            return self.base_tau
        
        # FlexMatch logic: tau_c = base_tau * beta_c
        # If class is poorly learned, beta is low -> threshold is LOWERED to admit more pseudo-labels
        # WAIT, FlexMatch actually scales the threshold so that poorly learned classes have *lower* thresholds
        # to encourage learning them. 
        # Actually, standard curriculum: start high, relax later.
        # FlexMatch: T_t(c) = p_cutoff * (sigma_t(c) / max_c' sigma_t(c))
        # Meaning well-learned classes get high thresholds (only confident ones), poorly learned get low thresholds.
        
        # To avoid accepting garbage, we floor it at something sensible, e.g., 0.2
        scaled_tau = self.base_tau * self.beta[cls_id]
        return max(0.2, scaled_tau)
        

def generate_flexmatch_pseudo_labels(
    model_path: str,
    unlabeled_dir: str,
    output_dir: str,
    thresholds: FlexMatchThresholds,
    max_images: int = 0
):
    from ultralytics import YOLO
    model = YOLO(model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(unlabeled_dir) if f.endswith(('.jpg', '.png'))])
    if max_images > 0:
        image_files = random.sample(image_files, min(max_images, len(image_files)))
        
    stats = {'total_images': 0, 'images_with_labels': 0, 'total_boxes_kept': 0, 'total_boxes_rejected': 0}
    class_counts = defaultdict(int)
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(unlabeled_dir, img_name)
        
        # Predict with low threshold to let FlexMatch filter
        results = model.predict(img_path, device=0, conf=0.05, verbose=False, max_det=300)
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            label_lines = []
            
            for j in range(len(boxes)):
                cls = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                cx, cy, w, h = boxes.xywhn[j].cpu().numpy()
                
                tau_c = thresholds.get_threshold(cls)
                
                if conf >= tau_c:
                    label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    stats['total_boxes_kept'] += 1
                    class_counts[cls] += 1
                else:
                    stats['total_boxes_rejected'] += 1
            
            if label_lines:
                label_path = output_path / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
                stats['images_with_labels'] += 1
                
        stats['total_images'] += 1
        if (i + 1) % 50 == 0:
            print(f"    FlexMatch: {i+1}/{len(image_files)} images, {stats['total_boxes_kept']} kept")
            
    del model
    torch.cuda.empty_cache()
    return stats, dict(class_counts)


def run_flexmatch(args):
    import yaml
    set_seed(args.seed)
    
    model_weights = str(PROJECT_ROOT / "models" / "yolo12n.pt")
    
    # Needs a trained baseline model initially to have valid val metrics
    if "sku" in args.data_yaml.lower() and os.path.exists(PROJECT_ROOT / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt"):
        model_weights = str(PROJECT_ROOT / "results" / "week1_baselines" / "yolov12" / "train" / "weights" / "best.pt")
        print(f"Using pretrained baseline for SK110K.")
    elif "coco" in args.data_yaml.lower() and os.path.exists(PROJECT_ROOT / "results" / "coco1pct" / "supervised" / "train_seed42" / "weights" / "best.pt"):
        model_weights = str(PROJECT_ROOT / "results" / "coco1pct" / "supervised" / "train_seed42" / "weights" / "best.pt")
        print(f"Using supervised COCO baseline.")
        
    output_dir = PROJECT_ROOT / "results" / "baselines" / f"flexmatch_seed{args.seed}_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    num_classes = data_cfg.get('nc', 80)
    
    thresholds = FlexMatchThresholds(num_classes, base_threshold=args.base_threshold)
    
    print(f"\n{'='*70}")
    print(f" FlexMatch SSOD Baseline")
    print(f" Dataset: {args.data_yaml}")
    print(f" Classes: {num_classes}")
    print(f"{'='*70}")
    
    results_log = {'method': 'flexmatch', 'tag': args.tag, 'seed': args.seed, 'iterations': []}
    
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f" ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")
        
        iter_dir = output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        # 0. Update Learning Status (Thresholds)
        # We need to run validation to get per-class metrics
        print("\n  [Step 0] Evaluating Model to Update Class Learning Status...")
        from ultralytics import YOLO
        val_model = YOLO(model_weights)
        val_res = val_model.val(data=args.data_yaml, device=0, split='val', verbose=False)
        thresholds.update(val_res.box)
        del val_model
        torch.cuda.empty_cache()
        
        # Debug print a few thresholds
        print(f"    Class 0 tau: {thresholds.get_threshold(0):.3f}")
        if num_classes > 1:
            print(f"    Class 1 tau: {thresholds.get_threshold(1):.3f}")
            
        # 1. Pseudo-Label Generation
        print(f"\n  [Step 1] Generating FlexMatch Pseudo-Labels...")
        pseudo_dir = iter_dir / "pseudo_labels"
        stats, class_counts = generate_flexmatch_pseudo_labels(
            model_weights, args.unlabeled_dir, str(pseudo_dir),
            thresholds=thresholds, max_images=args.max_pseudo
        )
        print(f"    → Kept {stats['total_boxes_kept']} boxes, rejected {stats['total_boxes_rejected']}")
        
        # 2. Prepare Dataset
        print(f"\n  [Step 2] Preparing SSOD Dataset...")
        combo_dir = iter_dir / "dataset"
        ds_yaml = prepare_dataset(args.data_yaml, pseudo_dir, args.unlabeled_dir, combo_dir)
        
        # 3. Train Model
        print(f"\n  [Step 3] Training Student Model...")
        new_weights, map50 = train_iteration(
            model_weights, ds_yaml, str(iter_dir / "model"), 
            epochs=args.epochs, batch_size=args.batch_size
        )
        print(f"    → Iteration {iteration} mAP50: {map50:.4f}")
        
        model_weights = new_weights
        
        # Log
        results_log['iterations'].append({
            'iteration': iteration,
            'pseudo_stats': stats,
            'class_counts': class_counts,
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
    parser.add_argument('--data-yaml', required=True)
    parser.add_argument('--unlabeled-dir', required=True)
    parser.add_argument('--tag', required=True)
    
    parser.add_argument('--train-device', default='0')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-pseudo', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--base-threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    _setup_gpu(args.train_device)
    run_flexmatch(args)

if __name__ == '__main__':
    main()
