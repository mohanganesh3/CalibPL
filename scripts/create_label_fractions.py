#!/usr/bin/env python3
"""
Create Label-Fraction Splits for BMVC 2026 Experiments
======================================================
Creates 10%, 20%, 50% subsets of the training data.
For each fraction:
  - Copies subset of labeled images + labels
  - Remaining labeled images become "extra unlabeled"
  - Combined with original 8,000 unlabeled for total unlabeled pool
  - Creates YAML config for each fraction
"""

import os
import sys
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
DATA_DIR = PROJECT_ROOT / "data" / "SKU110K" / "yolo_format"
TRAIN_IMAGES = DATA_DIR / "train" / "images"
TRAIN_LABELS = DATA_DIR / "train" / "labels"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

FRACTIONS = [0.10, 0.20, 0.50]
SEED = 42


def create_fraction_split(fraction: float, seed: int = 42):
    """Create a label-fraction split."""
    random.seed(seed)
    
    # Get all training images
    all_images = sorted([f for f in os.listdir(TRAIN_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))])
    n_total = len(all_images)
    n_labeled = max(1, int(n_total * fraction))
    
    # Shuffle and split
    shuffled = all_images.copy()
    random.shuffle(shuffled)
    labeled_images = shuffled[:n_labeled]
    extra_unlabeled_images = shuffled[n_labeled:]
    
    frac_name = f"frac_{int(fraction * 100)}"
    frac_dir = DATA_DIR / frac_name
    
    # Create directory structure
    labeled_img_dir = frac_dir / "train" / "images"
    labeled_lbl_dir = frac_dir / "train" / "labels"
    unlabeled_img_dir = frac_dir / "unlabeled" / "images"
    
    for d in [labeled_img_dir, labeled_lbl_dir, unlabeled_img_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Symlink labeled images and labels
    for img in labeled_images:
        src_img = TRAIN_IMAGES / img
        dst_img = labeled_img_dir / img
        if not dst_img.exists():
            os.symlink(src_img, dst_img)
        
        # Corresponding label
        lbl = img.rsplit('.', 1)[0] + '.txt'
        src_lbl = TRAIN_LABELS / lbl
        dst_lbl = labeled_lbl_dir / lbl
        if src_lbl.exists() and not dst_lbl.exists():
            os.symlink(src_lbl, dst_lbl)
    
    # Symlink extra unlabeled (from the remaining labeled set)
    for img in extra_unlabeled_images:
        src_img = TRAIN_IMAGES / img
        dst_img = unlabeled_img_dir / img
        if not dst_img.exists():
            os.symlink(src_img, dst_img)
    
    # Also symlink original unlabeled images
    orig_unlabeled = DATA_DIR / "unlabeled" / "images"
    if orig_unlabeled.exists():
        for img in os.listdir(orig_unlabeled):
            src_img = orig_unlabeled / img
            dst_img = unlabeled_img_dir / img
            if not dst_img.exists():
                os.symlink(src_img, dst_img)
    
    # Create YAML config
    yaml_path = DATA_DIR / f"{frac_name}.yaml"
    yaml_content = f"""# SKU-110K {int(fraction*100)}% Label Fraction
# {n_labeled} labeled / {len(extra_unlabeled_images) + len(os.listdir(orig_unlabeled))} unlabeled

path: {frac_dir}
train: train/images
val: {VAL_DIR}
test: {TEST_DIR}

nc: 1
names:
  0: product
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    n_unlabeled = len(os.listdir(unlabeled_img_dir))
    print(f"  {frac_name}: {n_labeled} labeled, {n_unlabeled} unlabeled, ratio 1:{n_unlabeled/n_labeled:.1f}")
    print(f"  YAML: {yaml_path}")
    print(f"  Dirs: {labeled_img_dir} | {unlabeled_img_dir}")
    
    return {
        'fraction': fraction,
        'name': frac_name,
        'n_labeled': n_labeled,
        'n_unlabeled': n_unlabeled,
        'yaml': str(yaml_path),
        'labeled_dir': str(labeled_img_dir),
        'unlabeled_dir': str(unlabeled_img_dir),
    }


def main():
    print("=" * 60)
    print("Creating Label-Fraction Splits for BMVC 2026")
    print("=" * 60)
    
    n_total = len(os.listdir(TRAIN_IMAGES))
    print(f"Total training images: {n_total}")
    print(f"Seed: {SEED}")
    print()
    
    results = []
    for frac in FRACTIONS:
        results.append(create_fraction_split(frac, SEED))
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Fraction':<12} {'Labeled':<10} {'Unlabeled':<12} {'Ratio':<10} {'YAML'}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<12} {r['n_labeled']:<10} {r['n_unlabeled']:<12} 1:{r['n_unlabeled']/r['n_labeled']:<8.1f} {r['yaml']}")
    
    # Also note the 100% setting
    n_unlabeled_100 = len(os.listdir(DATA_DIR / "unlabeled" / "images"))
    print(f"{'100%':<12} {n_total:<10} {n_unlabeled_100:<12} 1:{n_unlabeled_100/n_total:<8.1f} data/SKU110K/yolo_format/sku110k.yaml (original)")
    
    print("\nDone! Use these YAML files with run_calibcotrain.py")


if __name__ == '__main__':
    main()
