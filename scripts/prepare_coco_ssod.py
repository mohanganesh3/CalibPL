#!/usr/bin/env python3
"""
Prepare COCO 2017 for Semi-Supervised Object Detection experiments.
Creates 1% and 5% labeled splits with the rest as unlabeled.

Steps:
1. Extract COCO if not already extracted
2. Convert COCO JSON annotations to YOLO format
3. Create 1% and 5% splits (stratified by image to maintain distribution)
4. Create dataset YAML files for each split

Usage:
    python3 scripts/prepare_coco_ssod.py
"""

import json
import os
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
COCO_DIR = PROJ / "data" / "coco"
YOLO_DIR = PROJ / "data" / "coco" / "yolo_format"


def extract_coco():
    """Extract COCO zips if not already extracted."""
    import zipfile
    
    for name in ["annotations_trainval2017", "val2017", "train2017"]:
        zip_path = COCO_DIR / f"{name}.zip"
        extract_dir = COCO_DIR
        
        # Check if already extracted
        if name == "annotations_trainval2017":
            target = COCO_DIR / "annotations"
        elif name == "val2017":
            target = COCO_DIR / "val2017"
        else:
            target = COCO_DIR / "train2017"
        
        if target.exists():
            print(f"  ✅ {name} already extracted")
            continue
        
        if not zip_path.exists():
            print(f"  ❌ {zip_path} not found! Download first.")
            return False
        
        print(f"  📦 Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print(f"  ✅ {name} extracted")
    
    return True


def coco_to_yolo_annotations():
    """Convert COCO JSON annotations to YOLO format."""
    yolo_train = YOLO_DIR / "train" / "labels"
    yolo_val = YOLO_DIR / "val" / "labels"
    yolo_train.mkdir(parents=True, exist_ok=True)
    yolo_val.mkdir(parents=True, exist_ok=True)
    
    # Also create image symlinks
    yolo_train_img = YOLO_DIR / "train" / "images"
    yolo_val_img = YOLO_DIR / "val" / "images"
    yolo_train_img.mkdir(parents=True, exist_ok=True)
    yolo_val_img.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val"]:
        ann_file = COCO_DIR / "annotations" / f"instances_{split}2017.json"
        if not ann_file.exists():
            print(f"  ❌ {ann_file} not found!")
            return False, {}
        
        print(f"  Converting {split}2017 annotations...")
        with open(ann_file) as f:
            coco = json.load(f)
        
        # Build category mapping (COCO category ID → YOLO class index)
        categories = {c['id']: i for i, c in enumerate(coco['categories'])}
        cat_names = {categories[c['id']]: c['name'] for c in coco['categories']}
        
        # Build image info
        images = {img['id']: img for img in coco['images']}
        
        # Group annotations by image
        img_anns = defaultdict(list)
        for ann in coco['annotations']:
            if ann.get('iscrowd', 0):
                continue
            img_anns[ann['image_id']].append(ann)
        
        # Convert to YOLO format
        label_dir = yolo_train if split == "train" else yolo_val
        img_dir = yolo_train_img if split == "train" else yolo_val_img
        src_img_dir = COCO_DIR / f"{split}2017"
        
        count = 0
        for img_id, img_info in images.items():
            w_img = img_info['width']
            h_img = img_info['height']
            filename = img_info['file_name']
            
            # Symlink image
            src = src_img_dir / filename
            dst = img_dir / filename
            if src.exists() and not dst.exists():
                os.symlink(src.resolve(), dst)
            
            # Write YOLO labels
            anns = img_anns.get(img_id, [])
            if anns:
                label_path = label_dir / filename.replace('.jpg', '.txt')
                lines = []
                for ann in anns:
                    cat_id = categories[ann['category_id']]
                    # COCO bbox: [x, y, w, h] (absolute)
                    bx, by, bw, bh = ann['bbox']
                    # Convert to YOLO: [cx, cy, w, h] (normalized)
                    cx = (bx + bw / 2) / w_img
                    cy = (by + bh / 2) / h_img
                    nw = bw / w_img
                    nh = bh / h_img
                    # Clamp to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(lines))
                count += 1
        
        print(f"  ✅ {split}: {count} images with labels, {len(images)} total")
    
    return True, cat_names


def create_ssod_splits(percentages=[1, 5]):
    """Create labeled/unlabeled splits for SSOD."""
    train_img_dir = YOLO_DIR / "train" / "images"
    train_lbl_dir = YOLO_DIR / "train" / "labels"
    
    # Get all training images with labels
    all_images = sorted([f.name for f in train_img_dir.iterdir() if f.suffix in ('.jpg', '.png')])
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(all_images)
    
    n_total = len(all_images)
    print(f"\n  Total training images: {n_total}")
    
    for pct in percentages:
        n_labeled = max(1, int(n_total * pct / 100))
        
        split_dir = YOLO_DIR / f"frac_{pct}"
        split_train = split_dir / "train"
        split_unlabeled = split_dir / "unlabeled"
        
        split_train_img = split_train / "images"
        split_train_lbl = split_train / "labels"
        split_unlabeled_img = split_unlabeled / "images"
        
        for d in [split_train_img, split_train_lbl, split_unlabeled_img]:
            d.mkdir(parents=True, exist_ok=True)
        
        labeled_images = all_images[:n_labeled]
        unlabeled_images = all_images[n_labeled:]
        
        # Symlink labeled
        for img in labeled_images:
            src_img = train_img_dir / img
            dst_img = split_train_img / img
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
            
            lbl_name = img.replace('.jpg', '.txt')
            src_lbl = train_lbl_dir / lbl_name
            dst_lbl = split_train_lbl / lbl_name
            if src_lbl.exists() and not dst_lbl.exists():
                os.symlink(src_lbl.resolve(), dst_lbl)
        
        # Symlink unlabeled
        for img in unlabeled_images:
            src_img = train_img_dir / img
            dst_img = split_unlabeled_img / img
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
        
        # Symlink val
        val_dir = split_dir / "val"
        val_src = YOLO_DIR / "val"
        if not val_dir.exists():
            os.symlink(val_src.resolve(), val_dir)
        
        # Write YAML
        yaml_path = YOLO_DIR / f"coco_frac_{pct}.yaml"
        yaml_content = {
            'path': str(split_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 80,
            'names': None  # Will be filled with COCO categories
        }
        
        print(f"  ✅ {pct}% split: {n_labeled} labeled + {len(unlabeled_images)} unlabeled")
        
        # Use symlinks for labels.cache compatibility
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
    
    return True


def create_yaml_with_names(cat_names, percentages=[1, 5]):
    """Update YAML files with category names."""
    for pct in percentages:
        yaml_path = YOLO_DIR / f"coco_frac_{pct}.yaml"
        split_dir = YOLO_DIR / f"frac_{pct}"
        
        yaml_content = {
            'path': str(split_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 80,
            'names': cat_names,
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        print(f"  ✅ Updated {yaml_path.name} with {len(cat_names)} categories")


def main():
    print("=" * 60)
    print(" COCO 2017 → Semi-Supervised OD Setup")
    print("=" * 60)
    
    # Step 1: Extract
    print("\n[1/4] Extracting COCO archives...")
    if not extract_coco():
        print("FAILED: Could not extract COCO. Make sure all zips are downloaded.")
        return
    
    # Step 2: Convert annotations
    print("\n[2/4] Converting COCO → YOLO format...")
    success, cat_names = coco_to_yolo_annotations()
    if not success:
        print("FAILED: Could not convert annotations")
        return
    
    # Step 3: Create splits
    print("\n[3/4] Creating 1%/5% labeled splits...")
    if not create_ssod_splits([1, 5]):
        print("FAILED: Could not create splits")
        return
    
    # Step 4: Update YAMLs with category names
    print("\n[4/4] Writing YAML configs...")
    create_yaml_with_names(cat_names, [1, 5])
    
    print("\n" + "=" * 60)
    print(" SETUP COMPLETE!")
    print("=" * 60)
    print(f"\n  YAML files:")
    print(f"    1%: {YOLO_DIR / 'coco_frac_1.yaml'}")
    print(f"    5%: {YOLO_DIR / 'coco_frac_5.yaml'}")
    print(f"\n  To run CW-PL on COCO 1%:")
    print(f"    python3 scripts/cwpl_cotrain.py --mode cwpl \\")
    print(f"      --data-yaml {YOLO_DIR / 'coco_frac_1.yaml'} \\")
    print(f"      --unlabeled-dir {YOLO_DIR / 'frac_1/unlabeled/images'} \\")
    print(f"      --train-device 1 --output-tag coco1pct")


if __name__ == '__main__':
    main()
