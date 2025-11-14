#!/usr/bin/env python3
"""
Create image symlinks so Ultralytics can find images next to labels.
Ultralytics expects: split/images/xxx.jpg alongside split/labels/xxx.txt
"""

import os
from pathlib import Path

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
YOLO_DIR = PROJECT_ROOT / "data" / "SKU110K" / "yolo_format"
RAW_IMAGES = PROJECT_ROOT / "data" / "SKU110K" / "raw" / "SKU110K_fixed" / "images"

for split in ['train', 'val', 'test', 'unlabeled']:
    labels_dir = YOLO_DIR / split / "labels"
    images_dir = YOLO_DIR / split / "images"
    
    if not labels_dir.exists():
        print(f"⚠ {split}/labels/ not found, skipping")
        continue
    
    images_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for label_file in labels_dir.iterdir():
        if label_file.suffix != '.txt':
            continue
        
        # Find corresponding image
        img_name = label_file.stem + '.jpg'
        src_img = RAW_IMAGES / img_name
        dst_img = images_dir / img_name
        
        if src_img.exists() and not dst_img.exists():
            os.symlink(src_img, dst_img)
            count += 1
        elif not src_img.exists():
            # Try with different naming
            for ext in ['.jpg', '.JPG', '.png', '.PNG']:
                alt = RAW_IMAGES / (label_file.stem + ext)
                if alt.exists() and not dst_img.exists():
                    os.symlink(alt, images_dir / (label_file.stem + ext))
                    count += 1
                    break
    
    total = len(list(images_dir.iterdir()))
    print(f"✓ {split}: created {count} symlinks ({total} total images)")

print("\nDone! Dataset ready for Ultralytics training.")
