#!/usr/bin/env python3
"""
Create 10% and 20% subsets of SKU110K coco_format/train.json matching the YOLO splits.
"""

import json
import os
from pathlib import Path

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
SKU_DIR = PROJ / "data" / "SKU110K"
SRC_JSON = SKU_DIR / "coco_format" / "train.json"

def create_sku_subset(pct=10):
    dst_json = SKU_DIR / "coco_format" / f"train_{pct}.json"
    print(f"Loading {SRC_JSON}...")
    
    with open(SRC_JSON, 'r') as f:
        coco = json.load(f)
        
    yolo_train_img_dir = SKU_DIR / "yolo_format" / f"frac_{pct}" / "train" / "images"
    if not yolo_train_img_dir.exists():
        print(f"Error: {yolo_train_img_dir} not found. Generate YOLO splits first.")
        return
        
    valid_filenames = set(os.listdir(yolo_train_img_dir))
    
    subset_images = [img for img in coco['images'] if img['file_name'] in valid_filenames]
    subset_image_ids = set(img['id'] for img in subset_images)
    
    print(f"Original images: {len(coco['images'])}")
    print(f"Subset images ({pct}%): {len(subset_images)} (matched from YOLO dir)")
    
    # Filter annotations
    subset_anns = [ann for ann in coco['annotations'] if ann['image_id'] in subset_image_ids]
    print(f"Subset annotations: {len(subset_anns)}")
    
    # Build new JSON
    new_coco = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': subset_images,
        'annotations': subset_anns,
        'categories': coco['categories']
    }
    
    with open(dst_json, 'w') as f:
        json.dump(new_coco, f)
        
    print(f"✓ Created {dst_json}")

if __name__ == "__main__":
    create_sku_subset(10)
    create_sku_subset(20)
