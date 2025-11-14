"""
Create a 1% subset of COCO instances_train2017.json for standard SSOD benchmarking.
Matches the seed used in YOLO split generation.
"""

import json
import random
from pathlib import Path

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
ANN_DIR = PROJ / "data" / "coco" / "annotations"
SRC_JSON = ANN_DIR / "instances_train2017.json"
def create_subset(pct=1):
    dst_json = ANN_DIR / f"instances_train2017_{pct}.json"
    print(f"Loading {SRC_JSON}...")
    with open(SRC_JSON, 'r') as f:
        coco = json.load(f)
    
    images = coco['images']
    random.seed(42)  # Match seed 42 from prepare_coco_ssod.py
    random.shuffle(images)
    
    n_subset = max(1, int(len(images) * pct / 100))
    subset_images = images[:n_subset]
    subset_image_ids = set(img['id'] for img in subset_images)
    
    print(f"Original images: {len(images)}")
    print(f"Subset images ({pct}%): {len(subset_images)}")
    
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
    create_subset(1)
    create_subset(5)
