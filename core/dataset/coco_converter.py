"""
Convert SKU-110K to COCO format for Faster R-CNN (Detectron2).

SKU-110K format: CSV with columns [image_name, x1, y1, x2, y2, class, ...]
COCO format: JSON with images, annotations, categories
"""

import pandas as pd
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_exact import DATA_ROOT

def convert_to_coco_format():
    """
    Convert SKU-110K annotations to COCO format.
    """
    
    print("="*80)
    print("CONVERTING TO COCO FORMAT")
    print("="*80)
    
    # Load splits
    splits_file = DATA_ROOT / "splits" / "reproduction_splits.json"
    if not splits_file.exists():
        print(f"✗ Splits file not found: {splits_file}")
        print("Please run: python core/dataset/create_splits.py")
        return
    
    with open(splits_file) as f:
        splits = json.load(f)
    
    # Load ALL annotation files (train, val, test) and combine
    ann_dir = DATA_ROOT / "raw" / "SKU110K_fixed" / "annotations"
    
    dfs = []
    for ann_file in ['annotations_train.csv', 'annotations_val.csv', 'annotations_test.csv']:
        ann_path = ann_dir / ann_file
        if ann_path.exists():
            df_part = pd.read_csv(ann_path, header=None,
                                 names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'])
            dfs.append(df_part)
    
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"Loaded {len(df)} annotations from all splits")
    
    # OPTIMIZATION: Group annotations by image name for fast lookup
    print("Grouping annotations by image...")
    annotations_by_image = df.groupby('image_name')
    print("✓ Done")
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        print(f"\nProcessing {split_name}...")
        
        image_list = splits[split_name]
        
        coco_format = {
            'info': {
                'description': f'SKU-110K {split_name} set for paper reproduction',
                'version': '1.0',
                'year': 2025,
                'dataset': 'SKU-110K'
            },
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'product', 'supercategory': 'retail'}
            ]
        }
        
        image_id = 0
        annotation_id = 0
        
        for img_name in tqdm(image_list, desc=f"Converting {split_name}"):
            # Get image info
            img_path = DATA_ROOT / "raw" / "SKU110K_fixed" / "images" / img_name
            
            try:
                img = Image.open(img_path)
                width, height = img.size
            except Exception as e:
                print(f"Warning: Could not open {img_name}: {e}")
                continue
            
            coco_format['images'].append({
                'id': image_id,
                'file_name': img_name,
                'width': width,
                'height': height
            })
            
            # Get annotations for this image (fast lookup using grouped data)
            try:
                img_anns = annotations_by_image.get_group(img_name)
            except KeyError:
                # No annotations for this image (shouldn't happen for labeled splits)
                img_anns = pd.DataFrame()
            
            for _, row in img_anns.iterrows():
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Skip invalid boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                
                coco_format['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': 1,  # Single class
                    'bbox': [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                    'area': float(bbox_width * bbox_height),
                    'iscrowd': 0
                })
                annotation_id += 1
            
            image_id += 1
        
        # Save
        output_path = DATA_ROOT / "coco_format" / f"{split_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(coco_format, f)
        
        print(f"✓ {split_name}: {len(coco_format['images'])} images, {len(coco_format['annotations'])} annotations")
        print(f"  Saved to: {output_path}")
    
    print("\n✓ COCO conversion complete")

if __name__ == "__main__":
    convert_to_coco_format()
    print("\n✓ Phase 1.3 COMPLETE: COCO format created")
    print("\nNext step:")
    print("  python core/dataset/yolo_converter.py")
