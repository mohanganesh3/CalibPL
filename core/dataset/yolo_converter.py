"""
Convert SKU-110K to YOLO format for YOLOv3 (Darknet).

SKU-110K format: CSV with [image_name, x1, y1, x2, y2, class, ...]
YOLO format: TXT with [class_id, x_center, y_center, width, height] (normalized 0-1)
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

def convert_to_yolo_format():
    """
    Convert SKU-110K annotations to YOLO format (Darknet).
    """
    
    print("="*80)
    print("CONVERTING TO YOLO FORMAT")
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
    for split_name in ['train', 'val', 'test', 'unlabeled']:
        print(f"\nProcessing {split_name}...")
        
        image_list = splits[split_name]
        
        # Create directories
        images_dir = DATA_ROOT / "yolo_format" / split_name / "images"
        labels_dir = DATA_ROOT / "yolo_format" / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file list
        file_list = []
        
        for img_name in tqdm(image_list, desc=f"Converting {split_name}"):
            # Get image dimensions
            img_path = DATA_ROOT / "raw" / "SKU110K_fixed" / "images" / img_name
            
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Warning: Could not open {img_name}: {e}")
                continue
            
            # For unlabeled images, create empty label file
            if split_name == 'unlabeled':
                label_file = labels_dir / f"{Path(img_name).stem}.txt"
                label_file.touch()  # Empty file
                file_list.append(str(images_dir / img_name))
                continue
            
            # Get annotations for this image (fast lookup using grouped data)
            try:
                img_anns = annotations_by_image.get_group(img_name)
            except KeyError:
                # No annotations for this image
                img_anns = pd.DataFrame()
            
            # Convert to YOLO format
            yolo_lines = []
            for _, row in img_anns.iterrows():
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Skip invalid boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                
                center_x = (x1 + x2) / 2.0 / img_width
                center_y = (y1 + y2) / 2.0 / img_height
                norm_width = bbox_width / img_width
                norm_height = bbox_height / img_height
                
                # Clip to [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                # YOLO format: class_id center_x center_y width height
                # Class 0 = product (single class)
                yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # Save label file
            label_file = labels_dir / f"{Path(img_name).stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            file_list.append(str(images_dir / img_name))
        
        # Save file list for Darknet
        list_file = DATA_ROOT / "yolo_format" / f"{split_name}.txt"
        with open(list_file, 'w') as f:
            f.write('\n'.join(file_list))
        
        print(f"✓ {split_name}: {len(image_list)} images")
        print(f"  Labels: {labels_dir}")
        print(f"  List: {list_file}")
    
    # Create data config file for Darknet
    data_cfg = f"""classes=1
train={DATA_ROOT}/yolo_format/train.txt
valid={DATA_ROOT}/yolo_format/val.txt
names={DATA_ROOT}/yolo_format/sku110k.names
backup={DATA_ROOT}/checkpoints/yolo
"""
    
    data_cfg_path = DATA_ROOT / "yolo_format" / "sku110k.data"
    with open(data_cfg_path, 'w') as f:
        f.write(data_cfg)
    
    # Create names file
    names_path = DATA_ROOT / "yolo_format" / "sku110k.names"
    with open(names_path, 'w') as f:
        f.write("product\n")
    
    print(f"\n✓ Created Darknet config: {data_cfg_path}")
    print(f"✓ Created class names: {names_path}")
    print("\n✓ YOLO conversion complete")

if __name__ == "__main__":
    convert_to_yolo_format()
    print("\n✓ Phase 1.4 COMPLETE: YOLO format created")
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE: Dataset Preparation")
    print("="*80)
    print("\nSummary:")
    print("  ✓ Downloaded SKU-110K (11GB)")
    print("  ✓ Created exact splits (1400/200/400/8000)")
    print("  ✓ Converted to COCO format (Faster R-CNN)")
    print("  ✓ Converted to YOLO format (YOLOv3)")
    print("\nNext Phase:")
    print("  Phase 2: Baseline Model Training")
    print("  See: PHASES_2_TO_6_IMPLEMENTATION.md")
