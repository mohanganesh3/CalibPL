"""
Create exact dataset splits as specified in paper Section 3.1.

Paper specifications:
- 10,000 images selected from SKU-110K
- 2,000 labeled (70% train, 10% val, 20% test)
- 8,000 unlabeled

Output: JSON file with image lists for each split.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_exact import DATASET_CONFIG, DATA_ROOT

def create_exact_splits():
    """
    Create splits EXACTLY as paper specifies.
    """
    
    print("="*80)
    print("CREATING EXACT DATASET SPLITS")
    print("="*80)
    
    # Load annotations
    ann_path = DATA_ROOT / "raw" / "SKU110K_fixed" / "annotations" / "annotations_train.csv"
    
    if not ann_path.exists():
        print(f"✗ Annotations not found at: {ann_path}")
        print("Please check if dataset was extracted correctly")
        return None
    
    # Load ALL annotation files (train, val, test) and combine
    # Paper selected 10,000 images from the entire dataset
    ann_dir = DATA_ROOT / "raw" / "SKU110K_fixed" / "annotations"
    
    dfs = []
    for ann_file in ['annotations_train.csv', 'annotations_val.csv', 'annotations_test.csv']:
        ann_path = ann_dir / ann_file
        if ann_path.exists():
            df_part = pd.read_csv(ann_path, header=None, 
                                 names=['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'])
            dfs.append(df_part)
            print(f"Loaded {ann_file}: {len(df_part)} annotations, {df_part['image_name'].nunique()} unique images")
    
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal annotations: {len(df)}")
    
    # Get unique images
    all_images = df['image_name'].unique()
    print(f"Total unique images: {len(all_images)}")
    
    # Set seed for reproducibility (CRITICAL)
    np.random.seed(DATASET_CONFIG['seed'])
    print(f"Random seed: {DATASET_CONFIG['seed']}")
    
    # Step 1: Select 10,000 images (paper doesn't specify which)
    if len(all_images) < 10000:
        raise ValueError(f"Not enough images! Found {len(all_images)}, need 10,000")
    
    selected_10k = np.random.choice(all_images, size=10000, replace=False)
    print(f"\n✓ Selected 10,000 images from {len(all_images)} total")
    
    # Step 2: Split into 2,000 labeled and 8,000 unlabeled
    labeled_2k = selected_10k[:2000]
    unlabeled_8k = selected_10k[2000:]
    
    print(f"✓ Labeled: {len(labeled_2k)} images")
    print(f"✓ Unlabeled: {len(unlabeled_8k)} images")
    
    # Step 3: Split labeled into train/val/test (70-10-20)
    n_train = int(0.70 * 2000)  # 1,400
    n_val = int(0.10 * 2000)    # 200
    n_test = 2000 - n_train - n_val  # 400 (remaining)
    
    train_images = labeled_2k[:n_train]
    val_images = labeled_2k[n_train:n_train+n_val]
    test_images = labeled_2k[n_train+n_val:]
    
    print(f"\n✓ Train: {len(train_images)} (70%)")
    print(f"✓ Val: {len(val_images)} (10%)")
    print(f"✓ Test: {len(test_images)} (20%)")
    
    # Verify split
    assert len(train_images) == 1400, f"Train should be 1400, got {len(train_images)}"
    assert len(val_images) == 200, f"Val should be 200, got {len(val_images)}"
    assert len(test_images) == 400, f"Test should be 400, got {len(test_images)}"
    assert len(unlabeled_8k) == 8000, f"Unlabeled should be 8000, got {len(unlabeled_8k)}"
    
    # Save splits
    splits = {
        'metadata': {
            'total_images_sku110k': int(len(all_images)),
            'selected_images': 10000,
            'labeled': 2000,
            'unlabeled': 8000,
            'seed': int(DATASET_CONFIG['seed']),
            'split_ratios': {
                'train': 0.70,
                'val': 0.10,
                'test': 0.20
            }
        },
        'train': train_images.tolist(),
        'val': val_images.tolist(),
        'test': test_images.tolist(),
        'unlabeled': unlabeled_8k.tolist()
    }
    
    output_path = DATA_ROOT / "splits" / "reproduction_splits.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Splits saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SPLIT SUMMARY")
    print("="*80)
    print(f"Train:     {len(train_images):,} images (labeled)")
    print(f"Val:       {len(val_images):,} images (labeled)")
    print(f"Test:      {len(test_images):,} images (labeled)")
    print(f"Unlabeled: {len(unlabeled_8k):,} images")
    print(f"Total:     {len(train_images) + len(val_images) + len(test_images) + len(unlabeled_8k):,} images")
    print("="*80)
    
    return splits

if __name__ == "__main__":
    splits = create_exact_splits()
    if splits:
        print("\n✓ Phase 1.2 COMPLETE: Dataset splits created")
        print("\nNext steps:")
        print("  python core/dataset/coco_converter.py")
        print("  python core/dataset/yolo_converter.py")
