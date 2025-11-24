"""
Exact configuration for paper reproduction.
All values either from paper or documented assumptions.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
DATA_ROOT = PROJECT_ROOT / "data" / "SKU110K"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "reproduction"

# Dataset configuration (From paper Section 3.1)
DATASET_CONFIG = {
    'total_images': 11762,           # SKU-110K full dataset
    'selected_images': 10000,        # Paper uses 10,000
    'labeled_images': 2000,          # Paper: 2,000 labeled
    'unlabeled_images': 8000,        # Paper: 8,000 unlabeled
    
    # Split ratios (From paper: 70-10-20)
    'train_ratio': 0.70,             # 1,400 images
    'val_ratio': 0.10,               # 200 images
    'test_ratio': 0.20,              # 400 images
    
    # Random seed for reproducibility
    'seed': 42,                      # CRITICAL: Fixed for exact reproduction
}

# Target metrics (From paper Section 4)
TARGET_METRICS = {
    'mAP': 0.596,
    'AP_75': 0.663,
    'AR_300': 0.627,
    'tolerance': 0.005,  # ±0.5% acceptable for reproduction
}

def create_directories():
    """Create all necessary directories."""
    dirs = [
        DATA_ROOT / "raw",
        DATA_ROOT / "splits",
        DATA_ROOT / "coco_format",
        DATA_ROOT / "yolo_format" / "train" / "images",
        DATA_ROOT / "yolo_format" / "train" / "labels",
        DATA_ROOT / "yolo_format" / "val" / "images",
        DATA_ROOT / "yolo_format" / "val" / "labels",
        DATA_ROOT / "yolo_format" / "test" / "images",
        DATA_ROOT / "yolo_format" / "test" / "labels",
        EXPERIMENTS_ROOT / "logs",
        EXPERIMENTS_ROOT / "checkpoints",
        EXPERIMENTS_ROOT / "results",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✓ All directories created")

# Complete configuration dictionary for reproduction pipeline
CONFIG = {
    'data': {
        'sku110k_root': str(DATA_ROOT),
        'labeled_images': DATASET_CONFIG['labeled_images'],
        'unlabeled_images': DATASET_CONFIG['unlabeled_images'],
        'train_ratio': DATASET_CONFIG['train_ratio'],
        'val_ratio': DATASET_CONFIG['val_ratio'],
        'test_ratio': DATASET_CONFIG['test_ratio'],
    },
    'hardware': {
        'device': 'cpu',  # Forced CPU (K80 not supported by PyTorch 2.x)
        'num_gpus': 0,
    },
    'training': {
        'frcnn': {
            'batch_size': 2,
            'epochs': 100,
            'learning_rate': 0.001,
        },
        'yolo': {
            'batch_size': 12,
            'epochs': 100,
            'learning_rate': 0.001,
        },
    },
    'cotraining': {
        'confidence_threshold': 0.7,
        'num_iterations': 5,
    },
    'optimization': {
        'num_iterations': 50,
    },
    'random_seed': DATASET_CONFIG['seed'],
    'target_metrics': TARGET_METRICS,
}

if __name__ == "__main__":
    create_directories()
    print("\n✓ Configuration loaded successfully!")
    print(f"Dataset: {DATASET_CONFIG['labeled_images']} labeled + {DATASET_CONFIG['unlabeled_images']} unlabeled")
    print(f"Target mAP: {TARGET_METRICS['mAP']}")
