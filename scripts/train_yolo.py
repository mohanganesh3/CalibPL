"""
Train YOLOv3 baseline on 1,400 labeled images.
Based on paper Section 3.2.2.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.yolo_exact import YOLOv3Exact

def main():
    import os
    # Force CPU - K80's cuDNN incompatible (compute capability 3.7 < 5.0 required)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['YOLO_DISABLE_AMP'] = '1'
    
    print("="*80)
    print("TRAINING: YOLOv3 + Darknet53")
    print("="*80)
    print("Dataset: 1,400 labeled images from SKU-110K")
    print("Device: CPU (K80 cuDNN incompatible)")
    print("Epochs: 100")
    print("Batch size: 12")
    print("Expected time: 40-60 hours on CPU")
    print("Expected baseline mAP@0.50: ~0.45-0.50")
    print("="*80)
    
    # Initialize model
    model = YOLOv3Exact()
    
    # Train (batch_size adjusted for Tesla K80)
    print("\nStarting training...")
    train_results = model.train(
        epochs=100,
        batch_size=12,  # Adjusted for K80 (paper likely used 16 on RTX 4080)
        img_size=640
    )
    
    # Evaluate on validation
    print("\n" + "="*80)
    print("VALIDATION EVALUATION")
    print("="*80)
    val_results = model.evaluate(split='val')
    
    # Evaluate on test
    print("\n" + "="*80)
    print("TEST EVALUATION")
    print("="*80)
    test_results = model.evaluate(split='test')
    
    # Save results
    import json
    results = {
        'model': 'YOLOv3 + Darknet53',
        'training_samples': 1400,
        'epochs': 100,
        'validation': {
            'mAP@0.50': float(val_results.box.map50),
            'mAP@0.50-0.95': float(val_results.box.map),
            'mAP@0.75': float(val_results.box.map75),
        } if val_results else None,
        'test': {
            'mAP@0.50': float(test_results.box.map50),
            'mAP@0.50-0.95': float(test_results.box.map),
            'mAP@0.75': float(test_results.box.map75),
        } if test_results else None
    }
    
    results_file = model.output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ YOLOv3 TRAINING COMPLETE")
    print("="*80)
    print(f"Checkpoints: {model.output_dir}")
    print(f"Results: {results_file}")
    
    if test_results:
        print(f"Test mAP@0.50: {test_results.box.map50:.4f}")
        print(f"Test mAP@0.75: {test_results.box.map75:.4f}")
        print(f"Test mAP@0.50-0.95: {test_results.box.map:.4f}")
    
    print("="*80)

if __name__ == "__main__":
    main()
