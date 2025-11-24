"""
Phase 2: Train Baseline Models (Faster R-CNN + YOLOv3)

This script trains both baseline models on 1,400 labeled images:
1. Faster R-CNN with ResNet50-FPN (Detectron2)
2. YOLOv3 with Darknet53 (Ultralytics)

Expected baseline performance (before co-training):
- mAP@0.50: ~0.45-0.50
- AP@0.75: ~0.35-0.40
"""

import sys
import json
from pathlib import Path
import torch

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.faster_rcnn_exact import FasterRCNNExact
from core.models.yolo_exact import YOLOv3Exact

def check_dependencies():
    """Check if required packages are installed."""
    print("="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    deps = {
        'torch': None,
        'detectron2': None,
        'ultralytics': None
    }
    
    # PyTorch
    try:
        import torch
        deps['torch'] = torch.__version__
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
        deps['torch'] = None
    
    # Detectron2
    try:
        import detectron2
        deps['detectron2'] = detectron2.__version__
        print(f"✓ Detectron2: {detectron2.__version__}")
    except ImportError:
        print("✗ Detectron2 not installed")
        print("  Install: pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html")
        deps['detectron2'] = None
    
    # Ultralytics
    try:
        import ultralytics
        deps['ultralytics'] = ultralytics.__version__
        print(f"✓ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics not installed")
        print("  Install: pip install ultralytics")
        deps['ultralytics'] = None
    
    print("="*80)
    
    return deps

def train_faster_rcnn():
    """Train Faster R-CNN baseline."""
    print("\n" + "="*80)
    print("PHASE 2.1: FASTER R-CNN BASELINE TRAINING")
    print("="*80)
    
    model = FasterRCNNExact()
    
    # Train
    trainer = model.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = model.evaluate(split='val')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(split='test')
    
    # Save results
    results = {
        'model': 'Faster R-CNN + ResNet50-FPN',
        'dataset': 'SKU-110K (1,400 train images)',
        'validation': {
            'bbox': val_results.get('bbox', {})
        },
        'test': {
            'bbox': test_results.get('bbox', {})
        }
    }
    
    results_path = model.output_dir / "baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    return results

def train_yolo():
    """Train YOLOv3 baseline."""
    print("\n" + "="*80)
    print("PHASE 2.2: YOLOv3 BASELINE TRAINING")
    print("="*80)
    
    model = YOLOv3Exact()
    
    # Train (100 epochs, adjust if needed)
    train_results = model.train(
        epochs=100,
        batch_size=16,
        img_size=640
    )
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = model.evaluate(split='val')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(split='test')
    
    # Save results
    results = {
        'model': 'YOLOv3 + Darknet53',
        'dataset': 'SKU-110K (1,400 train images)',
        'validation': {
            'mAP@0.50': float(val_results.box.map50) if val_results else None,
            'mAP@0.50-0.95': float(val_results.box.map) if val_results else None,
            'mAP@0.75': float(val_results.box.map75) if val_results else None,
        },
        'test': {
            'mAP@0.50': float(test_results.box.map50) if test_results else None,
            'mAP@0.50-0.95': float(test_results.box.map) if test_results else None,
            'mAP@0.75': float(test_results.box.map75) if test_results else None,
        }
    }
    
    results_path = model.output_dir / "baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    return results

def main():
    """Main training pipeline for Phase 2."""
    print("="*80)
    print("PHASE 2: BASELINE MODEL TRAINING")
    print("="*80)
    print("\nPaper: A Co-Training Semi-Supervised Framework")
    print("Authors: Yazdanjouei et al., 2025")
    print("\nObjective: Train baseline models on 1,400 labeled images")
    print("Models: Faster R-CNN + ResNet50, YOLOv3 + Darknet53")
    print("Expected baseline mAP@0.50: ~0.45-0.50 (before co-training)")
    print("="*80)
    
    # Check dependencies
    deps = check_dependencies()
    
    if not deps['torch']:
        print("\n✗ PyTorch is required. Please install PyTorch first.")
        return
    
    # Train models
    results = {}
    
    # 1. Faster R-CNN
    if deps['detectron2']:
        try:
            results['faster_rcnn'] = train_faster_rcnn()
        except Exception as e:
            print(f"\n✗ Faster R-CNN training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⊗ Skipping Faster R-CNN (detectron2 not installed)")
    
    # 2. YOLOv3
    if deps['ultralytics']:
        try:
            results['yolov3'] = train_yolo()
        except Exception as e:
            print(f"\n✗ YOLOv3 training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⊗ Skipping YOLOv3 (ultralytics not installed)")
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 2 COMPLETE: BASELINE MODELS TRAINED")
    print("="*80)
    
    if 'faster_rcnn' in results:
        print("\n✓ Faster R-CNN:")
        test_bbox = results['faster_rcnn']['test']['bbox']
        if test_bbox:
            print(f"  Test mAP@0.50: {test_bbox.get('AP', 'N/A')}")
            print(f"  Test mAP@0.75: {test_bbox.get('AP75', 'N/A')}")
    
    if 'yolov3' in results:
        print("\n✓ YOLOv3:")
        test = results['yolov3']['test']
        print(f"  Test mAP@0.50: {test.get('mAP@0.50', 'N/A'):.4f}")
        print(f"  Test mAP@0.75: {test.get('mAP@0.75', 'N/A'):.4f}")
        print(f"  Test mAP@0.50-0.95: {test.get('mAP@0.50-0.95', 'N/A'):.4f}")
    
    # Save combined results
    summary_path = PROJECT_ROOT / "data" / "SKU110K" / "checkpoints" / "baseline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_path}")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("Phase 3: Train ensemble classifier (XGBoost + RF + SVM)")
    print("Phase 4: Implement co-training framework")
    print("Phase 5: Hyperparameter optimization")
    print("Phase 6: Final evaluation")
    print("\nSee: PHASES_2_TO_6_IMPLEMENTATION.md")
    print("="*80)

if __name__ == "__main__":
    main()
