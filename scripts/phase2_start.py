"""
Quick start script for Phase 2 training.

This script provides commands to train baseline models separately.
Training can take 4-8 hours per model on RTX 4080.
"""

print("="*80)
print("PHASE 2: BASELINE MODEL TRAINING")
print("="*80)
print()
print("Two models need to be trained on 1,400 labeled images:")
print()
print("1. Faster R-CNN + ResNet50 (Detectron2)")
print("   Training time: ~4-6 hours on RTX 4080")
print("   Expected baseline mAP@0.50: ~0.45-0.50")
print()
print("2. YOLOv3 + Darknet53 (Ultralytics)")
print("   Training time: ~3-5 hours on RTX 4080 (100 epochs)")
print("   Expected baseline mAP@0.50: ~0.45-0.50")
print()
print("="*80)
print("TRAINING COMMANDS")
print("="*80)
print()
print("Option 1: Train both models sequentially")
print("  python scripts/train_baseline.py")
print()
print("Option 2: Train individually")
print("  python scripts/train_faster_rcnn.py")
print("  python scripts/train_yolo.py")
print()
print("Option 3: Run in background")
print("  nohup python scripts/train_baseline.py > logs/phase2_training.log 2>&1 &")
print()
print("="*80)
print()

# Check dependencies
print("Checking dependencies...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import detectron2
    print(f"✓ Detectron2 {detectron2.__version__}")
except ImportError:
    print("✗ Detectron2 not installed")
    print("  Install: pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'")

try:
    import ultralytics
    print(f"✓ Ultralytics {ultralytics.__version__}")
except ImportError:
    print("✗ Ultralytics not installed")
    print("  Install: pip install ultralytics")

print()
print("="*80)
print("DATASET STATUS")
print("="*80)

from pathlib import Path
data_root = Path("/home/mohanganesh/retail-shelf-detection/data/SKU110K")

if (data_root / "coco_format" / "train.json").exists():
    print("✓ COCO format ready (Faster R-CNN)")
else:
    print("✗ COCO format missing")

if (data_root / "yolo_format" / "train.txt").exists():
    print("✓ YOLO format ready (YOLOv3)")
else:
    print("✗ YOLO format missing")

print()
print("Ready to train! Choose an option above to begin.")
