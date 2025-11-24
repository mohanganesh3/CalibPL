"""
YOLOv3 with Darknet53 implementation for exact paper reproduction.

Paper: Section 3.2.2 - YOLO Architecture
- Backbone: Darknet53
- Training: 1,400 labeled images from SKU-110K
- Framework: Ultralytics YOLOv3 (PyTorch)
"""

import torch
from pathlib import Path
import yaml
import sys
from collections import defaultdict

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_exact import DATA_ROOT, TARGET_METRICS

class YOLOv3Exact:
    """
    YOLOv3 model configured to match paper specifications.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize YOLOv3 with paper-exact configuration.
        
        Args:
            output_dir: Directory to save checkpoints and logs
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "SKU110K" / "checkpoints" / "yolov3"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.data_yaml = self._create_data_config()
    
    def _create_data_config(self):
        """
        Create YOLO data configuration file.
        
        Returns:
            Path to data.yaml file
        """
        self._verify_split_layout()

        data_config = {
            'path': str(DATA_ROOT / "yolo_format"),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {0: 'product'},
            'nc': 1  # Number of classes
        }
        
        # Save to YAML
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        
        return yaml_path

    def _verify_split_layout(self):
        """
        Validate that each split has images with corresponding YOLO labels.

        Ultralytics will otherwise continue with an effectively empty dataset,
        which produces meaningless training logs and metrics.
        """
        split_root = DATA_ROOT / "yolo_format"
        expected = {
            'train': 1400,
            'val': 200,
            'test': 400,
        }
        summary = defaultdict(dict)

        for split, expected_count in expected.items():
            image_dir = split_root / split / "images"
            label_dir = split_root / split / "labels"

            image_files = sorted(
                list(image_dir.glob("*.jpg")) +
                list(image_dir.glob("*.jpeg")) +
                list(image_dir.glob("*.png"))
            )
            missing_labels = [
                image_path.name for image_path in image_files
                if not (label_dir / f"{image_path.stem}.txt").exists()
            ]

            summary[split]["images"] = len(image_files)
            summary[split]["missing_labels"] = len(missing_labels)

            if len(image_files) != expected_count:
                raise RuntimeError(
                    f"{split} split has {len(image_files)} images; "
                    f"expected {expected_count}. Check dataset preparation."
                )

            if missing_labels:
                preview = ", ".join(missing_labels[:5])
                raise RuntimeError(
                    f"{split} split is missing {len(missing_labels)} labels "
                    f"(examples: {preview})."
                )

        print("Dataset audit passed for YOLO training:")
        for split in ("train", "val", "test"):
            print(
                f"  {split}: {summary[split]['images']} images, "
                f"{summary[split]['missing_labels']} missing labels"
            )
    
    def train(self, epochs=100, batch_size=12, img_size=640):
        """
        Train YOLOv3 baseline model.
        
        Paper specifications (adjusted for Tesla K80):
        - Epochs: 100 (standard for YOLO)
        - Batch size: 12 (adjusted from 16 for K80, ~6 per GPU for 2 GPUs)
        - Image size: 640x640 (YOLO standard)
        - Optimizer: SGD with momentum
        - Hardware: Tesla K80 is 5-6x slower than paper's RTX 4080
        
        Args:
            epochs: Number of training epochs (default 100)
            batch_size: Batch size (default 16)
            img_size: Input image size (default 640)
        
        Returns:
            results: Training results
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("✗ Ultralytics not installed")
            print("Install with: pip install ultralytics")
            return None
        
        print("="*80)
        print("TRAINING YOLOv3 BASELINE")
        print("="*80)
        print(f"Dataset: SKU-110K (1,400 train images)")
        print(f"Model: YOLOv3 + Darknet53")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {img_size}")
        print(f"Output: {self.output_dir}")
        print("="*80)
        
        # Load YOLOv3 model (Darknet53 backbone)
        # Note: Ultralytics uses 'yolov3' which has Darknet53 backbone
        # Use absolute path to pretrained weights
        pretrained_path = PROJECT_ROOT / 'yolov3u.pt'
        if not pretrained_path.exists():
            print(f"Warning: {pretrained_path} not found, downloading...")
            self.model = YOLO('yolov3.pt')  # Will download if needed
        else:
            self.model = YOLO(str(pretrained_path))  # Pre-trained weights
        
        # Force CPU - K80's cuDNN incompatible
        import os
        device = 'cpu'
        os.environ['YOLO_DISABLE_AMP'] = '1'
        
        results = self.model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=str(self.output_dir),
            name='baseline',
            pretrained=True,  # Use ImageNet pre-trained weights
            optimizer='SGD',
            lr0=0.01,  # Initial learning rate
            momentum=0.937,  # SGD momentum
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=50,  # Early stopping patience
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            device=device,  # Use specified GPUs
            workers=8,
            verbose=True,
            amp=False  # Disable AMP to avoid compatibility issues
        )
        
        print("\n✓ Training complete")
        print(f"Best weights: {self.output_dir}/baseline/weights/best.pt")
        
        return results
    
    def evaluate(self, weights_path=None, split='test'):
        """
        Evaluate model on test set.
        
        Args:
            weights_path: Path to model weights (default: best.pt from training)
            split: Dataset split to evaluate ('val' or 'test')
        
        Returns:
            dict: Evaluation metrics (mAP, AP@0.75, etc.)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("✗ Ultralytics not installed")
            return None
        
        print("="*80)
        print(f"EVALUATING YOLOv3 ON {split.upper()}")
        print("="*80)
        
        # Load trained model
        if weights_path is None:
            weights_path = self.output_dir / "baseline" / "weights" / "best.pt"
        
        if not Path(weights_path).exists():
            print(f"✗ Weights not found: {weights_path}")
            print("Please train the model first or provide valid weights path")
            return None
        
        self.model = YOLO(str(weights_path))
        
        # Update data config for test split
        if split == 'test':
            # Create temporary config for test eval
            test_config = {
                'path': str(DATA_ROOT / "yolo_format"),
                'train': 'train/images',
                'val': 'test/images',  # Use test set for validation
                'test': 'test/images',
                'names': {0: 'product'},
                'nc': 1
            }
            test_yaml = self.output_dir / "data_test.yaml"
            with open(test_yaml, 'w') as f:
                yaml.dump(test_config, f)
            data_path = test_yaml
        else:
            data_path = self.data_yaml
        
        # Evaluate
        results = self.model.val(
            data=str(data_path),
            batch=16,
            imgsz=640,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        
        print("\n✓ Evaluation complete")
        print(f"mAP@0.50: {results.box.map50:.4f}")
        print(f"mAP@0.50-0.95: {results.box.map:.4f}")
        print(f"mAP@0.75: {results.box.map75:.4f}")
        
        return results
    
    def predict(self, image_path, weights_path=None, conf_threshold=0.5):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            weights_path: Path to model weights
            conf_threshold: Confidence threshold for detections
        
        Returns:
            results: Detection results
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("✗ Ultralytics not installed")
            return None
        
        # Load model
        if weights_path is None:
            weights_path = self.output_dir / "baseline" / "weights" / "best.pt"
        
        self.model = YOLO(str(weights_path))
        
        # Predict
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        return results

if __name__ == "__main__":
    print("YOLOv3 Exact Implementation")
    print("Paper: A Co-Training Semi-Supervised Framework (Yazdanjouei et al., 2025)")
    print("\nTo train:")
    print("  python scripts/train_yolo_baseline.py")
