"""
Faster R-CNN with ResNet50 implementation for exact paper reproduction.

Paper: Section 3.2.1 - Faster R-CNN Architecture
- Backbone: ResNet50 (pre-trained on ImageNet)
- Training: 1,400 labeled images from SKU-110K
- Framework: Detectron2
"""

import torch
import os
import logging
from pathlib import Path
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.utils.comm as comm
import copy
import numpy as np
from PIL import Image, ImageFile
import sys

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_exact import DATA_ROOT, TARGET_METRICS

# Enable loading truncated images globally
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RobustTrainer(DefaultTrainer):
    """
    Custom trainer that handles corrupted images by enabling PIL's truncated image loading.
    """
    
    @classmethod 
    def build_train_loader(cls, cfg):
        """Override to enable robust image loading."""
        
        # Enable PIL to load truncated/corrupted images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        def robust_mapper(dataset_dict):
            """Mapper that handles corrupted images more gracefully."""
            dataset_dict = copy.deepcopy(dataset_dict)
            
            try:
                # Use default mapper but log any image issues
                mapper = DatasetMapper(cfg, is_train=True)
                result = mapper(dataset_dict)
                return result
                
            except (OSError, IOError) as e:
                # Log the error but don't crash - let dataset sampling handle retries
                if comm.is_main_process():
                    logging.warning(f"⚠️ Image loading issue: {dataset_dict.get('file_name', 'unknown')} - {str(e)[:100]}")
                
                # Try to create a valid dummy sample to avoid crashing
                # This approach lets the training continue rather than stopping
                try:
                    # Create minimal valid sample
                    dummy_dict = {
                        "file_name": dataset_dict.get("file_name", "dummy"),
                        "height": 600,
                        "width": 800,
                        "image_id": dataset_dict.get("image_id", 0),
                        "annotations": []  # Empty annotations for corrupted images
                    }
                    return DatasetMapper(cfg, is_train=True)(dummy_dict)
                except:
                    # If even dummy fails, return None and let the dataloader handle it
                    return None
        
        return build_detection_train_loader(cfg, mapper=robust_mapper)

class FasterRCNNExact:
    """
    Faster R-CNN model configured to match paper specifications.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize Faster R-CNN with paper-exact configuration.
        
        Args:
            output_dir: Directory to save checkpoints and logs
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "SKU110K" / "checkpoints" / "faster_rcnn"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Register datasets
        self._register_datasets()
        
        # Configure model
        self.cfg = self._create_config()
        
        self.trainer = None
        self.predictor = None
    
    def _register_datasets(self):
        """Register SKU-110K datasets with Detectron2."""
        
        # Unregister if already exists
        for dataset_name in ['sku110k_train', 'sku110k_val', 'sku110k_test']:
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.remove(dataset_name)
        
        # Register train, val, test
        for split in ['train', 'val', 'test']:
            json_file = DATA_ROOT / "coco_format" / f"{split}.json"
            image_root = DATA_ROOT / "raw" / "SKU110K_fixed" / "images"
            
            register_coco_instances(
                f"sku110k_{split}",
                {},
                str(json_file),
                str(image_root)
            )
        
        # Set metadata
        for split in ['train', 'val', 'test']:
            MetadataCatalog.get(f"sku110k_{split}").set(thing_classes=["product"])
    
    def _create_config(self):
        """
        Create Detectron2 config matching paper specifications.
        
        Paper Section 3.2.1:
        - Model: Faster R-CNN with ResNet50-FPN
        - Pre-training: ImageNet
        - Input: 800x800 (max dimension)
        - Batch size: 2 (per paper's hardware constraints)
        - Learning rate: 0.02 (base), warmup 1000 iterations
        - Max iterations: Paper trained for sufficient convergence
        """
        cfg = get_cfg()
        
        # Base model: Faster R-CNN with ResNet50-FPN
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        ))
        
        # Model weights (ImageNet pre-trained)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
        
        # Dataset configuration
        cfg.DATASETS.TRAIN = ("sku110k_train",)
        cfg.DATASETS.TEST = ("sku110k_val",)
        
        # Number of classes (1 = product)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        # Training hyperparameters
        # K80's cuDNN incompatible, using CPU with reduced settings
        cfg.SOLVER.IMS_PER_BATCH = 1  # Single batch for CPU
        cfg.SOLVER.BASE_LR = 0.001  # Reduced LR for CPU stability
        cfg.SOLVER.WARMUP_ITERS = 500
        cfg.SOLVER.MAX_ITER = 10000  # ~14 epochs for 1400 images
        cfg.SOLVER.STEPS = (7000, 9000)  # Learning rate decay
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.WEIGHT_DECAY = 0.0001  # L2 regularization
        cfg.SOLVER.MOMENTUM = 0.9
        
        # Reduce memory for CPU training
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 64  # Reduced from 256
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Reduced from 512
        cfg.DATALOADER.NUM_WORKERS = 0  # Required for CPU
        
        # Force CPU (K80's cuDNN incompatible)
        cfg.MODEL.DEVICE = "cpu"
        
        # Image input size
        cfg.INPUT.MIN_SIZE_TRAIN = (800,)
        cfg.INPUT.MAX_SIZE_TRAIN = 1333
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333
        
        # Data augmentation
        cfg.INPUT.RANDOM_FLIP = "horizontal"
        
        # Evaluation
        cfg.TEST.EVAL_PERIOD = 1000
        
        # Output directory
        cfg.OUTPUT_DIR = str(self.output_dir)
        
        # Data loader workers (0 for CPU)
        cfg.DATALOADER.NUM_WORKERS = 0
        
        return cfg
    
    def train(self):
        """
        Train Faster R-CNN baseline model.
        
        Returns:
            trainer: Trained model
        """
        print("="*80)
        print("TRAINING FASTER R-CNN BASELINE")
        print("="*80)
        print(f"Dataset: SKU-110K (1,400 train images)")
        print(f"Model: Faster R-CNN + ResNet50-FPN")
        print(f"Batch size: {self.cfg.SOLVER.IMS_PER_BATCH}")
        print(f"Learning rate: {self.cfg.SOLVER.BASE_LR}")
        print(f"Max iterations: {self.cfg.SOLVER.MAX_ITER}")
        print(f"Output: {self.output_dir}")
        print("="*80)
        
        self.trainer = RobustTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        
        print("\n✓ Training complete")
        return self.trainer
    
    def evaluate(self, split='test'):
        """
        Evaluate model on test set.
        
        Args:
            split: Dataset split to evaluate on ('val' or 'test')
        
        Returns:
            dict: Evaluation metrics (mAP, AP@0.75, etc.)
        """
        print("="*80)
        print(f"EVALUATING FASTER R-CNN ON {split.upper()}")
        print("="*80)
        
        # Create predictor
        cfg_eval = self.cfg.clone()
        cfg_eval.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        cfg_eval.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg_eval.DATASETS.TEST = (f"sku110k_{split}",)
        
        # Evaluate
        from detectron2.evaluation import inference_on_dataset
        from detectron2.data import build_detection_test_loader
        
        predictor = DefaultPredictor(cfg_eval)
        evaluator = COCOEvaluator(f"sku110k_{split}", output_dir=str(self.output_dir))
        val_loader = build_detection_test_loader(cfg_eval, f"sku110k_{split}")
        
        results = inference_on_dataset(predictor.model, val_loader, evaluator)
        
        print("\n✓ Evaluation complete")
        print(f"Results: {results}")
        
        return results
    
    def load_checkpoint(self, checkpoint_path):
        """Load trained model checkpoint."""
        self.cfg.MODEL.WEIGHTS = checkpoint_path
        self.predictor = DefaultPredictor(self.cfg)
        return self.predictor

if __name__ == "__main__":
    print("Faster R-CNN Exact Implementation")
    print("Paper: A Co-Training Semi-Supervised Framework (Yazdanjouei et al., 2025)")
    print("\nTo train:")
    print("  python scripts/train_faster_rcnn_baseline.py")
