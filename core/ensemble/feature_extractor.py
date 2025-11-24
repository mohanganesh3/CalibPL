"""
Feature extraction from trained Faster R-CNN and YOLO models.
Implements Section 3.2 of the paper.

Extracts:
- Faster R-CNN: Features from ResNet50 backbone (before detection head)
- YOLO: Features from Darknet53 backbone (before detection head)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from trained detection models.
    Paper Section 3.2: "Feature extraction is performed using the backbone networks"
    """
    
    def __init__(
        self,
        frcnn_model_path: str = None,
        yolo_model_path: str = None,
        device: str = 'cuda'
    ):
        """
        Initialize feature extractor with trained models.
        
        Args:
            frcnn_model_path: Path to trained Faster R-CNN checkpoint
            yolo_model_path: Path to trained YOLO checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.frcnn_model = None
        self.yolo_model = None
        
        if frcnn_model_path:
            self.load_frcnn(frcnn_model_path)
        if yolo_model_path:
            self.load_yolo(yolo_model_path)
    
    def load_frcnn(self, model_path: str):
        """Load trained Faster R-CNN model for feature extraction."""
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.modeling import build_model
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        
        logger.info(f"Loading Faster R-CNN from {model_path}")
        
        # Setup config
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = self.device
        
        # Build model
        self.frcnn_model = build_model(cfg)
        self.frcnn_model.eval()
        
        # Store config for later use
        self.frcnn_cfg = cfg
        
        logger.info("Faster R-CNN loaded successfully")
    
    def load_yolo(self, model_path: str):
        """Load trained YOLO model for feature extraction."""
        from ultralytics import YOLO
        
        logger.info(f"Loading YOLO from {model_path}")
        
        # Load YOLO model
        self.yolo_model = YOLO(model_path)
        self.yolo_model.to(self.device)
        
        logger.info("YOLO loaded successfully")
    
    def extract_frcnn_features(
        self,
        images: List[str],
        batch_size: int = 4
    ) -> np.ndarray:
        """
        Extract features from Faster R-CNN backbone (ResNet50-FPN).
        
        Paper: "Faster R-CNN uses a ResNet50 backbone for feature extraction"
        Features are extracted from the FPN output (before detection heads).
        
        Args:
            images: List of image paths
            batch_size: Batch size for feature extraction
            
        Returns:
            features: numpy array of shape (N, feature_dim)
        """
        import cv2
        from detectron2.data import transforms as T
        
        if self.frcnn_model is None:
            raise ValueError("Faster R-CNN model not loaded")
        
        logger.info(f"Extracting Faster R-CNN features from {len(images)} images")
        
        all_features = []
        
        # Prepare image transformation
        aug = T.ResizeShortestEdge(
            [self.frcnn_cfg.INPUT.MIN_SIZE_TEST, self.frcnn_cfg.INPUT.MIN_SIZE_TEST],
            self.frcnn_cfg.INPUT.MAX_SIZE_TEST
        )
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="FRCNN features"):
                batch_imgs = images[i:i + batch_size]
                batch_features = []
                
                for img_path in batch_imgs:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Apply augmentation
                    aug_input = T.AugInput(img)
                    transforms = aug(aug_input)
                    img = aug_input.image
                    
                    # Convert to tensor
                    img = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
                    img = img.to(self.device)
                    
                    # Create input dict
                    inputs = {"image": img, "height": img.shape[1], "width": img.shape[2]}
                    
                    # Extract features from backbone
                    # Get features from the FPN backbone (before detection heads)
                    images_tensor = self.frcnn_model.preprocess_image([inputs])
                    features = self.frcnn_model.backbone(images_tensor.tensor)
                    
                    # Pool features from all FPN levels
                    # Paper uses global average pooling
                    pooled_features = []
                    for key in sorted(features.keys()):
                        feat = features[key]
                        # Global average pooling
                        pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                        pooled = pooled.flatten(1)
                        pooled_features.append(pooled)
                    
                    # Concatenate features from all FPN levels
                    feature_vector = torch.cat(pooled_features, dim=1)
                    batch_features.append(feature_vector.cpu().numpy())
                
                all_features.extend(batch_features)
        
        # Convert to numpy array
        features_array = np.vstack(all_features)
        
        logger.info(f"Extracted features shape: {features_array.shape}")
        return features_array
    
    def extract_yolo_features(
        self,
        images: List[str],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Extract features from YOLO backbone (Darknet53).
        
        Paper: "YOLO uses a Darknet53 backbone for feature extraction"
        Features are extracted from the last convolutional layer (before detection heads).
        
        Args:
            images: List of image paths
            batch_size: Batch size for feature extraction
            
        Returns:
            features: numpy array of shape (N, feature_dim)
        """
        import cv2
        
        if self.yolo_model is None:
            raise ValueError("YOLO model not loaded")
        
        logger.info(f"Extracting YOLO features from {len(images)} images")
        
        all_features = []
        
        # Hook to extract features from backbone
        features_hook = []
        
        def hook_fn(module, input, output):
            # Global average pooling on spatial dimensions
            pooled = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
            features_hook.append(pooled.flatten(1))
        
        # Register hook on the last backbone layer
        # For YOLOv8/v11, this is typically model.model[-1] or model.model.model[-1]
        try:
            # Try YOLOv8/v11 structure
            hook_layer = self.yolo_model.model.model[-1]
        except:
            # Fallback to alternative structure
            hook_layer = self.yolo_model.model[-1]
        
        handle = hook_layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="YOLO features"):
                batch_imgs = images[i:i + batch_size]
                features_hook.clear()
                
                # YOLO inference (features will be captured by hook)
                _ = self.yolo_model.predict(
                    batch_imgs,
                    verbose=False,
                    device=self.device
                )
                
                # Collect features
                if features_hook:
                    batch_features = torch.cat(features_hook, dim=0)
                    all_features.append(batch_features.cpu().numpy())
        
        # Remove hook
        handle.remove()
        
        # Convert to numpy array
        features_array = np.vstack(all_features)
        
        logger.info(f"Extracted features shape: {features_array.shape}")
        return features_array
    
    def extract_both_features(
        self,
        images: List[str],
        frcnn_batch_size: int = 4,
        yolo_batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from both models.
        
        Returns:
            frcnn_features: Features from Faster R-CNN (N, frcnn_dim)
            yolo_features: Features from YOLO (N, yolo_dim)
        """
        frcnn_features = self.extract_frcnn_features(images, frcnn_batch_size)
        yolo_features = self.extract_yolo_features(images, yolo_batch_size)
        
        return frcnn_features, yolo_features
    
    def save_features(
        self,
        frcnn_features: np.ndarray,
        yolo_features: np.ndarray,
        labels: np.ndarray,
        output_dir: str
    ):
        """Save extracted features to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'frcnn_features.npy', frcnn_features)
        np.save(output_dir / 'yolo_features.npy', yolo_features)
        np.save(output_dir / 'labels.npy', labels)
        
        logger.info(f"Features saved to {output_dir}")
    
    @staticmethod
    def load_features(features_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load saved features from disk."""
        features_dir = Path(features_dir)
        
        frcnn_features = np.load(features_dir / 'frcnn_features.npy')
        yolo_features = np.load(features_dir / 'yolo_features.npy')
        labels = np.load(features_dir / 'labels.npy')
        
        logger.info(f"Features loaded from {features_dir}")
        logger.info(f"FRCNN features: {frcnn_features.shape}")
        logger.info(f"YOLO features: {yolo_features.shape}")
        logger.info(f"Labels: {labels.shape}")
        
        return frcnn_features, yolo_features, labels
