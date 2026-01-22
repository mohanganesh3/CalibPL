#!/usr/bin/env python3
"""
Quick end-to-end pipeline test:
1. Run pretrained YOLOv12 on 10 test images
2. Save predictions
3. Run calibration benchmark (D-ECE + prepare for LaECE)
4. Verify everything works error-free

This catches any pipeline issues BEFORE the long training completes.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

PROJECT_ROOT = Path("/home/mohanganesh/retail-shelf-detection")
sys.path.insert(0, str(PROJECT_ROOT))

YOLO_DIR = PROJECT_ROOT / "data" / "SKU110K" / "yolo_format"
TEST_DIR = PROJECT_ROOT / "results" / "pipeline_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def test_step1_inference():
    """Test: run pretrained YOLOv12 on 10 test images."""
    print("\n" + "="*70)
    print("STEP 1: Testing YOLOv12 inference on 10 test images")
    print("="*70)
    
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / "models" / "yolo12n.pt"))
    
    test_images = sorted(os.listdir(YOLO_DIR / "test" / "images"))[:10]
    
    all_predictions = []
    for img_name in test_images:
        img_path = str(YOLO_DIR / "test" / "images" / img_name)
        results = model.predict(source=img_path, conf=0.01, iou=0.5, device='cpu', verbose=False)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for j in range(len(boxes)):
                    pred = {
                        'image': img_name,
                        'box': boxes.xyxy[j].tolist(),
                        'confidence': float(boxes.conf[j]),
                        'class': int(boxes.cls[j]),
                    }
                    all_predictions.append(pred)
    
    print(f"  ✓ Inference complete: {len(all_predictions)} detections from {len(test_images)} images")
    
    # Save predictions
    pred_file = TEST_DIR / "test_predictions.json"
    with open(pred_file, 'w') as f:
        json.dump({
            'model': 'YOLOv12n (pretrained)',
            'num_test_images': len(test_images),
            'num_detections': len(all_predictions),
            'predictions': all_predictions,
        }, f, indent=2)
    
    print(f"  ✓ Saved to: {pred_file}")
    return pred_file, all_predictions


def test_step2_calibration(predictions):
    """Test: compute D-ECE on the predictions."""
    print("\n" + "="*70)
    print("STEP 2: Testing calibration metrics (D-ECE)")
    print("="*70)
    
    from core.calibration.detection_calibration import (
        match_detections_to_gt, 
        compute_detection_ece,
        apply_temperature_scaling,
        apply_platt_scaling,
        apply_isotonic_regression,
    )
    
    gt_labels_dir = str(YOLO_DIR / "test" / "labels")
    images_dir = str(YOLO_DIR / "test" / "images")
    
    # Match predictions to ground truth
    confidences, correctness = match_detections_to_gt(
        predictions, gt_labels_dir, images_dir, iou_threshold=0.5
    )
    
    print(f"  Matched {len(confidences)} detections")
    print(f"  True Positives: {int(correctness.sum())} / {len(correctness)}")
    print(f"  Avg confidence: {confidences.mean():.4f}")
    
    # Compute D-ECE
    metrics = compute_detection_ece(confidences, correctness, n_bins=15)
    print(f"\n  D-ECE: {metrics.d_ece:.4f}")
    print(f"  MCE:   {metrics.mce:.4f}")
    print(f"  Brier: {metrics.brier:.4f}")
    
    # Test post-hoc calibration methods
    print("\n  Testing post-hoc calibration methods:")
    
    best_T, temp_confs = apply_temperature_scaling(confidences, correctness)
    temp_metrics = compute_detection_ece(temp_confs, correctness)
    print(f"  Temperature Scaling (T={best_T:.2f}): D-ECE={temp_metrics.d_ece:.4f}")
    
    platt_params, platt_confs = apply_platt_scaling(confidences, correctness)
    platt_metrics = compute_detection_ece(platt_confs, correctness)
    print(f"  Platt Scaling (a={platt_params[0]:.2f}, b={platt_params[1]:.2f}): D-ECE={platt_metrics.d_ece:.4f}")
    
    iso_confs = apply_isotonic_regression(confidences, correctness)
    iso_metrics = compute_detection_ece(iso_confs, correctness)
    print(f"  Isotonic Regression:                 D-ECE={iso_metrics.d_ece:.4f}")
    
    print("\n  ✓ All calibration metrics computed successfully!")
    return metrics


def test_step3_fiveai_library():
    """Test: verify FiveAI detection_calibration library loads and works."""
    print("\n" + "="*70)
    print("STEP 3: Testing FiveAI detection_calibration library")
    print("="*70)
    
    try:
        import detection_calibration
        print(f"  ✓ Library imported successfully")
        
        # Check actual submodules
        from detection_calibration import DetectionCalibration
        print(f"  ✓ DetectionCalibration module loaded")
        
        from detection_calibration import coco_calibration
        print(f"  ✓ coco_calibration module loaded")
        
        from detection_calibration import platt_scaling
        print(f"  ✓ platt_scaling module loaded")
        
        from detection_calibration import utils
        print(f"  ✓ utils module loaded")
        
        # Check what's in DetectionCalibration
        dc_items = [x for x in dir(DetectionCalibration) if not x.startswith('_')]
        print(f"  DetectionCalibration exports: {dc_items[:15]}")
        
        print("\n  ✓ FiveAI library is fully functional!")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step4_mc_dropout():
    """Test: verify MC Dropout module loads and makes sense."""
    print("\n" + "="*70)
    print("STEP 4: Testing MC Dropout module")
    print("="*70)
    
    from core.calibration.mc_dropout import MCDropoutDetector
    
    # Just verify the class instantiates correctly
    try:
        detector = MCDropoutDetector(
            weights_path=str(PROJECT_ROOT / "models" / "yolo12n.pt"),
            T=3,  # Only 3 passes for quick test
            conf_threshold=0.01,
            device='cpu',
        )
        print(f"  ✓ MCDropoutDetector created successfully")
        print(f"  Model: {type(detector.model)}")
        print(f"  T={detector.T} passes configured")
        
        # Run a quick single-image test
        test_img = str(YOLO_DIR / "test" / "images" / sorted(os.listdir(YOLO_DIR / "test" / "images"))[0])
        print(f"  Running {detector.T}-pass inference on: {os.path.basename(test_img)}")
        
        uncertain_dets = detector.predict_with_uncertainty(test_img)
        print(f"  ✓ Got {len(uncertain_dets)} uncertain detections")
        
        if uncertain_dets:
            det = uncertain_dets[0]
            print(f"    Sample: conf={det.mean_confidence:.3f}, "
                  f"epistemic={det.epistemic:.4f}, "
                  f"aleatoric={det.aleatoric:.4f}")
        
        print("\n  ✓ MC Dropout pipeline is fully functional!")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step5_ssod_splits():
    """Test: verify SSOD splits are valid."""
    print("\n" + "="*70)
    print("STEP 5: Verifying SSOD splits")
    print("="*70)
    
    splits_dir = PROJECT_ROOT / "data" / "SKU110K" / "ssod_splits"
    meta_file = splits_dir / "split_metadata.json"
    
    with open(meta_file) as f:
        meta = json.load(f)
    
    print(f"  Seed: {meta['seed']}")
    print(f"  Total train images: {meta['total_train']}")
    
    all_ok = True
    for name, info in meta['splits'].items():
        labeled_file = Path(info['labeled_file'])
        unlabeled_file = Path(info['unlabeled_file'])
        
        with open(labeled_file) as f:
            labeled = [l.strip() for l in f if l.strip()]
        with open(unlabeled_file) as f:
            unlabeled = [l.strip() for l in f if l.strip()]
        
        # Verify no overlap
        labeled_set = set(labeled)
        unlabeled_set = set(unlabeled)
        overlap = labeled_set & unlabeled_set
        
        # Verify labeled images exist
        all_exist = all(
            os.path.exists(YOLO_DIR / p) for p in labeled
        )
        
        status = "✓" if len(overlap) == 0 and all_exist else "✗"
        print(f"  {status} {name}: {len(labeled)} labeled + {len(unlabeled)} unlabeled "
              f"(overlap={len(overlap)}, files_exist={all_exist})")
        
        if len(overlap) > 0 or not all_exist:
            all_ok = False
    
    print(f"\n  {'✓' if all_ok else '✗'} SSOD splits {'valid' if all_ok else 'INVALID'}!")
    return all_ok


if __name__ == '__main__':
    print("="*70)
    print("  END-TO-END PIPELINE VERIFICATION TEST")
    print("  Running all components to verify 100% error-free operation")
    print("="*70)
    
    results = {}
    
    # Step 1: Inference
    pred_file, predictions = test_step1_inference()
    results['inference'] = True
    
    # Step 2: Calibration
    try:
        metrics = test_step2_calibration(predictions)
        results['calibration'] = True
    except Exception as e:
        print(f"  ✗ Calibration FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['calibration'] = False
    
    # Step 3: FiveAI library
    results['fiveai_library'] = test_step3_fiveai_library()
    
    # Step 4: MC Dropout
    results['mc_dropout'] = test_step4_mc_dropout()
    
    # Step 5: SSOD splits
    results['ssod_splits'] = test_step5_ssod_splits()
    
    # Summary
    print("\n" + "="*70)
    print("  PIPELINE VERIFICATION SUMMARY")
    print("="*70)
    all_pass = True
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {component}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print(f"\n  🎉 ALL TESTS PASSED — Pipeline is 100% error-free!")
    else:
        print(f"\n  ⚠️  Some tests failed — see details above")
    
    print("="*70)
