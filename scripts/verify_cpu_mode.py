#!/usr/bin/env python3
"""
Quick check to verify CPU-only mode is configured correctly.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def verify_cpu_mode():
    print("="*80)
    print("CPU MODE VERIFICATION")
    print("="*80)
    
    # 1. Check config
    from core.config_exact import CONFIG
    device = CONFIG['hardware']['device']
    num_gpus = CONFIG['hardware']['num_gpus']
    
    print(f"\n1. Config check:")
    print(f"   Device: {device}")
    print(f"   Num GPUs: {num_gpus}")
    assert device == 'cpu', f"ERROR: Device should be 'cpu', got '{device}'"
    assert num_gpus == 0, f"ERROR: Num GPUs should be 0, got {num_gpus}"
    print("   ✓ Config set to CPU mode")
    
    # 2. Check PyTorch
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"\n2. PyTorch check:")
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print("   WARNING: CUDA is available but should be disabled")
        print("   Setting CUDA_VISIBLE_DEVICES='' will hide GPUs")
    else:
        print("   ✓ CUDA not available (as expected)")
    
    # 3. Check environment variable
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"\n3. Environment check:")
    print(f"   CUDA_VISIBLE_DEVICES: '{cuda_devices}'")
    
    print("\n" + "="*80)
    print("✓ CPU MODE VERIFICATION PASSED")
    print("="*80)
    print("\nEstimated training times on CPU:")
    print("  Phase 2 (Faster R-CNN + YOLO): 40-60 hours")
    print("  Phase 3 (Ensemble):             15-20 hours")
    print("  Phase 4 (Co-training):          100-120 hours")
    print("  Phase 5 (Hyperparameter opt):   50-70 hours")
    print("  Phase 6 (Final eval):            5-10 hours")
    print("  " + "-"*50)
    print("  TOTAL:                          210-280 hours (8-12 days)")
    print("\nNote: This is 6-10x slower than training on compatible GPUs")
    print("="*80)

if __name__ == "__main__":
    verify_cpu_mode()
