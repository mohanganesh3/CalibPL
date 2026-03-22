"""
Benchmark Runtime Overhead of PSS-MSC.
Goal: To address the Area Chair concern about additional inference cost.
"""

import torch
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path

PROJ = Path("/home/mohanganesh/retail-shelf-detection")
MODEL_PATH = PROJ / "models" / "yolo11n.pt" # Use small model for benchmark

def benchmark_overhead():
    device = torch.device("cpu")
    model = YOLO(MODEL_PATH).to(device)
    
    # Dummy image
    img = torch.randn(1, 3, 640, 640).to(device)
    
    print("Benchmarking Baseline Inference (CPU)...")
    start = time.time()
    for _ in range(50): # Reduced iterations for CPU
        _ = model(img, verbose=False)
    baseline_time = (time.time() - start) / 50
    
    print(f"Baseline Time: {baseline_time*1000:.2f} ms/image")
    
    # Simulate PSS-MSC (3 scales + 2 augmentations = 5 total forward passes)
    print("Benchmarking PSS-MSC (5 Forward Passes - CPU)...")
    start = time.time()
    for _ in range(50):
        # Scale 1 (480)
        _ = model(torch.randn(1, 3, 480, 480).to(device), verbose=False)
        # Scale 2 (640)
        _ = model(torch.randn(1, 3, 640, 640).to(device), verbose=False)
        # Scale 3 (800)
        _ = model(torch.randn(1, 3, 800, 800).to(device), verbose=False)
        # Aug 1 (Flip)
        _ = model(img, verbose=False)
        # Aug 2 (Photometric)
        _ = model(img, verbose=False)
    pss_time = (time.time() - start) / 50
    
    print(f"PSS-MSC Time: {pss_time*1000:.2f} ms/image")
    print(f"Overhead: {pss_time/baseline_time:.1f}x")
    
    # Save results
    with open('results/benchmark_runtime.txt', 'w') as f:
        f.write(f"Baseline: {baseline_time*1000:.2f} ms\n")
        f.write(f"PSS-MSC: {pss_time*1000:.2f} ms\n")
        f.write(f"Overhead: {pss_time/baseline_time:.1f}x\n")

if __name__ == "__main__":
    benchmark_overhead()
