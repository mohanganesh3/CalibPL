"""
Benchmark: CGJS Computational Overhead on GPU.
"""

import torch
torch.backends.cudnn.enabled = False
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from scripts.prediction_stability import compute_cgjs_for_image

def benchmark_gpu_overhead():
    PROJ = Path("/home/mohanganesh/retail-shelf-detection")
    model_path = PROJ / "models/yolov8n.pt"
    # Use a real image for representative timing
    img_path = PROJ / "data/coco/train2017/000000116061.jpg"
    
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    
    model = YOLO(model_path).to(device)
    
    print(f"--- GPU BENCHMARK (Hardware: {gpu_name}) ---")
    
    # 1. Warm-up
    for _ in range(5):
        _ = model.predict(str(img_path), verbose=False)
        
    # 2. Baseline Inference (Single Forward)
    iters = 100
    start = time.time()
    for _ in range(iters):
        results = model.predict(str(img_path), verbose=False)
    baseline_time = (time.time() - start) / iters * 1000 # ms/image
    
    # 3. CGJS Overhead (Original, Multi-Scale + Augmentations = |A|=5)
    start = time.time()
    for _ in range(iters):
        _ = compute_cgjs_for_image(model, str(img_path), results[0], device=device, use_multi_scale=True)
    cgjs_time = (time.time() - start) / iters * 1000 # ms/image
    
    # 4. Lightweight CGJS Overhead (|A|=2 augmentations)
    start = time.time()
    for _ in range(iters):
        _ = compute_cgjs_for_image(model, str(img_path), results[0], device=device, use_multi_scale=False, lightweight=True)
    fast_cgjs_time = (time.time() - start) / iters * 1000 # ms/image
    
    print(f"Baseline Latency:        {baseline_time:.2f} ms/img")
    print(f"Full CGJS (|A|=5):       +{cgjs_time:.2f} ms/img (Total: {baseline_time + cgjs_time:.2f} ms | {(baseline_time + cgjs_time) / (baseline_time + 1e-6):.1f}x)")
    print(f"Lightweight CGJS (|A|=2): +{fast_cgjs_time:.2f} ms/img (Total: {baseline_time + fast_cgjs_time:.2f} ms | {(baseline_time + fast_cgjs_time) / (baseline_time + 1e-6):.1f}x)")
    
    print("\n* Benchmarked with batch size 1 (default YOLO inference style)")
    print("* Full CGJS configuration: 2 scales + 3 photometric/geom augs = 5 passes")
    print("* Fast CGJS configuration: 2 augs (Flip + Brightness), no multi-scale = 2 passes")

if __name__ == "__main__":
    benchmark_gpu_overhead()
