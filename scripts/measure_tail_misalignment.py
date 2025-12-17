#!/usr/bin/env python3
"""
Empirical measurement of the tail misalignment function g(s) = P(L=1|S=s).

This script directly validates Assumption (A) of Proposition 1:
    g(s) ≤ s − δ for s ∈ [s_0, 1]

If this assumption holds empirically, then Proposition 1's conclusion is not just
a mathematical possibility—it is an empirical fact about our detectors.

Protocol:
1. Run detector inference on validation set
2. Match predictions to ground truth (IoU ≥ 0.5 → localization correct)
3. Bin predictions by score s into fine bins (0.05 width)
4. Compute g(s) = P(L=1|S=s) for each bin
5. Split analysis by density: sparse (≤5 obj/img) vs dense (>12 obj/img)
6. Output: reliability diagram + tail misalignment measurements

Key insight: if dense images show g(s) < s in the tail while sparse images show
g(s) ≈ s, then we have proven that the phenomenon is density-dependent and NMS
is the causal mechanism.

GPU: CUDA_VISIBLE_DEVICES=2,3 --gpu 0/1 (maps to physical 2/3)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

# Tesla K80 cuDNN fix
torch.backends.cudnn.enabled = False

PROJ = Path(__file__).resolve().parent.parent

# COCO ID mapping
COCO_CONTIGUOUS_TO_DATASET_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]
COCO_DATASET_ID_TO_CONTIGUOUS = {cid: i for i, cid in enumerate(COCO_CONTIGUOUS_TO_DATASET_ID)}


def xywh_to_xyxy(b: List[float]) -> np.ndarray:
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou_vec_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and many boxes. Boxes are [x1,y1,x2,y2]."""
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / (union + 1e-9)


def best_iou_class_aware(
    pred_box: np.ndarray,
    pred_cls: int,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> float:
    """Best IoU with GT of matching class."""
    if len(gt_boxes) == 0:
        return 0.0
    m = gt_classes == pred_cls
    if not np.any(m):
        return 0.0
    ious = iou_vec_xyxy(pred_box, gt_boxes[m])
    return float(np.max(ious)) if len(ious) else 0.0


def main():
    parser = argparse.ArgumentParser(description="Measure g(s) = P(L=1|S=s) for Proposition 1 validation")
    parser.add_argument("--weights", type=str, required=True, help="Detectron2 .pth weights")
    parser.add_argument("--gt-json", type=str, default=str(PROJ / "data/coco/annotations/instances_val2017.json"))
    parser.add_argument("--image-dir", type=str, default=str(PROJ / "data/coco/val2017"))
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (after CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--max-images", type=int, default=1000, help="Max images to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--score-thresh", type=float, default=0.05, help="Score threshold for predictions")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for localization correctness")
    parser.add_argument("--sparse-max", type=int, default=5, help="Max objects for sparse classification")
    parser.add_argument("--dense-min", type=int, default=12, help="Min objects for dense classification")
    parser.add_argument("--out-dir", type=str, default=str(PROJ / "results/tail_misalignment"))
    args = parser.parse_args()

    setup_logger()
    
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    
    gt_json = Path(args.gt_json)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT
    print(f"[GT] Loading {gt_json}...", file=sys.stderr)
    with open(gt_json) as f:
        gt = json.load(f)
    
    images = gt.get("images", [])
    if not images:
        raise RuntimeError("No images in GT JSON")
    
    # Build GT lookup
    gt_boxes_by_img: Dict[int, List[np.ndarray]] = {}
    gt_cls_by_img: Dict[int, List[int]] = {}
    
    for ann in gt.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        contig = COCO_DATASET_ID_TO_CONTIGUOUS.get(cat_id)
        if contig is None:
            continue
        box = xywh_to_xyxy(ann.get("bbox", [0, 0, 0, 0]))
        gt_boxes_by_img.setdefault(img_id, []).append(box)
        gt_cls_by_img.setdefault(img_id, []).append(contig)
    
    # Compute density for each image
    img_density = {im["id"]: len(gt_boxes_by_img.get(im["id"], [])) for im in images}
    
    # Sample images
    rng = np.random.default_rng(args.seed)
    rng.shuffle(images)
    images = images[:args.max_images]
    
    # Build predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    predictor = DefaultPredictor(cfg)
    
    print(f"[RUN] Device: {cfg.MODEL.DEVICE}, Score threshold: {args.score_thresh}", file=sys.stderr)
    
    # Collect samples: (score, loc_correct, density_category)
    # density_category: 0=sparse, 1=medium, 2=dense
    samples: List[Tuple[float, int, int]] = []
    
    t0 = time.time()
    for k, im in enumerate(images):
        if k % 50 == 0:
            print(f"[RUN] {k}/{len(images)} images...", file=sys.stderr)
        
        img_id = im["id"]
        fn = im.get("file_name")
        if not fn:
            continue
        
        img_path = image_dir / fn
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        density = img_density.get(img_id, 0)
        if density <= args.sparse_max:
            density_cat = 0  # sparse
        elif density >= args.dense_min:
            density_cat = 2  # dense
        else:
            density_cat = 1  # medium
        
        outputs = predictor(img)
        inst = outputs["instances"].to("cpu")
        
        if len(inst) == 0:
            continue
        
        boxes = inst.pred_boxes.tensor.numpy().astype(np.float32)
        scores = inst.scores.numpy().astype(np.float32)
        classes = inst.pred_classes.numpy().astype(np.int32)
        
        gt_boxes = np.stack(gt_boxes_by_img.get(img_id, []), axis=0) if img_id in gt_boxes_by_img else np.zeros((0, 4), dtype=np.float32)
        gt_cls = np.array(gt_cls_by_img.get(img_id, []), dtype=np.int32) if img_id in gt_cls_by_img else np.zeros((0,), dtype=np.int32)
        
        for i in range(len(boxes)):
            s = float(scores[i])
            c = int(classes[i])
            best_iou = best_iou_class_aware(boxes[i], c, gt_boxes, gt_cls)
            loc_corr = 1 if best_iou >= args.match_iou else 0
            samples.append((s, loc_corr, density_cat))
    
    dt = time.time() - t0
    print(f"[RUN] Done in {dt:.1f}s, collected {len(samples)} samples", file=sys.stderr)
    
    if len(samples) == 0:
        raise RuntimeError("No samples collected!")
    
    # Convert to arrays
    s_arr = np.array([x[0] for x in samples], dtype=np.float32)
    loc_arr = np.array([x[1] for x in samples], dtype=np.float32)
    density_arr = np.array([x[2] for x in samples], dtype=np.int32)
    
    # Define bins
    bins = np.arange(0, 1.05, 0.05)  # 20 bins from 0 to 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Compute g(s) for each density category
    results = {}
    density_names = ["sparse", "medium", "dense"]
    
    for d, name in enumerate(density_names):
        mask = density_arr == d
        if not np.any(mask):
            results[name] = {"n": 0, "g_values": [], "n_per_bin": []}
            continue
        
        s_d = s_arr[mask]
        loc_d = loc_arr[mask]
        
        g_values = []
        n_per_bin = []
        
        for i in range(len(bins) - 1):
            bin_mask = (s_d >= bins[i]) & (s_d < bins[i + 1])
            n_bin = np.sum(bin_mask)
            n_per_bin.append(int(n_bin))
            if n_bin >= 10:
                g = float(np.mean(loc_d[bin_mask]))
            else:
                g = np.nan
            g_values.append(g)
        
        results[name] = {
            "n": int(np.sum(mask)),
            "g_values": g_values,
            "n_per_bin": n_per_bin,
        }
    
    # Measure tail misalignment (s >= 0.8)
    tail_start = 0.8
    tail_mask = s_arr >= tail_start
    
    tail_analysis = {}
    for d, name in enumerate(density_names):
        mask = (density_arr == d) & tail_mask
        if np.sum(mask) < 50:
            tail_analysis[name] = {"n_tail": int(np.sum(mask)), "delta": None, "mean_s": None, "mean_g": None}
            continue
        
        mean_s = float(np.mean(s_arr[mask]))
        mean_g = float(np.mean(loc_arr[mask]))  # g(s) in tail = P(L=1|S>=0.8)
        delta = mean_s - mean_g  # This is the tail misalignment
        
        tail_analysis[name] = {
            "n_tail": int(np.sum(mask)),
            "mean_s": mean_s,
            "mean_g": mean_g,
            "delta": delta,
        }
    
    # Save results
    run_id = f"tail_misalignment_seed{args.seed}_maximg{args.max_images}"
    
    payload = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "weights": str(weights),
        "params": {
            "max_images": args.max_images,
            "seed": args.seed,
            "score_thresh": args.score_thresh,
            "match_iou": args.match_iou,
            "sparse_max": args.sparse_max,
            "dense_min": args.dense_min,
        },
        "runtime_seconds": dt,
        "total_samples": len(samples),
        "bin_edges": bins.tolist(),
        "bin_centers": bin_centers.tolist(),
        "density_results": results,
        "tail_analysis": tail_analysis,
    }
    
    out_json = out_dir / f"{run_id}.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote: {out_json}")
    
    # Create reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: g(s) vs s for each density
    ax = axes[0]
    colors = {"sparse": "blue", "medium": "gray", "dense": "red"}
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    
    for name in density_names:
        if results[name]["n"] > 0:
            g_vals = results[name]["g_values"]
            # Only plot bins with enough samples
            valid = [i for i, (g, n) in enumerate(zip(g_vals, results[name]["n_per_bin"])) if not np.isnan(g)]
            if valid:
                ax.plot(
                    [bin_centers[i] for i in valid],
                    [g_vals[i] for i in valid],
                    "o-",
                    color=colors[name],
                    label=f"{name} (n={results[name]['n']})",
                    markersize=4,
                )
    
    ax.set_xlabel("Classification Score s", fontsize=12)
    ax.set_ylabel("g(s) = P(Loc Correct | S=s)", fontsize=12)
    ax.set_title("Localization Reliability Diagram by Density", fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Right: Tail misalignment (bar chart)
    ax = axes[1]
    names_with_data = [n for n in density_names if tail_analysis[n]["delta"] is not None]
    if names_with_data:
        deltas = [tail_analysis[n]["delta"] for n in names_with_data]
        bar_colors = [colors[n] for n in names_with_data]
        bars = ax.bar(names_with_data, deltas, color=bar_colors, alpha=0.7)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(r"$\delta = \bar{s} - g(\bar{s})$ in tail ($s \geq 0.8$)", fontsize=12)
        ax.set_title("Tail Misalignment by Density", fontsize=14)
        
        # Add value labels
        for bar, delta in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2, delta + 0.01, f"{delta:.3f}", 
                    ha="center", va="bottom", fontsize=10)
    
    plt.tight_layout()
    out_fig = out_dir / f"{run_id}.png"
    plt.savefig(out_fig, dpi=150)
    print(f"[OK] Wrote: {out_fig}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TAIL MISALIGNMENT ANALYSIS (Proposition 1 Validation)")
    print("=" * 60)
    for name in density_names:
        ta = tail_analysis[name]
        if ta["delta"] is not None:
            print(f"{name.upper():8s}: n={ta['n_tail']:5d}, mean_s={ta['mean_s']:.4f}, g(s)={ta['mean_g']:.4f}, δ={ta['delta']:.4f}")
        else:
            print(f"{name.upper():8s}: insufficient samples (n={ta['n_tail']})")
    print("=" * 60)
    
    # Key question: is δ_dense > δ_sparse?
    if tail_analysis["dense"]["delta"] is not None and tail_analysis["sparse"]["delta"] is not None:
        diff = tail_analysis["dense"]["delta"] - tail_analysis["sparse"]["delta"]
        print(f"\nDensity effect: δ_dense - δ_sparse = {diff:.4f}")
        if diff > 0.05:
            print("✓ SUPPORTS Proposition 1: tail misalignment is larger in dense scenes")
        elif diff < -0.05:
            print("✗ CONTRADICTS Proposition 1: tail misalignment is smaller in dense scenes")
        else:
            print("○ INCONCLUSIVE: tail misalignment difference is small")


if __name__ == "__main__":
    main()
