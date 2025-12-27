#!/usr/bin/env python3
"""Empirical validation of Proposition 1 (NMS competition amplifies localization overconfidence).

This script measures, on real detector predictions, how the *gap* between
classification confidence and localization correctness changes with the
"competition size" induced by greedy NMS.

Protocol (single-model, single-dataset):
1) Run Detectron2 inference with NMS effectively disabled
   (ROI_HEADS.NMS_THRESH_TEST=1.0) and a low score threshold.
2) Re-apply greedy, per-class NMS clustering at an IoU threshold `--nms-iou`.
   Each cluster corresponds to a competition set; its representative is the
   highest-score box (the one NMS would keep).
3) For each cluster representative, compute localization correctness against GT
   (IoU >= `--match-iou`, class-aware), and record:
      - cluster_size n
      - top_score s
      - loc_correct in {0,1}
      - (optional) cls_correct at IoU >= 0.1
4) Aggregate mean(s), mean(loc_correct), and mean(s - loc_correct) vs n.

Outputs:
- JSON summary + CSV table under `results/proposition1_validation/`.

Hardware discipline:
- Always export CUDA_VISIBLE_DEVICES=2,3 before running on this machine.
- Use `--gpu 0/1` to bind to physical GPU 2/3 when CUDA_VISIBLE_DEVICES is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

# Tesla K80 cuDNN fix (consistent with our training scripts).
torch.backends.cudnn.enabled = False

PROJ = Path(__file__).resolve().parent.parent

# COCO category ids are not contiguous (they skip {12,26,29,30,45,66,68,69,71,83}).
# Detectron2 uses contiguous ids [0..79] internally.
COCO_CONTIGUOUS_TO_DATASET_ID = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]
COCO_DATASET_ID_TO_CONTIGUOUS = {cid: i for i, cid in enumerate(COCO_CONTIGUOUS_TO_DATASET_ID)}


def _xywh_to_xyxy(b: List[float]) -> np.ndarray:
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _iou_vec_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and many boxes. Boxes are [x1,y1,x2,y2]."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / (union + 1e-6)


def _greedy_nms_cluster_sizes(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thresh: float,
) -> List[Tuple[int, int]]:
    """Return (leader_index, cluster_size) for greedy per-class NMS clustering.

    This reproduces the competition sets induced by greedy NMS:
    sort by score desc; keep a leader; suppress all remaining boxes with IoU >= thresh.

    Args:
        boxes_xyxy: (N,4)
        scores: (N,)
        classes: (N,) contiguous class ids
        iou_thresh: IoU threshold for suppression

    Returns:
        list of (leader_idx, cluster_size)
    """
    if len(boxes_xyxy) == 0:
        return []

    out: List[Tuple[int, int]] = []

    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        if len(idx) == 0:
            continue

        order = idx[np.argsort(scores[idx])[::-1]]
        used = np.zeros(len(order), dtype=bool)

        for oi in range(len(order)):
            if used[oi]:
                continue
            leader = order[oi]
            used[oi] = True

            if oi == len(order) - 1:
                out.append((int(leader), 1))
                continue

            rest_mask = ~used
            rest_mask[: oi + 1] = False
            rest = order[rest_mask]

            if len(rest) == 0:
                out.append((int(leader), 1))
                continue

            ious = _iou_vec_xyxy(boxes_xyxy[leader], boxes_xyxy[rest])
            suppressed = rest[ious >= iou_thresh]

            if len(suppressed) > 0:
                # Mark suppressed indices as used.
                # Map global indices back to positions in `order`.
                sup_pos = np.isin(order, suppressed)
                used |= sup_pos

            out.append((int(leader), int(1 + len(suppressed))))

    return out


def _best_iou_class_aware(
    pred_box: np.ndarray,
    pred_cls: int,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> float:
    if len(gt_boxes) == 0:
        return 0.0
    m = gt_classes == int(pred_cls)
    if not np.any(m):
        return 0.0
    ious = _iou_vec_xyxy(pred_box, gt_boxes[m])
    return float(np.max(ious)) if len(ious) else 0.0


@dataclass
class Row:
    n: int
    count: int
    mean_score: float
    mean_loc_acc: float
    mean_gap: float
    mean_cls_acc_iou01: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to Detectron2 .pth weights (must match faster_rcnn_R_50_FPN_3x config).",
    )
    ap.add_argument(
        "--gt-json",
        type=str,
        default=str(PROJ / "data" / "coco" / "annotations" / "instances_val2017.json"),
        help="COCO ground-truth JSON (val2017 by default).",
    )
    ap.add_argument(
        "--image-dir",
        type=str,
        default=str(PROJ / "data" / "coco" / "val2017"),
        help="Directory containing the images referenced by --gt-json.",
    )
    ap.add_argument("--gpu", type=int, default=0, help="Visible GPU index (after CUDA_VISIBLE_DEVICES).")
    ap.add_argument("--max-images", type=int, default=200, help="Max #images to sample from GT JSON.")
    ap.add_argument("--sample-seed", type=int, default=0, help="RNG seed used to sample images.")
    ap.add_argument(
        "--score-thresh",
        type=float,
        default=0.001,
        help="Detectron2 score threshold for candidate boxes (before our clustering).",
    )
    ap.add_argument(
        "--detections-per-image",
        type=int,
        default=1000,
        help="Max #detections per image returned by Detectron2 (NMS disabled).",
    )
    ap.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="IoU threshold used to form NMS competition clusters.",
    )
    ap.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for localization correctness (TP if IoU>=match_iou w/ GT of same class).",
    )
    ap.add_argument(
        "--cap-n",
        type=int,
        default=50,
        help="Cap competition size n at this value (the last bin aggregates all n>=cap_n).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJ / "results" / "proposition1_validation"),
        help="Output directory.",
    )
    args = ap.parse_args()

    setup_logger()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    gt_json = Path(args.gt_json)
    if not gt_json.exists():
        raise FileNotFoundError(f"gt json not found: {gt_json}")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"nms_competition_gap_"
        f"maximg{int(args.max_images)}_seed{int(args.sample_seed)}_"
        f"thr{args.score_thresh}_topk{int(args.detections_per_image)}_"
        f"nms{args.nms_iou}_match{args.match_iou}_cap{int(args.cap_n)}"
    )
    out_json = out_dir / f"{run_id}.json"
    out_csv = out_dir / f"{run_id}.csv"

    # --- Load GT ---
    print(f"[GT] Loading {gt_json} ...", file=sys.stderr)
    with gt_json.open() as f:
        gt = json.load(f)

    images = list(gt.get("images", []) or [])
    if not images:
        raise RuntimeError("GT JSON has no images")

    # Build GT maps
    gt_boxes_by_img: Dict[int, List[np.ndarray]] = {}
    gt_cls_by_img: Dict[int, List[int]] = {}

    for ann in gt.get("annotations", []) or []:
        try:
            img_id = int(ann.get("image_id"))
            cat_id = int(ann.get("category_id"))
            contig = COCO_DATASET_ID_TO_CONTIGUOUS.get(cat_id)
            if contig is None:
                continue
            box = _xywh_to_xyxy(ann.get("bbox", [0, 0, 0, 0]))
        except Exception:
            continue

        gt_boxes_by_img.setdefault(img_id, []).append(box)
        gt_cls_by_img.setdefault(img_id, []).append(int(contig))

    rng = np.random.default_rng(int(args.sample_seed))
    rng.shuffle(images)
    images = images[: int(args.max_images)]

    # --- Build predictor (NMS disabled) ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.score_thresh)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 1.0
    cfg.TEST.DETECTIONS_PER_IMAGE = int(args.detections_per_image)

    predictor = DefaultPredictor(cfg)

    print(
        f"[RUN] device={cfg.MODEL.DEVICE} | nms_disabled(NMS_THRESH_TEST=1.0) | "
        f"score_thresh={args.score_thresh} | topk={args.detections_per_image}",
        file=sys.stderr,
    )

    # --- Accumulate per-cluster samples ---
    samples_n: List[int] = []
    samples_s: List[float] = []
    samples_loc: List[int] = []
    samples_cls01: List[int] = []

    t0 = time.time()
    for k, im in enumerate(images):
        if k % 20 == 0:
            print(f"[RUN] {k}/{len(images)} images...", file=sys.stderr)

        img_id = int(im.get("id"))
        fn = im.get("file_name")
        if not fn:
            continue

        img_path = image_dir / fn
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        outputs = predictor(img)
        inst = outputs["instances"].to("cpu")

        if len(inst) == 0:
            continue

        boxes = inst.pred_boxes.tensor.numpy().astype(np.float32)
        scores = inst.scores.numpy().astype(np.float32)
        classes = inst.pred_classes.numpy().astype(np.int32)

        clusters = _greedy_nms_cluster_sizes(boxes, scores, classes, float(args.nms_iou))

        gt_boxes = np.stack(gt_boxes_by_img.get(img_id, []), axis=0) if img_id in gt_boxes_by_img else np.zeros((0, 4), dtype=np.float32)
        gt_cls = np.array(gt_cls_by_img.get(img_id, []), dtype=np.int32) if img_id in gt_cls_by_img else np.zeros((0,), dtype=np.int32)

        for leader_idx, n in clusters:
            n_cap = int(args.cap_n) if n >= int(args.cap_n) else int(n)
            s = float(scores[leader_idx])
            c = int(classes[leader_idx])

            best_iou = _best_iou_class_aware(boxes[leader_idx], c, gt_boxes, gt_cls)
            loc_corr = 1 if best_iou >= float(args.match_iou) else 0
            cls_corr01 = 1 if best_iou >= 0.1 else 0

            samples_n.append(n_cap)
            samples_s.append(s)
            samples_loc.append(loc_corr)
            samples_cls01.append(cls_corr01)

    dt = time.time() - t0

    if not samples_n:
        raise RuntimeError("No samples collected (check paths / thresholds)")

    n_arr = np.array(samples_n, dtype=np.int32)
    s_arr = np.array(samples_s, dtype=np.float32)
    loc_arr = np.array(samples_loc, dtype=np.float32)
    cls01_arr = np.array(samples_cls01, dtype=np.float32)

    rows: List[Row] = []
    for n in range(1, int(args.cap_n) + 1):
        m = n_arr == n
        count = int(np.sum(m))
        if count == 0:
            continue
        mean_s = float(np.mean(s_arr[m]))
        mean_loc = float(np.mean(loc_arr[m]))
        mean_gap = float(np.mean(s_arr[m] - loc_arr[m]))
        mean_cls01 = float(np.mean(cls01_arr[m]))
        rows.append(Row(n=n, count=count, mean_score=mean_s, mean_loc_acc=mean_loc, mean_gap=mean_gap, mean_cls_acc_iou01=mean_cls01))

    payload = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "weights": str(weights),
        "gt_json": str(gt_json),
        "image_dir": str(image_dir),
        "device": str(cfg.MODEL.DEVICE),
        "gpu_visible_index": int(args.gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "params": {
            "max_images": int(args.max_images),
            "sample_seed": int(args.sample_seed),
            "score_thresh": float(args.score_thresh),
            "detections_per_image": int(args.detections_per_image),
            "nms_iou": float(args.nms_iou),
            "match_iou": float(args.match_iou),
            "cap_n": int(args.cap_n),
        },
        "runtime_seconds": float(dt),
        "num_clusters": int(len(samples_n)),
        "rows": [r.__dict__ for r in rows],
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("n,count,mean_score,mean_loc_acc,mean_gap,mean_cls_acc_iou01\n")
        for r in rows:
            f.write(
                f"{r.n},{r.count},{r.mean_score:.6f},{r.mean_loc_acc:.6f},{r.mean_gap:.6f},{r.mean_cls_acc_iou01:.6f}\n"
            )

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_csv}")


if __name__ == "__main__":
    main()
