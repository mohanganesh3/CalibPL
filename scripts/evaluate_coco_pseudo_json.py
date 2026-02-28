#!/usr/bin/env python3
"""Evaluate pseudo-label COCO JSON against COCO ground truth for the *unlabeled* subset.

This is used for CalibPL diagnostics on COCO 1%.

Given:
- pseudo_json: COCO-format JSON that contains labeled train images + pseudo-labeled unlabeled images
- unlabeled_dir: directory containing the unlabeled images (filenames define the evaluation subset)
- gt_json: COCO instances_train2017.json (or equivalent) containing ground-truth for those images

We compute (class-aware) TP/FP/FN at IoU>=iou_thresh.

Notes:
- We evaluate only on images whose basenames are present in unlabeled_dir.
- Category matching is done by COCO category_id (1-based).
- For iterative diagnostics, it is important to include *sampled unlabeled images even when no
    pseudo-labels were kept for them*, otherwise recall will be overestimated. Use
    subset='pseudo_json_images' (or the legacy alias 'pseudo_json').
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set


def _xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return x, y, x + w, y + h


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _load_gt_by_filename(gt_json: Path, keep_fns: Optional[Set[str]] = None) -> Dict[str, List[dict]]:
    """Load GT annotations grouped by image *filename*.

    If keep_fns is provided, only images whose basename is in keep_fns are indexed.
    This makes repeated diagnostics on a subset (e.g., unlabeled split) much faster.
    """
    print(f"[GT] Loading {gt_json} ...", file=sys.stderr)
    with gt_json.open() as f:
        gt = json.load(f)

    # Keep only the images we will actually evaluate.
    img_id_to_fn: Dict[int, str] = {}
    for im in gt.get("images", []):
        fn = Path(im.get("file_name", "")).name
        if keep_fns is not None and fn not in keep_fns:
            continue
        img_id_to_fn[int(im["id"])] = fn

    gt_by_fn: Dict[str, List[dict]] = {}
    anns = gt.get("annotations", [])
    for k, ann in enumerate(anns):
        if k % 500000 == 0 and k > 0:
            print(f"[GT] Scanned {k}/{len(anns)} annotations...", file=sys.stderr)
        fn = img_id_to_fn.get(int(ann.get("image_id", -1)))
        if fn is None:
            continue
        gt_by_fn.setdefault(fn, []).append(ann)

    print(f"[GT] Indexed {len(gt_by_fn)} images.", file=sys.stderr)
    return gt_by_fn


def _load_pl_by_filename(
    pseudo_json: Path, keep_fns: Optional[Set[str]] = None
) -> Tuple[Dict[str, List[dict]], List[str], List[str]]:
    """Load pseudo-label annotations grouped by image *filename*.

    Returns:
        pl_by_fn: filename -> list of pseudo annotations (only images with at least one annotation)
        image_fns: all image basenames present in pseudo_json['images'] (after keep_fns filtering)
        ann_fns: image basenames that have at least one pseudo annotation

    If keep_fns is provided, only images whose basename is in keep_fns are indexed.
    """
    print(f"[PL] Loading {pseudo_json} ...", file=sys.stderr)
    with pseudo_json.open() as f:
        pl = json.load(f)

    img_id_to_fn: Dict[int, str] = {}
    for im in pl.get("images", []):
        fn = Path(im.get("file_name", "")).name
        if keep_fns is not None and fn not in keep_fns:
            continue
        img_id_to_fn[int(im["id"])] = fn

    pl_by_fn: Dict[str, List[dict]] = {}
    image_fns = set(img_id_to_fn.values())
    ann_fns = set()
    anns = pl.get("annotations", [])
    for k, ann in enumerate(anns):
        if k % 500000 == 0 and k > 0:
            print(f"[PL] Scanned {k}/{len(anns)} annotations...", file=sys.stderr)
        fn = img_id_to_fn.get(int(ann.get("image_id", -1)))
        if fn is None:
            continue
        ann_fns.add(fn)
        pl_by_fn.setdefault(fn, []).append(ann)

    print(f"[PL] Indexed {len(pl_by_fn)} images (with any pseudo ann).", file=sys.stderr)
    print(f"[PL] Found {len(image_fns)} images in pseudo_json['images'] (subset after filtering).", file=sys.stderr)
    return pl_by_fn, sorted(image_fns), sorted(ann_fns)


def evaluate_pseudo_json(
    pseudo_json: Path,
    unlabeled_dir: Path,
    gt_json: Path,
    iou_thresh: float = 0.5,
    prev_pseudo_json: Optional[Path] = None,
    subset: str = "unlabeled_dir",
) -> dict:
    unlabeled_files = {
        p.name for p in unlabeled_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}
    }

    prev_max_ann_id: Optional[int] = None
    if prev_pseudo_json is not None:
        # Compute a stable id threshold for “newly added” pseudo-labels.
        # In our iterative pipelines, each iteration starts ann_id at (max previous id + 1),
        # and carries over previous annotations unchanged.
        with prev_pseudo_json.open() as f:
            prev = json.load(f)
        prev_img_id_to_fn: Dict[int, str] = {}
        for im in prev.get("images", []):
            fn = Path(im.get("file_name", "")).name
            if fn in unlabeled_files:
                prev_img_id_to_fn[int(im["id"])] = fn

        m = -1
        for ann in prev.get("annotations", []):
            if int(ann.get("image_id", -1)) in prev_img_id_to_fn:
                try:
                    m = max(m, int(ann.get("id")))
                except Exception:
                    continue
        prev_max_ann_id = m if m >= 0 else None
        print(
            f"[PL] prev_pseudo_json={prev_pseudo_json} | prev_max_ann_id={prev_max_ann_id}",
            file=sys.stderr,
        )

    pl_by_fn, image_fns, ann_fns = _load_pl_by_filename(pseudo_json, keep_fns=unlabeled_files)

    # Backward-compatible alias: historically we had subset='pseudo_json' but accidentally
    # evaluated only images with at least one pseudo annotation (overestimates recall).
    if subset == "pseudo_json":
        subset = "pseudo_json_images"

    if subset not in {"unlabeled_dir", "pseudo_json_images", "pseudo_json_anns"}:
        raise ValueError(
            f"Unknown subset={subset!r}. Expected 'unlabeled_dir', 'pseudo_json_images', or 'pseudo_json_anns'."
        )

    if subset == "pseudo_json_images":
        # Evaluate on all sampled unlabeled images present in pseudo_json['images'].
        # This includes images where no pseudo labels were kept, so recall is not inflated.
        eval_fns = set(image_fns)
    elif subset == "pseudo_json_anns":
        # Evaluate only images that have at least one pseudo annotation.
        # Useful for conditional precision/recall, but will overestimate recall for strict gates.
        eval_fns = set(ann_fns)
    else:
        # Evaluate against all filenames in unlabeled_dir (slow; recall will be tiny if you
        # pseudo-label only a small sampled subset).
        eval_fns = unlabeled_files

    if not eval_fns:
        # Nothing to evaluate.
        return {
            "pseudo_json": str(pseudo_json),
            "unlabeled_dir": str(unlabeled_dir),
            "gt_json": str(gt_json),
            "iou_thresh": float(iou_thresh),
            "subset": subset,
            "images_eval_set": 0,
            "images_considered": 0,
            "images_with_any_pred": 0,
            "images_with_any_gt": 0,
            "total_pl": 0,
            "total_gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
        }

    gt_by_fn = _load_gt_by_filename(gt_json, keep_fns=eval_fns)

    if prev_max_ann_id is not None:
        # Filter to only “new” annotations by id threshold.
        filtered: Dict[str, List[dict]] = {}
        new_total = 0
        for fn, anns in pl_by_fn.items():
            new_anns = []
            for a in anns:
                try:
                    if int(a.get("id", -1)) > prev_max_ann_id:
                        new_anns.append(a)
                except Exception:
                    continue
            if new_anns:
                filtered[fn] = new_anns
                new_total += len(new_anns)
        pl_by_fn = filtered
        print(f"[PL] Filtering to new annotations only: total_new={new_total}", file=sys.stderr)

    tp = fp = fn = 0
    total_pl = total_gt = 0
    images_considered = 0
    images_with_any_pred = 0
    images_with_any_gt = 0

    for fn_img in sorted(eval_fns):
        preds = pl_by_fn.get(fn_img, [])
        gts = gt_by_fn.get(fn_img, [])

        if preds:
            images_with_any_pred += 1
        if gts:
            images_with_any_gt += 1

        if not preds and not gts:
            continue

        images_considered += 1
        total_pl += len(preds)
        total_gt += len(gts)

        pred_boxes = [(_xywh_to_xyxy(p["bbox"]), int(p["category_id"])) for p in preds]
        gt_boxes = [(_xywh_to_xyxy(g["bbox"]), int(g["category_id"])) for g in gts]

        matched_gt = set()
        for pb, pc in pred_boxes:
            best_iou = 0.0
            best_j = -1
            for j, (gb, gc) in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if pc != gc:
                    continue
                iou = _iou_xyxy(pb, gb)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                matched_gt.add(best_j)
                tp += 1
            else:
                fp += 1

        fn += (len(gt_boxes) - len(matched_gt))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "pseudo_json": str(pseudo_json),
        "unlabeled_dir": str(unlabeled_dir),
        "gt_json": str(gt_json),
        "iou_thresh": float(iou_thresh),
        "subset": subset,
        "images_eval_set": int(len(eval_fns)),
        "images_considered": int(images_considered),
        "images_with_any_pred": int(images_with_any_pred),
        "images_with_any_gt": int(images_with_any_gt),
        "total_pl": int(total_pl),
        "total_gt": int(total_gt),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pseudo-json", type=str, required=True)
    p.add_argument("--unlabeled-dir", type=str, required=True)
    p.add_argument(
        "--prev-pseudo-json",
        type=str,
        default=None,
        help="If set, evaluate only annotations newly added since this previous pseudo JSON (id threshold).",
    )
    p.add_argument(
        "--gt-json",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "coco" / "annotations" / "instances_train2017.json"),
    )
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--subset",
        type=str,
        default="unlabeled_dir",
        choices=["unlabeled_dir", "pseudo_json", "pseudo_json_images", "pseudo_json_anns"],
        help=(
            "Evaluation subset: 'unlabeled_dir' evaluates against all filenames in unlabeled_dir (slow; recall tiny if you pseudo-label only a small sample). "
            "'pseudo_json_images' evaluates on *all* unlabeled images present in pseudo_json['images'] (recommended for diagnostics). "
            "'pseudo_json_anns' evaluates only images that have at least one pseudo annotation (conditional metrics; recall inflated for strict gates). "
            "Legacy alias: 'pseudo_json' == 'pseudo_json_images'."
        ),
    )
    args = p.parse_args()

    stats = evaluate_pseudo_json(
        pseudo_json=Path(args.pseudo_json),
        unlabeled_dir=Path(args.unlabeled_dir),
        gt_json=Path(args.gt_json),
        iou_thresh=float(args.iou),
        prev_pseudo_json=Path(args.prev_pseudo_json) if args.prev_pseudo_json else None,
        subset=str(args.subset),
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
