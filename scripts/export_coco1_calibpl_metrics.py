#!/usr/bin/env python3
"""Export clean JSON pseudo-label quality metrics for CalibPL COCO 1% runs.

Why this exists:
- The repo already has `scripts/evaluate_coco_pseudo_json.py`, which prints progress
  to stderr and returns JSON to stdout.
- For artifact discipline and aggregation, we want deterministic, machine-loadable
  JSON files written directly by a script (no shell redirection hazards).

This script writes (when available):
- results/diagnostic/calibpl_full_densityadapt_coco1_seed{seed}_iter1_pseudo_metrics.json
- results/diagnostic/calibpl_full_densityadapt_coco1_seed{seed}_iter2_union_pseudo_metrics.json
- results/diagnostic/calibpl_full_densityadapt_coco1_seed{seed}_iter2_newonly_pseudo_metrics.json

Metrics are computed on the sampled unlabeled subset present in pseudo_json
(`subset=pseudo_json`), at IoU>=0.5 by default.

Example:
  python scripts/export_coco1_calibpl_metrics.py --seed 44 --train-max-iter 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

# Allow running this file directly (python scripts/export_coco1_calibpl_metrics.py)
# even though `scripts/` is not an importable package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_coco_pseudo_json import evaluate_pseudo_json  # noqa: E402


def _write_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    # Validate
    json.loads(tmp.read_text())
    tmp.replace(out_path)


def _calibpl_run_dir(seed: int, train_max_iter: Optional[int]) -> Path:
    suffix = "" if train_max_iter is None else f"_maxit{int(train_max_iter)}"
    return REPO_ROOT / "results" / "phase_c_calibpl" / "coco1" / "cgjs_full" / f"seed{seed}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-max-iter", type=int, default=50)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = _calibpl_run_dir(args.seed, args.train_max_iter)

    ul_dir = REPO_ROOT / "data" / "coco" / "unlabeled2017"
    gt_json = REPO_ROOT / "data" / "coco" / "annotations" / "instances_train2017.json"

    # Iteration 1
    p1 = run_dir / "iter_1" / "pseudo_train.json"
    if p1.exists():
        d1 = evaluate_pseudo_json(
            pseudo_json=p1,
            unlabeled_dir=ul_dir,
            gt_json=gt_json,
            iou_thresh=float(args.iou),
            prev_pseudo_json=None,
            subset="pseudo_json",
        )
        _write_json(
            d1,
            REPO_ROOT
            / "results"
            / "diagnostic"
            / f"calibpl_full_densityadapt_coco1_seed{args.seed}_iter1_pseudo_metrics.json",
        )
    else:
        print(f"[warn] missing: {p1}")

    # Iteration 2
    p2 = run_dir / "iter_2" / "pseudo_train.json"
    if p2.exists():
        # UNION view
        d2u = evaluate_pseudo_json(
            pseudo_json=p2,
            unlabeled_dir=ul_dir,
            gt_json=gt_json,
            iou_thresh=float(args.iou),
            prev_pseudo_json=None,
            subset="pseudo_json",
        )
        _write_json(
            d2u,
            REPO_ROOT
            / "results"
            / "diagnostic"
            / f"calibpl_full_densityadapt_coco1_seed{args.seed}_iter2_union_pseudo_metrics.json",
        )

        # NEW-ONLY view (requires iter1)
        if p1.exists():
            d2n = evaluate_pseudo_json(
                pseudo_json=p2,
                unlabeled_dir=ul_dir,
                gt_json=gt_json,
                iou_thresh=float(args.iou),
                prev_pseudo_json=p1,
                subset="pseudo_json",
            )
            _write_json(
                d2n,
                REPO_ROOT
                / "results"
                / "diagnostic"
                / f"calibpl_full_densityadapt_coco1_seed{args.seed}_iter2_newonly_pseudo_metrics.json",
            )
        else:
            print(f"[warn] missing (needed for new-only): {p1}")
    else:
        print(f"[warn] missing: {p2}")


if __name__ == "__main__":
    main()
