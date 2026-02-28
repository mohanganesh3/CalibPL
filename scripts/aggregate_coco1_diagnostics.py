#!/usr/bin/env python3
"""Aggregate COCO1% diagnostic artifacts into a single CSV/Markdown table.

This script is intentionally dependency-free (stdlib only). It is designed to be
robust to partially-finished experiments: missing files are left blank.

Expected inputs (by convention in this repo):
- Pseudo-label quality metrics: JSON dumped by scripts/evaluate_coco_pseudo_json.py
- AP summaries: summary.json produced by the training/eval loop

Usage example:
  python scripts/aggregate_coco1_diagnostics.py --seeds 42 43 44 \
    --out-csv results/diagnostic/coco1_diagnostics_3seeds.csv \
    --out-md  results/diagnostic/coco1_diagnostics_3seeds.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        # Treat unreadable/corrupted files as missing.
        return None


def _get_summary_ap(summary_json: Path) -> Tuple[Optional[float], Optional[float]]:
    d = _read_json(summary_json)
    if not d:
        return None, None

    # Common keys used in this repo.
    ap = d.get("map50_95")
    ap50 = d.get("map50")

    try:
        ap = float(ap) if ap is not None else None
    except Exception:
        ap = None

    try:
        ap50 = float(ap50) if ap50 is not None else None
    except Exception:
        ap50 = None

    return ap, ap50


def _get_pseudo_metrics(metrics_json: Path) -> Dict[str, Optional[float]]:
    d = _read_json(metrics_json)
    if not d:
        return {
            "images": None,
            "total_pl": None,
            "total_gt": None,
            "precision": None,
            "recall": None,
        }

    def _as_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    def _as_float(x: Any) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    return {
        "images": _as_int(d.get("images_considered")),
        "total_pl": _as_int(d.get("total_pl")),
        "total_gt": _as_int(d.get("total_gt")),
        "precision": _as_float(d.get("precision")),
        "recall": _as_float(d.get("recall")),
    }


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def _fmt_int(x: Optional[int]) -> str:
    return "" if x is None else str(x)


@dataclass(frozen=True)
class RowSpec:
    method: str
    iteration: str
    pseudo_metrics_path: Optional[str]
    ap_summary_path: Optional[str]


def build_row_specs(seed: int) -> List[RowSpec]:
    # NOTE: Many methods may not exist for all seeds yet; this is fine.
    s = str(seed)
    return [
        RowSpec(
            method="supervised",
            iteration="-",
            pseudo_metrics_path=None,
            ap_summary_path=f"results/phase14_baselines/coco1/seed{s}/summary.json",
        ),
        RowSpec(
            method="fixed_tau05",
            iteration="iter1",
            pseudo_metrics_path=f"results/diagnostic/fixed_tau05_coco1_seed{s}_iter1_pseudo_metrics_clean.json",
            ap_summary_path=f"results/phase14_selftrain/coco1/seed{s}_maxit50/iter_1/summary.json",
        ),
        RowSpec(
            method="fixed_tau05",
            iteration="iter2_newonly",
            pseudo_metrics_path=f"results/diagnostic/fixed_tau05_coco1_seed{s}_iter2_newonly_pseudo_metrics_clean.json",
            ap_summary_path=f"results/phase14_selftrain/coco1/seed{s}_maxit50/iter_2/summary.json",
        ),
        RowSpec(
            method="calibpl_beta0_densityadapt",
            iteration="iter1",
            pseudo_metrics_path=f"results/diagnostic/cgjs_beta0_densityadapt_coco1_seed{s}_iter1_pseudo_metrics.json",
            ap_summary_path=f"results/phase_c_calibpl/coco1/cgjs_beta0/seed{s}_maxit50/iter_1/summary.json",
        ),
        RowSpec(
            method="calibpl_full_densityadapt",
            iteration="iter1",
            pseudo_metrics_path=f"results/diagnostic/calibpl_full_densityadapt_coco1_seed{s}_iter1_pseudo_metrics.json",
            ap_summary_path=f"results/phase_c_calibpl/coco1/cgjs_full/seed{s}_maxit50/iter_1/summary.json",
        ),
        RowSpec(
            method="calibpl_full_densityadapt",
            iteration="iter2_union",
            pseudo_metrics_path=f"results/diagnostic/calibpl_full_densityadapt_coco1_seed{s}_iter2_union_pseudo_metrics.json",
            ap_summary_path=f"results/phase_c_calibpl/coco1/cgjs_full/seed{s}_maxit50/iter_2/summary.json",
        ),
        RowSpec(
            method="calibpl_full_densityadapt",
            iteration="iter2_newonly",
            pseudo_metrics_path=f"results/diagnostic/calibpl_full_densityadapt_coco1_seed{s}_iter2_newonly_pseudo_metrics.json",
            ap_summary_path=f"results/phase_c_calibpl/coco1/cgjs_full/seed{s}_maxit50/iter_2/summary.json",
        ),
    ]


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "seed",
        "method",
        "iteration",
        "images",
        "total_pl",
        "total_gt",
        "precision",
        "recall",
        "ap",
        "ap50",
        "pseudo_metrics_path",
        "ap_summary_path",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_md(rows: List[Dict[str, str]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    cols = ["seed", "method", "iteration", "precision", "recall", "ap", "ap50"]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")

    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--out-md", type=str, required=True)
    args = ap.parse_args()

    out_csv = (REPO_ROOT / args.out_csv).resolve()
    out_md = (REPO_ROOT / args.out_md).resolve()

    rows: List[Dict[str, str]] = []

    for seed in args.seeds:
        for spec in build_row_specs(seed):
            pseudo_path = Path(spec.pseudo_metrics_path) if spec.pseudo_metrics_path else None
            ap_path = Path(spec.ap_summary_path) if spec.ap_summary_path else None

            pseudo_abs = (REPO_ROOT / pseudo_path).resolve() if pseudo_path else None
            ap_abs = (REPO_ROOT / ap_path).resolve() if ap_path else None

            pm = _get_pseudo_metrics(pseudo_abs) if pseudo_abs else {
                "images": None,
                "total_pl": None,
                "total_gt": None,
                "precision": None,
                "recall": None,
            }
            ap_val, ap50_val = _get_summary_ap(ap_abs) if ap_abs else (None, None)

            rows.append(
                {
                    "seed": str(seed),
                    "method": spec.method,
                    "iteration": spec.iteration,
                    "images": _fmt_int(pm.get("images")),
                    "total_pl": _fmt_int(pm.get("total_pl")),
                    "total_gt": _fmt_int(pm.get("total_gt")),
                    "precision": _fmt(pm.get("precision")),
                    "recall": _fmt(pm.get("recall")),
                    "ap": _fmt(ap_val),
                    "ap50": _fmt(ap50_val),
                    "pseudo_metrics_path": spec.pseudo_metrics_path or "",
                    "ap_summary_path": spec.ap_summary_path or "",
                }
            )

    write_csv(rows, out_csv)
    write_md(rows, out_md)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
