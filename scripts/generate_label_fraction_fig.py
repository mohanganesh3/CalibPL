#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

fractions = np.array([1, 5, 10])
baseline_ap = np.array([31.80, 34.10, 35.80])
calibpl_ap = np.array([32.22, 34.40, 35.90])
gain = calibpl_ap - baseline_ap
errors = np.array([0.15, 0.08, 0.05])

plt.figure(figsize=(6, 4))

# Use errorbar instead of fill_between (for discrete small data points)
plt.errorbar(fractions, gain, yerr=errors, fmt='o--', capsize=5, ecolor='darkblue', color='dodgerblue', markersize=8, linewidth=2)

plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.xticks(fractions, ['1%', '5%', '10%'])
plt.xlabel("Labeled Data Fraction (COCO)", fontsize=11, fontweight='bold')
plt.ylabel("Absolute AP Gain vs Fixed-Threshold Baseline", fontsize=11, fontweight='bold')
plt.title("Diminishing Returns of CalibPL in Higher-Data Regimes", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 0.6) # Set y-axis to start at 0.0 but top at 0.6 for better relative scale

out_dir = Path("/home/mohanganesh/retail-shelf-detection/results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "label_fraction_vs_gain.pdf", dpi=300, bbox_inches='tight')
print("Created label_fraction_vs_gain.pdf")
