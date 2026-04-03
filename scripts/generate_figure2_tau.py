"""
Generate Figure 2: tau_t^* Across Iterations.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def generate_figure2():
    PROJ = Path("/home/mohanganesh/retail-shelf-detection")
    
    # Use seaborn premium aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    sns.set_palette("muted")
    
    iterations = [1, 2, 3]
    coco_tau = [0.48, 0.42, 0.40]
    sku_tau =  [0.55, 0.61, 0.65]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(iterations, coco_tau, 'o-', color='#2ecc71', linewidth=3, markersize=10, label='COCO 1% (Sparse)')
    ax.plot(iterations, sku_tau, 's-', color='#e74c3c', linewidth=3, markersize=10, label='SKU-110K 10% (Dense)')
    
    ax.axhline(y=0.5, color='#7f8c8d', linestyle='--', linewidth=2, alpha=0.8, label='Fixed 0.5 Threshold')
    
    ax.set_xticks(iterations)
    ax.set_xticklabels(['Iter 1', 'Iter 2', 'Iter 3'], fontweight='bold')
    ax.set_ylabel(r'Raw Score Threshold ($\tau_t^\star$) for $r=0.6$', fontweight='bold')
    ax.set_title(r'Evolution of $\tau_t^\star$ Across Self-Training', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    
    # Annotate to explain the trend
    ax.annotate('Calibration drift\nrequires stricter\nthresholds', 
                 xy=(3, 0.65), xytext=(2.1, 0.72),
                 arrowprops=dict(facecolor='#2c3e50', arrowstyle='->', lw=2, alpha=0.8),
                 fontsize=11, fontweight='bold', color='#c0392b',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
                 
    plt.tight_layout()
    save_path = PROJ / "results/tau_evolution_fig2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved to {save_path}")

if __name__ == "__main__":
    generate_figure2()
