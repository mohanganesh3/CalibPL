"""
Generate Figure 3: Density-Conditioned AP (Precision vs. Object Count).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def generate_figure3():
    PROJ = Path("/home/mohanganesh/retail-shelf-detection")
    
    # Use seaborn premium aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    sns.set_palette("deep")
    
    # Bins: Sparse (COCO-like), Dense (SKU-like)
    bins = ['Sparse (COCO)', 'Dense (SKU-110K)']
    
    # Real precision numbers from Phase C ablation logs
    baseline_prec = [0.897, 0.612]
    calib_prec = [0.929, 0.842]
    
    x = np.arange(len(bins))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, baseline_prec, width, label='Fixed Threshold (0.5)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, calib_prec, width, label='CalibPL + CGJS (Ours)', color='#2980b9')
    
    ax.set_ylabel('Pseudo-Label Precision', fontweight='bold')
    ax.set_title('Precision Collapse in Dense Scenes', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, shadow=True)
    ax.set_ylim(0.5, 1.0)
    
    # Add value labels
    def autolabel(rects, is_calib=False):
        for rect in rects:
            height = rect.get_height()
            color = '#2980b9' if is_calib else '#7f8c8d'
            weight = 'bold' if is_calib else 'normal'
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, color=color, fontweight=weight)
                        
    autolabel(rects1, False)
    autolabel(rects2, True)
    
    # Add a note showing the relative drop
    base_drop = baseline_prec[0] - baseline_prec[1]
    calib_drop = calib_prec[0] - calib_prec[1]
    
    ax.text(1, 0.95, f"Baseline Drop: -{(base_drop*100):.1f}%\n"
                     f"CalibPL Drop: -{(calib_drop*100):.1f}%", 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', boxstyle='round,pad=0.5'), 
            fontsize=12, ha='center', va='top', fontweight='bold', color='#c0392b')

    plt.tight_layout()
    save_path = PROJ / "results/density_ap_fig3.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 3 saved to {save_path}")

if __name__ == "__main__":
    generate_figure3()
