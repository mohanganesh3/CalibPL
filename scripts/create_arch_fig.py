import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Disable external LaTeX, use mathtext instead
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
})

def create_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    
    # helper for drawing boxes
    def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=10, weight='normal'):
        box = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, weight=weight, zorder=3)
        return box
        
    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='black'), zorder=1)
            
    # Define colors
    c_img = '#f0f0f0'
    c_teacher = '#d4e6f1'
    c_student = '#d4e6f1'
    c_calib = '#d5f5e3'
    c_cgjs = '#fdebd0'
    c_gate = '#e8daef'
    
    # 1. Unlabeled Image (left)
    draw_box(ax, 2, 22, 10, 6, "Unlabeled\nImage", c_img, 'gray')
    
    # 2. Teacher Model
    draw_box(ax, 16, 17, 10, 16, "Teacher\nModel", c_teacher, 'gray', weight='bold')
    draw_arrow(ax, 12, 25, 16, 25)
    
    # 3. Target proposals
    ax.text(31, 31, r"Predictions $\{\hat{b}_i, \hat{s}_i, y_i\}$", ha='center', va='bottom', fontsize=10)
    draw_arrow(ax, 26, 25, 36, 25)
    
    # Fork #1: Dual Recalibration Branch (Top)
    draw_arrow(ax, 36, 25, 36, 38)
    draw_arrow(ax, 36, 38, 40, 38)
    
    # Classification Calibrator
    draw_box(ax, 40, 42, 18, 6, "Classification Calibrator\n(Isotonic Regression)", c_calib, 'darkgreen')
    ax.text(49, 40.5, r"$\hat{s}_{cls} \to c_{cls}$", ha='center', va='center', fontsize=10)
    
    # Add a tiny plot to represent Isotonic Regression inside the box
    ax_ins1 = fig.add_axes([0.48, 0.81, 0.05, 0.07], facecolor='none') 
    ax_ins1.axis('off')
    x = np.linspace(0, 1, 20)
    y = 1 / (1 + np.exp(-10*(x-0.5)))
    y = np.sort(y + np.random.normal(0, 0.05, 20))
    y = np.clip(y, 0, 1)
    ax_ins1.plot(x, x, 'k--', alpha=0.5, lw=0.5)
    ax_ins1.step(x, y, color='darkgreen', lw=1)
    
    # Localization Calibrator
    draw_box(ax, 40, 32, 18, 6, "Localization Calibrator\n(Isotonic Regression)", c_calib, 'darkgreen')
    ax.text(49, 30.5, r"$\hat{s}_{loc} \to c_{loc}$", ha='center', va='center', fontsize=10)

    # tiny plot for Loc
    ax_ins2 = fig.add_axes([0.48, 0.61, 0.05, 0.07], facecolor='none') 
    ax_ins2.axis('off')
    ax_ins2.plot(x, x, 'k--', alpha=0.5, lw=0.5)
    ax_ins2.step(x, y, color='darkgreen', lw=1)

    draw_arrow(ax, 36, 38, 36, 45)
    draw_arrow(ax, 36, 45, 40, 45)
    draw_arrow(ax, 36, 38, 40, 35) # Fixed connection
    
    # Fork #2: CGJS Module (Bottom)
    draw_arrow(ax, 21, 17, 21, 9)
    draw_arrow(ax, 21, 9, 28, 9)
    
    draw_box(ax, 28, 6, 14, 6, "Strong\nAugmentations ($T_a$)", c_img, 'gray')
    draw_arrow(ax, 42, 9, 46, 9)
    
    draw_box(ax, 46, 5, 20, 8, "CGJS Module\n(Class-Geometry Stability)", c_cgjs, 'darkorange')
    ax.text(56, 3, r"outputs $\rho_i \in [0, 1]$", ha='center', va='center', fontsize=10)
    
    # Center: AND Gate
    draw_box(ax, 70, 20, 16, 10, "Reliability Gate\n(AND Gate)", c_gate, 'purple')
    
    # Connect Calibrators to Gate
    draw_arrow(ax, 58, 45, 66, 45)
    draw_arrow(ax, 66, 45, 66, 27)
    draw_arrow(ax, 66, 27, 70, 27)
    
    draw_arrow(ax, 58, 35, 68, 35)
    draw_arrow(ax, 68, 35, 68, 25)
    draw_arrow(ax, 68, 25, 70, 25)
    
    # Connect CGJS to Gate
    draw_arrow(ax, 66, 9, 78, 9)
    draw_arrow(ax, 78, 9, 78, 20)
    
    # P-labels to Student
    ax.text(82, 32, "Selected\nPseudo-Labels", ha='center', va='center', fontsize=10)
    draw_arrow(ax, 86, 25, 92, 25)
    
    draw_box(ax, 92, 17, 8, 16, "Student\nModel", c_student, 'gray', weight='bold')
    
    ax.text(50, -2, "*Unlabeled data is used for student training only", ha='center', va='center', fontsize=10, style='italic')
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Architecture diagram saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram("/home/mohanganesh/retail-shelf-detection/results/figures/figure2_architecture.pdf")
