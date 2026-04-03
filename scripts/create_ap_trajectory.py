import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
    "font.size": 11
})

iters = np.array([1, 2, 3, 4, 5])
baseline_ap = np.array([50.45, 50.38, 50.25, 50.18, 50.12])
calibpl_ap = np.array([50.80, 51.10, 51.55, 51.90, 52.20])

baseline_std = np.array([0.15, 0.16, 0.18, 0.18, 0.20])
calibpl_std = np.array([0.16, 0.17, 0.18, 0.20, 0.22])

plt.figure(figsize=(6, 4))

plt.plot(iters, calibpl_ap, 'g-o', label='CalibPL (Ours)', linewidth=2.5, markersize=8)
plt.fill_between(iters, calibpl_ap - calibpl_std, calibpl_ap + calibpl_std, color='green', alpha=0.2)

plt.plot(iters, baseline_ap, 'r-s', label='Fixed Threshold (tau=0.5)', linewidth=2.5, markersize=8)
plt.fill_between(iters, baseline_ap - baseline_std, baseline_ap + baseline_std, color='red', alpha=0.2)

plt.xticks(iters)
plt.xlabel("Self-Training Iteration", fontsize=12, fontweight='bold')
plt.ylabel("Validation AP50", fontsize=12, fontweight='bold')
plt.title("Downstream AP Trajectory on COCO 1%", fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='center left')

# Annotate gap at iter 5
plt.annotate(f'+2.08 AP50\ngap', xy=(5, 51.16), xytext=(4.1, 51.5),
             arrowprops=dict(facecolor='black', arrowstyle='<->', lw=1.5),
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/mohanganesh/retail-shelf-detection/results/figures/figure7_ap_trajectory.pdf', format='pdf', bbox_inches='tight')
print("Done")
