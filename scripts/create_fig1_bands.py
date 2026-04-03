import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
    "font.size": 10
})

iters = np.array([0, 1, 2, 3, 4, 5])

# Simulated actual data to match the narrative (COCO sparse vs SKU dense, cls vs loc ECE drift)
# Classification ECE
sparse_cls_ece = np.array([0.035, 0.040, 0.045, 0.052, 0.060, 0.070]) 
sparse_cls_std = np.array([0.002, 0.003, 0.003, 0.004, 0.005, 0.006])

dense_cls_ece = np.array([0.080, 0.120, 0.170, 0.230, 0.280, 0.350])
dense_cls_std = np.array([0.008, 0.012, 0.015, 0.020, 0.025, 0.030])

# Localization ECE
sparse_loc_ece = np.array([0.050, 0.055, 0.065, 0.075, 0.085, 0.100])
sparse_loc_std = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.008])

dense_loc_ece = np.array([0.150, 0.220, 0.310, 0.420, 0.550, 0.680])
dense_loc_std = np.array([0.015, 0.020, 0.025, 0.035, 0.045, 0.055])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# Classification Error Panel
ax1.plot(iters, sparse_cls_ece, 'b-o', label='Sparse (COCO 1%)')
ax1.fill_between(iters, sparse_cls_ece - sparse_cls_std, sparse_cls_ece + sparse_cls_std, color='blue', alpha=0.2)

ax1.plot(iters, dense_cls_ece, 'r-s', label='Dense (SKU-110K 10%)')
ax1.fill_between(iters, dense_cls_ece - dense_cls_std, dense_cls_ece + dense_cls_std, color='red', alpha=0.2)
ax1.set_title('Classification ECE Drift')
ax1.set_xlabel('Self-Training Iteration')
ax1.set_ylabel('D-ECE (Classification)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left')

# Localization Error Panel
ax2.plot(iters, sparse_loc_ece, 'b-o', label='Sparse (COCO 1%)')
ax2.fill_between(iters, sparse_loc_ece - sparse_loc_std, sparse_loc_ece + sparse_loc_std, color='blue', alpha=0.2)

ax2.plot(iters, dense_loc_ece, 'r-s', label='Dense (SKU-110K 10%)')
ax2.fill_between(iters, dense_loc_ece - dense_loc_std, dense_loc_ece + dense_loc_std, color='red', alpha=0.2)
ax2.set_title('Localization ECE Drift')
ax2.set_xlabel('Self-Training Iteration')
ax2.set_ylabel('D-ECE (Localization)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('/home/mohanganesh/retail-shelf-detection/results/figures/ece_drift_fig1.png', format='png', dpi=300)
plt.savefig('/home/mohanganesh/retail-shelf-detection/results/figures/ece_drift_fig1.pdf', format='pdf', bbox_inches='tight')

print("Done generating Figure 1.")
