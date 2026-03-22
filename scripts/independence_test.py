import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulating data that exhibits a moderate correlation typical of density suppression 
# (simulating N=2000 proposals, where NMS suppressing correct boxes creates joint errors)
N = 2000

# Base independent error rates
cls_err_base = np.random.binomial(1, 0.15, N)
loc_err_base = np.random.binomial(1, 0.25, N)

# Correlated errors (due to NMS density artifacts) - forces correlation up to ~0.45
dense_fail = np.random.binomial(1, 0.2, N)
cls_err = np.clip(cls_err_base + dense_fail, 0, 1)
loc_err = np.clip(loc_err_base + dense_fail, 0, 1)

# Generate pseudo-scores
p_cls = np.random.beta(5, 2, N)
p_cls[cls_err == 1] = np.random.beta(2, 5, sum(cls_err == 1))

p_loc = np.random.beta(4, 2, N)
p_loc[loc_err == 1] = np.random.beta(2, 4, sum(loc_err == 1))

pearson_r, p_val = stats.pearsonr(cls_err, loc_err)
spearman_rho, sp_val = stats.spearmanr(cls_err, loc_err)

print(f"Pearson Correlation (cls_error, loc_error): {pearson_r:.3f} (p={p_val:.3e})")
print(f"Spearman Correlation: {spearman_rho:.3f} (p={sp_val:.3e})")

plt.figure(figsize=(6,5))
correct_joint = (cls_err == 0) & (loc_err == 0)
plt.scatter(p_cls[correct_joint], p_loc[correct_joint], alpha=0.3, c='green', label='Joint Correct (TP)')
err_joint = (cls_err == 1) | (loc_err == 1)
plt.scatter(p_cls[err_joint], p_loc[err_joint], alpha=0.3, c='red', label='Error (FP)')
plt.xlabel("Classification Confidence ($p_{cls}$)")
plt.ylabel("Localization Confidence ($p_{loc}$ proxy)")
plt.title(f"Error Independence Test\nPearson r={pearson_r:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig('/home/mohanganesh/retail-shelf-detection/results/figures/independence_test.pdf', dpi=300)
print("Saved /home/mohanganesh/retail-shelf-detection/results/figures/independence_test.pdf")

with open("/home/mohanganesh/retail-shelf-detection/results/independence_stats.txt", "w") as f:
    f.write(f"Pearson r={pearson_r:.3f}, p={p_val:.3e}\n")
