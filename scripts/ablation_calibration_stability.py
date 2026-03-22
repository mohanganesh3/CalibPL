"""
Ablation Study: Calibration Stability vs. Labeled Sample Size.
Goal: To quantify the risk of over-fitting isotonic regression on small COCO 1% splits.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

def calculate_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_idx = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if np.any(bin_idx):
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_idx) / len(y_prob)
    return ece

def run_ablation():
    # Simulate a "Ground Truth" reliability curve (sigmoidal)
    def gt_reliability(conf):
        return 1 / (1 + np.exp(-10 * (conf - 0.5)))

    sample_sizes = [50, 100, 250, 500, 1000, 2000]
    trials = 10
    results = []

    for n in sample_sizes:
        trial_ece = []
        for _ in range(trials):
            # Generate synthetic "Validation Set"
            val_conf = np.random.uniform(0, 1, n)
            val_prob = gt_reliability(val_conf)
            val_true = (np.random.random(n) < val_prob).astype(int)

            # Fit calibrator
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(val_conf, val_true)

            # Evaluate on large "Unlabeled Pool" proxy
            pool_conf = np.random.uniform(0, 1, 5000)
            pool_true = (np.random.random(5000) < gt_reliability(pool_conf)).astype(int)
            pool_calibrated = iso.predict(pool_conf)

            ece = calculate_ece(pool_true, pool_calibrated)
            trial_ece.append(ece)
        
        results.append({
            'sample_size': n,
            'avg_ece': np.mean(trial_ece),
            'std_ece': np.std(trial_ece)
        })

    df = pd.DataFrame(results)
    print(df)
    
    # Save results for LaTeX
    df.to_csv('results/ablation_cal_stability.csv', index=False)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(df['sample_size'], df['avg_ece'], yerr=df['std_ece'], marker='o', capsize=5)
    plt.title('Isotonic Calibration Stability vs. Sample Size')
    plt.xlabel('Number of Labeled Boxes (Val Split)')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.grid(True)
    plt.savefig('results/figures/ablation_cal_stability.png')
    print("Ablation results saved to results/figures/ablation_cal_stability.png")

if __name__ == "__main__":
    import os
    os.makedirs('results/figures', exist_ok=True)
    run_ablation()
