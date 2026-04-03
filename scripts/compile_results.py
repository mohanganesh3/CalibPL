#!/usr/bin/env python3
"""
Results Compiler for CalibPL
Parses all experiment JSON logs and format them into a Publication-Ready LaTeX table.
"""

import os
import json
import numpy as np
from pathlib import Path

def generate_latex_table(results_dir):
    methods = {
        'supervised_s42': 'Supervised Baseline (1%)',
        'pseudo50_s42': 'Pseudo-Label (\\tau=0.5)',
        'calibpl_s42': 'CalibPL (Ours - Conf)',
        'calibpl_pss_s42': 'CalibPL + PSS (Ours - Full)'
    }
    
    table = []
    table.append("\\begin{table}[h]")
    table.append("\\centering")
    table.append("\\caption{Semi-Supervised Object Detection Performance on COCO 1\\% subset.}")
    table.append("\\label{tab:main_results}")
    table.append("\\begin{tabular}{l c c c c}")
    table.append("\\toprule")
    table.append("Method & \\multicolumn{3}{c}{mAP50 across Iterations} & Best mAP50 \\\\")
    table.append(" & Iter 1 & Iter 2 & Iter 3 & \\\\")
    table.append("\\midrule")
    
    log_dir = Path(results_dir) / "baselines" # or results/calibpl_v3 etc
    
    # We will search the logs
    raw_dirs = [
        Path(results_dir) / "coco1pct" / "supervised",
        Path(results_dir) / "calibpl_v3"
    ]
    
    # Dummy parsing logic for now since we know the directory structure
    for key, name in methods.items():
        # find the associated json summary
        # we know calibpl scripts output to results/calibpl_v3/{method}_seed42_coco1pct/summary.json
        # supervised outputs to results/coco1pct/supervised/train_seed42/results.csv
        best_map = 0.0
        iters = ["-", "-", "-"]
        
        if key == 'supervised_s42':
            csv_path = Path(results_dir) / "coco1pct" / "supervised" / "train_seed42" / "results.csv"
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if 'metrics/mAP50(B)' in df.columns:
                    best_map = df['metrics/mAP50(B)'].max()
                    iters = [f"{best_map:.3f}", "-", "-"]
        else:
            method_name = key.replace('_s42', '')
            if method_name == 'pseudo50':
                method_name = 'pseudo_label'
            
            json_path = Path(results_dir) / "calibpl_v3" / f"{method_name}_seed42_coco1pct" / "summary.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                    maps = [float(r['map50']) for r in data.get('iterations', [])]
                    if len(maps) > 0:
                        best_map = max(maps)
                        for i in range(min(3, len(maps))):
                            iters[i] = f"{maps[i]:.3f}"
                            
        table.append(f"{name} & {iters[0]} & {iters[1]} & {iters[2]} & \\textbf{{{best_map:.3f}}} \\\\")
        
    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    
    return "\n".join(table)

if __name__ == "__main__":
    PROJ_DIR = "/home/mohanganesh/retail-shelf-detection"
    RESULTS_DIR = os.path.join(PROJ_DIR, "results")
    latex_code = generate_latex_table(RESULTS_DIR)
    
    out_file = os.path.join(RESULTS_DIR, "table_results.tex")
    with open(out_file, 'w') as f:
        f.write(latex_code)
    print(latex_code)
    print(f"Results table written to {out_file}")
