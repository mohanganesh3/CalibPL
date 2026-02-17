#!/bin/bash
# ============================================================
# BMVC 2026: Launch CalibPL SKU-110K Seeds 43 + 44 (Multi-seed completion)
# GPU 3: seeds 43 → 44 (sequential)
# ============================================================

set -e
PROJ="/home/mohanganesh/retail-shelf-detection"
LOG_DIR="$PROJ/logs/sku_calibpl_bmvc"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " BMVC CalibPL SKU-110K Multi-Seed Completion"
echo " GPU 3: Seeds 43 + 44"
echo " $(date)"
echo "============================================================"

nohup bash -c "
  echo '[GPU3] Starting CalibPL SKU10 Seed 43...'
  PYTHONUNBUFFERED=1 python $PROJ/scripts/run_calibpl_frcnn_true.py \
    --dataset sku10 \
    --seed 43 \
    --gpu 3 \
    --rel-thresh 0.6 \
    --cgjs-thresh 0.5 \
    --iterations 3 \
    --batch-size 4 \
    2>&1
  echo '[GPU3] SKU Seed 43 DONE. Starting Seed 44...'
  PYTHONUNBUFFERED=1 python $PROJ/scripts/run_calibpl_frcnn_true.py \
    --dataset sku10 \
    --seed 44 \
    --gpu 3 \
    --rel-thresh 0.6 \
    --cgjs-thresh 0.5 \
    --iterations 3 \
    --batch-size 4 \
    2>&1
  echo '[GPU3] SKU Seed 44 DONE.'
" > "$LOG_DIR/gpu3_sku_seeds43_44.log" 2>&1 &
GPU3_PID=$!

echo "[GPU3] PID=$GPU3_PID | SKU seeds 43+44 launched"
echo "Monitor: tail -f $LOG_DIR/gpu3_sku_seeds43_44.log"
