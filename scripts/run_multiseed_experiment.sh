#!/bin/bash
# ==============================================================================
# MULTI-SEED STATISTICAL VALIDATION (Phase 3)
# ==============================================================================
# Runs the 3-strategy × 5-iteration experiment for multiple seeds.
# Uses setsid-safe architecture to survive SSH disconnects.
# All crash-prevention measures from the Grand Experiment are inherited:
#   - workers=0 (no DataLoader multiprocessing deadlocks)
#   - amp=False, half=False (K80 BF16 compatibility)
#   - SGD optimizer (no auto-optimizer BF16 selection)
#   - gc.collect() + torch.cuda.empty_cache() between phases
#   - sys.exit(0) for DDP worker termination
#   - stream=True, workers=0 for MC Dropout inference
# ==============================================================================

set -e

# K80-optimized hyperparameters (proven stable over 15+ hours)
ITERATIONS=5
EPOCHS_PER_ITER=10
MAX_PSEUDO=300
T=5
TRAIN_DEVICE="0,1,2,3"
INFER_DEVICE="0"
ALPHA=0.5

# Accept seed as argument, default to 123
SEED=${1:-123}
LOG_FILE="results/grand_experiment_seed${SEED}.log"

echo "======================================================================" | tee -a $LOG_FILE
echo "MULTI-SEED EXPERIMENT — SEED=$SEED" | tee -a $LOG_FILE
echo "Start Time: $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE

# 1. Baseline: Naive Confidence
echo ">>> [1/3] STRATEGY: CONFIDENCE (seed=$SEED)" | tee -a $LOG_FILE
python -u scripts/run_calibcotrain.py \
    --method confidence \
    --iterations $ITERATIONS \
    --epochs-per-iter $EPOCHS_PER_ITER \
    --max-pseudo $MAX_PSEUDO \
    --seed $SEED \
    --train-device $TRAIN_DEVICE \
    --infer-device $INFER_DEVICE \
    >> $LOG_FILE 2>&1

# 2. Epistemic Uncertainty Only
echo ">>> [2/3] STRATEGY: EPISTEMIC (seed=$SEED)" | tee -a $LOG_FILE
python -u scripts/run_calibcotrain.py \
    --method epistemic \
    --iterations $ITERATIONS \
    --epochs-per-iter $EPOCHS_PER_ITER \
    --max-pseudo $MAX_PSEUDO \
    --mc-T $T \
    --seed $SEED \
    --train-device $TRAIN_DEVICE \
    --infer-device $INFER_DEVICE \
    >> $LOG_FILE 2>&1

# 3. Combined Uncertainty (α=0.5)
echo ">>> [3/3] STRATEGY: COMBINED α=$ALPHA (seed=$SEED)" | tee -a $LOG_FILE
python -u scripts/run_calibcotrain.py \
    --method combined \
    --alpha $ALPHA \
    --iterations $ITERATIONS \
    --epochs-per-iter $EPOCHS_PER_ITER \
    --max-pseudo $MAX_PSEUDO \
    --mc-T $T \
    --seed $SEED \
    --train-device $TRAIN_DEVICE \
    --infer-device $INFER_DEVICE \
    >> $LOG_FILE 2>&1

echo "======================================================================" | tee -a $LOG_FILE
echo "SEED $SEED COMPLETE!" | tee -a $LOG_FILE
echo "End Time: $(date)" | tee -a $LOG_FILE
echo "======================================================================" | tee -a $LOG_FILE
