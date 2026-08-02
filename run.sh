#!/bin/bash

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Set this to "malimg" or "cifar10" to toggle the pipeline

DATASET="malimg"
EPOCHS=1
N_SHADOWS=4

echo "========================================"
echo "Starting LiRA Pipeline for: $DATASET"
echo "========================================"

# 1. Train all 16 shadow models
echo "[1/4] Training Shadow Models..."
for ((id=0; id<N_SHADOWS; id++))
do
    echo " -> Training Shadow Model $id"
    python3 train.py --dataset $DATASET --epochs $EPOCHS --shadow_id $id --n_shadows $N_SHADOWS
done

# 2. Run Inference
echo "[2/4] Running Inference..."
python3 inference.py --dataset $DATASET

# 3. Calculate Scores
echo "[3/4] Calculating Scores..."
python3 score.py --dataset $DATASET

# 4. Generate FPR/TPR Plots
echo "[4/4] Generating Plots..."
python3 plot.py --dataset $DATASET



