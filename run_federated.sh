#!/bin/bash

# Federated learning (FedAvg) variant of the LiRA pipeline.
# Each shadow model is trained with a server + clients instead of centralized training.

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASET="cifar10"
GLOBAL_EPOCHS=10
LOCAL_EPOCHS=1
NUM_CLIENTS=30
CLIENT_FRACTION=0.1
PARTITION="dirichlet"
ALPHA=0.5
N_SHADOWS=4
N_QUERIES=2

echo "========================================"
echo "Starting Federated LiRA Pipeline for: $DATASET"
echo "  FedAvg: $NUM_CLIENTS clients, $CLIENT_FRACTION/round, $LOCAL_EPOCHS local epochs, $GLOBAL_EPOCHS global rounds"
echo "  Partition: $PARTITION (alpha=$ALPHA)"
echo "========================================"

# 0. Prepare the selected dataset before training starts
if [ "$DATASET" = "cifar10" ]; then
    echo "[0/4] Preparing CIFAR-10 dataset..."
    python - <<'PY'
from dataset_utils import resolve_dataset_root
resolve_dataset_root('cifar10')
print('CIFAR-10 dataset ready')
PY
elif [ "$DATASET" = "malimg" ]; then
    echo "[0/4] Preparing MalImg dataset..."
    python - <<'PY'
from dataset_utils import resolve_dataset_root
resolve_dataset_root('malimg')
print('MalImg dataset ready')
PY
elif [ "$DATASET" = "malnet" ]; then
    echo "[0/4] Preparing MalNet dataset..."
    python - <<'PY'
from dataset_utils import resolve_dataset_root
resolve_dataset_root('malnet')
print('MalNet dataset ready')
PY
else
    echo "Unsupported dataset: $DATASET"
    exit 1
fi

# 1. Train all shadow models with federated learning
echo "[1/4] Training Shadow Models (Federated)..."
for ((id=0; id<N_SHADOWS; id++))
do
    echo " -> Training Shadow Model $id"
    python train.py --dataset $DATASET --federated \
        --global_epochs $GLOBAL_EPOCHS --local_epochs $LOCAL_EPOCHS \
        --num_clients $NUM_CLIENTS --client_fraction $CLIENT_FRACTION \
        --partition $PARTITION --alpha $ALPHA \
        --shadow_id $id --n_shadows $N_SHADOWS
done

# 2. Run Inference
echo "[2/4] Running Inference..."
python inference.py --dataset $DATASET --n_queries $N_QUERIES

# 3. Calculate Scores
echo "[3/4] Calculating Scores..."
python score.py --dataset $DATASET

# 4. Generate FPR/TPR Plots
echo "[4/4] Generating Plots..."
python plot.py --dataset $DATASET
