#!/bin/bash

# Capture the seed passed from Script 1
SEED=$1

if [ -z "$SEED" ]; then
  echo "Error: No seed provided."
  exit 1
fi

echo "Starting training with Seed: $SEED"

# No need for CUDA_VISIBLE_DEVICES or manual loops. 
# SLURM assigns the specific GPU to this job automatically.

python3 run/main.py \
    --cuda \
    --model_form prajjwal1/bert-small \
    --rand_seed "$SEED" \
    --id "small_re_1e-4_095" \
    --train \
    --test \
    --get_rationales \
    --total_features 256 \
    --target_sparsity 0.095 \
    --select_lambda 1e-4 > /dev/null 2>&1

echo "Job for seed $SEED finished."