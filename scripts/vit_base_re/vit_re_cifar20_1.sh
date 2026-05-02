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
    --dataset cifar100 \
    --model_form google/vit-base-patch16-224 \
    --rand_seed "$SEED" \
    --id "vit_re_cifar20" \
    --weight_decay 1e-3 \
    --dropout 0.1 \
    --train \
    --test \
    --get_rationales \
    --total_features 196 \
    --target_sparsity 0.15 \
    --select_lambda 5e-3 > /dev/null 2>&1

echo "Job for seed $SEED finished."