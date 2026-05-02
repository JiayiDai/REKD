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

python3 run/main_kd.py \
    --cuda \
    --dataset cifar100 \
    --model_form WinKawaks/vit-tiny-patch16-224 \
    --model_form_t google/vit-base-patch16-224 \
    --rand_seed "$SEED" \
    --id "vit_tiny_kd_cifar20_alpha7" \
    --id_t "vit_re_cifar202031" \
    --weight_decay 1e-3 \
    --dropout 0.1 \
    --train \
    --test \
    --get_rationales \
    --total_features 196 \
    --target_sparsity 0.150 \
    --select_lambda 1e-2 \
    --alpha_re 0.7 \
    --kd_r_lambda 5e-1 > /dev/null 2>&1

echo "Job for seed $SEED finished."