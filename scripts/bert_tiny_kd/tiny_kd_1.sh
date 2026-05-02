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
    --model_form prajjwal1/bert-tiny \
    --model_form_t bert-base-uncased \
    --rand_seed "$SEED" \
    --id "tiny_kd_5e-1_09" \
    --id_t "bert_re12031" \
    --train \
    --test \
    --get_rationales \
    --total_features 256 \
    --target_sparsity 0.09 \
    --select_lambda 2e-4 \
    --kd_r_lambda 5e-1 > /dev/null 2>&1

echo "Job for seed $SEED finished."