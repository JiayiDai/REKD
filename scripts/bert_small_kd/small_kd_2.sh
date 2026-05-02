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
    --model_form prajjwal1/bert-small \
    --model_form_t bert-base-uncased \
    --rand_seed "$SEED" \
    --id "small_kd_5e-1_10" \
    --id_t "bert_re_5e-52035" \
    --train \
    --test \
    --get_rationales \
    --total_features 256 \
    --target_sparsity 0.10 \
    --select_lambda 1e-4 \
    --kd_r_lambda 5e-1 > /dev/null 2>&1

echo "Job for seed $SEED finished."