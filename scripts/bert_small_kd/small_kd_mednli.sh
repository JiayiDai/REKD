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
    --dataset mednli \
    --cuda \
    --model_form prajjwal1/bert-small \
    --model_form_t bert-base-uncased \
    --rand_seed "$SEED" \
    --id "small_kd_mednli_7_2035t" \
    --id_t "bert_re_mednli_7_5e_42035" \
    --train \
    --test \
    --get_rationales \
    --total_features 128 \
    --target_sparsity 0.07 \
    --select_lambda 5e-4 \
    --kd_r_lambda 5e-1 > /dev/null 2>&1

echo "Job for seed $SEED finished."