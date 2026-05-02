#!/bin/bash

# Define the command
CMD="python3 run/main_inference.py \
    --cuda \
    --dataset cifar100 \
    --model_form google/vit-base-patch16-224 \
    --rand_seed 2031 \
    --id vit_re_cifar20 \
    --save_dir saved \
    --dropout 0.1 \
    --test \
    --get_rationales"

echo "Starting inference..."

# Run directly in the foreground
# We redirect stdout/stderr to the log file as you intended
$CMD > run_re_cifar100.log 2>&1

echo "Job finished."