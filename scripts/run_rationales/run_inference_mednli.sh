#!/bin/bash

# Define the command
CMD="python3 run/main_inference.py \
    --cuda \
    --dataset mednli \
    --model_form bert-base-uncased \
    --rand_seed 2035 \
    --id bert_re_mednli_7_5e_4 \
    --save_dir saved \
    --batch_size 32 \
    --test \
    --get_rationales"

echo "Starting inference..."

# Run directly in the foreground
# We redirect stdout/stderr to the log file as you intended
$CMD > run_re_t.log 2>&1

echo "Job finished."