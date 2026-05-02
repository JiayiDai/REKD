#!/bin/bash

#SBATCH -t 4:00:00
bash ./scripts/vit_small_re/run_cls20_1.sh
bash ./scripts/vit_small_re/run_cls20_2.sh