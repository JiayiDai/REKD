#!/bin/bash

#SBATCH -t 4:00:00

bash ./scripts/vit_base_re/run_cls4.sh
bash ./scripts/vit_base_re/run_cls20_1.sh
