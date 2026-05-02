#!/bin/bash

#SBATCH -t 4:00:00

bash ./scripts/target_performance_points/vit_base_re/run_cls4.sh
bash ./scripts/target_performance_points/vit_base_re/run_cls0.sh

