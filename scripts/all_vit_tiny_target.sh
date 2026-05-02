#!/bin/bash

#SBATCH -t 4:00:00
bash ./scripts/target_performance_points/vit_tiny_re/run_cls0.sh
bash ./scripts/target_performance_points/vit_tiny_re/run_cls1.sh
bash ./scripts/target_performance_points/vit_tiny_re/run_cls2.sh
bash ./scripts/target_performance_points/vit_tiny_re/run_cls3.sh
bash ./scripts/target_performance_points/vit_tiny_re/run_cls4.sh
bash ./scripts/target_performance_points/vit_tiny_re/run_cls5.sh
bash ./scripts/target_performance_points/vit_tiny_re/run_cls6.sh
bash ./scripts/target_performance_points/vit_tiny_re/run_cls7.sh