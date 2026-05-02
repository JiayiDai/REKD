#!/bin/bash

#SBATCH -t 2:00:00
bash ./scripts/bert_mini_kd_fixed_alpha/run_mini_kd1.sh
bash ./scripts/bert_mini_kd_fixed_alpha/run_mini_kd2.sh
