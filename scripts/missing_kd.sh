#!/bin/bash

#SBATCH -t 4:00:00
bash ./scripts/vit_small_kd/run1.sh
bash ./scripts/vit_small_kd/run2.sh

bash ./scripts/vit_small_kd/run20_1.sh

bash ./scripts/vit_tiny_kd/run20_1.sh
bash ./scripts/vit_tiny_kd/run20_2.sh