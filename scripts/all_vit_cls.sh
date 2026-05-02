#!/bin/bash

#SBATCH -t 4:00:00


bash ./scripts/vit_base_cls/run_cls.sh
bash ./scripts/vit_small_cls/run_cls.sh
bash ./scripts/vit_tiny_cls/run_cls.sh

bash ./scripts/vit_base_cls/run_cls20.sh
bash ./scripts/vit_small_cls/run_cls20.sh
bash ./scripts/vit_tiny_cls/run_cls20.sh