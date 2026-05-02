#!/bin/bash
#SBATCH -A aip-rgoebel
#SBATCH -p gpubase_interac
#SBATCH -J read_re_images
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -o /dev/null

module purge
source ~/.bashrc
unset PYTHONPATH
module load python/3.11.5
module load cuda/12.6
module load gcc arrow/21.0.0
export HF_HUB_DISABLE_XET=1

source ~/projects/aip-rgoebel/jdai/env_re/bin/activate

bash ./scripts/run_rationales/run_inference_cifar10.sh

deactivate