#!/bin/bash

# List of random seeds to use
seeds=(2026 2027 2028 2029 2030 2031 2032 2033 2034 2035)
#seeds=(2026)

for seed in "${seeds[@]}"; do
    echo "Submitting job for seed: $seed"
    
    # We pass the ${seed} variable into the heredoc below
    sbatch << EOF
#!/bin/bash
#SBATCH -A aip-rgoebel
#SBATCH -p gpubase_interac
#SBATCH -J kd_vit_tiny10_r_dist_${seed}
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1       # Changed to 1 GPU per job (since we are running 1 seed per job)
#SBATCH -c 4
#SBATCH --mem=50G          # Reduced memory (200G is likely overkill for 1 seed, adjust if needed)
#SBATCH -o /dev/null  # Good practice to save logs, %x adds job name

module purge
source ~/.bashrc
unset PYTHONPATH
module load python/3.11.5
module load cuda/12.6
module load gcc arrow/21.0.0
export HF_HUB_DISABLE_XET=1


source ~/projects/aip-rgoebel/jdai/env_re/bin/activate

# Call the second script and pass the seed as an argument
# Make sure the path below matches where you save Script 2
bash ./rebuttal_scripts/parts_dist/vit_tiny/vit_kd_cifar10_r_dist.sh ${seed}

deactivate
EOF
sleep 1m
done