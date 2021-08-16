#!/bin/bash
#
#SBATCH --job-name=arithmetic_mul
#SBATCH --output=train_logs/arithmetic_mul_out.txt
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=single
#SBATCH --time=2-0:00:00
#SBATCH -e train_logs/arithmetic_mul_error.txt
# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:"$PATH"
LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/:"$LD_LIBRARY_PATH"
PATH=/usr/local/cuda/bin/:"$PATH" 

# JOB STEPS (example: write hostname to output file, and wait 1 minute)
#srun echo $CUDA_VISIBLE_DEVICES
srun python code/main.py --config configs/arithmetic__mul_hard.yaml


