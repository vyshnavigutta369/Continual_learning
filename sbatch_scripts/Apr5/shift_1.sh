#!/bin/bash
#SBATCH --job-name=shift_1
#SBATCH --output=shift_1.out
#SBATCH --error=shift_1.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH -p debug

pip install torch torchvision
export PATH=/nethome/vgutta7/miniconda3/bin:$PATH
srun bash experiments/10vs10/April5/shift_1.sh