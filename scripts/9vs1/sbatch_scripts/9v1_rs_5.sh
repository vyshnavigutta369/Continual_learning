#!/bin/bash
#SBATCH --job-name=9vs1_rs_5
#SBATCH --output=9vs1_rs_5.out
#SBATCH --error=9vs1_rs_5.err
#SBATCH --gres gpu:1
#SBATCH -p short

# pip install torch torchvision
srun bash experiments/9vs1/cifar2-tentask_rs_5.sh