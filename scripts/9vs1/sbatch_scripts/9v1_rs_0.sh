#!/bin/bash
#SBATCH --job-name=9vs1_rs_0
#SBATCH --output=9vs1_rs_0.out
#SBATCH --error=9vs1_rs_0.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH -p short
#SBATCH --constraint titan_x

# pip install torch torchvision
srun bash experiments/9vs1/cifar2-tentask_rs_0.sh