#!/bin/bash
#SBATCH --job-name=8vs2_rs_10
#SBATCH --output=8vs2_rs_10.out
#SBATCH --error=8vs2_rs_10.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 3
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH -p short
#SBATCH --constraint titan_x

# pip install torch torchvision
srun bash experiments/8vs2/cifar2-tentask_rs_10.sh
