#!/bin/bash
#SBATCH --job-name=8vs2_rs_100
#SBATCH --output=8vs2_rs_100.out
#SBATCH --error=8vs2_rs_100.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 3
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH -p long
#SBATCH --constraint titan_x

# pip install torch torchvision
srun bash experiments/8vs2/cifar2-tentask_rs_100.sh
