#!/bin/bash
#SBATCH --job-name=9vs1_rs_100
#SBATCH --output=9vs1_rs_100.out
#SBATCH --error=9vs1_rs_100.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 3
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH -p debug
#SBATCH --constraint titan_x

# pip install torch torchvision
srun bash experiments/9vs1/cifar2-tentask_rs_100.sh