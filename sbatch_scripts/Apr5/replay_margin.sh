#!/bin/bash
#SBATCH --job-name=replay_margin
#SBATCH --output=replay_margin.out
#SBATCH --error=replay_margin.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH -p short
#SBATCH --constraint titan_x

pip install torch torchvision
export PATH=/nethome/vgutta7/miniconda3/bin:$PATH
srun bash experiments/10vs10/April5/replay_margin.sh