#!/bin/bash
#SBATCH --job-name=ora
#SBATCH --output=ora.out
#SBATCH --error=ora.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH -p short

pip install torch torchvision
export PATH=/nethome/vgutta7/miniconda3/bin:$PATH
srun bash scripts/1.sh