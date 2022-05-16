#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000M               # memory (per node)
#SBATCH --time=6-00:00:00 
#SBATCH --output=/trinity/home/a.jha/a.jha/scripts/slurm_logs/gpu/slurm-%j.out

module purge > /dev/null 2>> /trinity/home/a.jha/a.jha/logs/error_module.txt

module load python/pytorch-1.6.0-3.7

srun python3 $@
