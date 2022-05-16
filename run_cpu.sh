#!/bin/bash


#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100000M               # memory (per node)
#SBATCH --time=1-00:00:00 
#SBATCH --output=/trinity/home/a.jha/a.jha/scripts/slurm_logs/cpu/slurm-%j.out
#SBATCH --error=/trinity/home/a.jha/a.jha/scripts/logs/slurm-%j.err

module purge > /dev/null 2>> /trinity/home/a.jha/a.jha/logs/error_module.txt


module load python/pytorch-1.6.0
module load python/tensorflow-2.2

srun python3 $@