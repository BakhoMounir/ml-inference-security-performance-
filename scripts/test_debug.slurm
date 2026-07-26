#!/bin/bash
#SBATCH --job-name=gpu_access_test
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "Running on $(hostname)"
nvidia-smi
