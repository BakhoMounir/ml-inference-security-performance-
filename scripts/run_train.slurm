#!/bin/bash
#SBATCH --job-name=distilbert_train
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
mkdir -p logs results models

source ~/.bashrc
conda activate base

cd "$SLURM_SUBMIT_DIR/src"

echo "=== $(date) : job started on $(hostname) ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CHECKPOINT=../models/distilbert_imdb_baseline.pt
if [ -f "$CHECKPOINT" ]; then
    echo "Checkpoint already exists -- skipping training."
else
    python -m train
fi

echo "=== $(date) : job finished ==="
