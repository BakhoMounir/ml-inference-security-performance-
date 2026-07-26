#!/bin/bash
#SBATCH --job-name=distilbert_full_pipeline
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# Submit with: sbatch scripts/run_full_pipeline.slurm
# Then disconnect -- this runs as a batch job, not interactive, so it
# survives SSH disconnects. Check on it later with squeue / the log files.

set -e   # stop immediately if any step fails, rather than silently
         # continuing to benchmark a broken/missing checkpoint

mkdir -p logs results models

source ~/.bashrc
conda activate your_env_name   # <-- EDIT: replace with your actual env name

cd "$SLURM_SUBMIT_DIR/src"

CHECKPOINT=../models/distilbert_imdb_baseline.pt

echo "=== $(date) : job started on $(hostname) ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

if [ -f "$CHECKPOINT" ]; then
    echo "=== Checkpoint already exists at $CHECKPOINT -- skipping training ==="
else
    echo "=== No checkpoint found -- training baseline (1 epoch, ~10 min on A40) ==="
    python -m train
fi

echo "=== Step 1: benchmark harness validation (small, fast) ==="
python -m benchmark.benchmark \
    --batch-sizes 1 2 \
    --num-warmup 3 \
    --num-iters 5 \
    --tag fp32_validation \
    --output ../results/benchmark_validation.csv

echo "=== Step 2: full FP32 batch-size sweep ==="
python -m benchmark.benchmark \
    --batch-sizes 1 2 4 8 16 32 64 \
    --checkpoint "$CHECKPOINT" \
    --num-warmup 10 \
    --num-iters 50 \
    --tag fp32 \
    --output ../results/benchmark_fp32.csv

echo "=== $(date) : job finished ==="
