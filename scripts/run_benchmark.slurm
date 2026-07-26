#!/bin/bash
#SBATCH --job-name=distilbert_bench
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1   # confirmed via `sinfo`: --gres=gpu:1 alone is not
                            # sufficient on this cluster, must name the GPU type
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

# Batch (non-interactive) job so it survives SSH disconnects -- avoids the
# queued-interactive-session problem noted in your HPC notes.

set -e

mkdir -p logs results

source ~/.bashrc
conda activate your_env_name   # <-- EDIT: replace with your actual env name

cd "$SLURM_SUBMIT_DIR"          # project ROOT -- do NOT cd into src/, since
                                 # the code imports as `from src.model...`,
                                 # which only resolves if src/ is a
                                 # subdirectory of the current working dir
export HF_HUB_OFFLINE=1

echo "=== Step 1: small validation run (should finish in seconds) ==="
python -m src.benchmark.benchmark \
    --batch-sizes 1 2 \
    --num-warmup 3 \
    --num-iters 5 \
    --tag fp32_validation \
    --output results/benchmark_validation.csv

echo "=== Step 2: full batch-size sweep (only runs if Step 1 succeeded) ==="
python -m src.benchmark.benchmark \
    --batch-sizes 1 2 4 8 16 32 64 \
    --checkpoint models/distilbert_imdb_baseline.pt \
    --num-warmup 10 \
    --num-iters 50 \
    --tag fp32 \
    --output results/benchmark_fp32.csv

echo "Done."
