#!/bin/bash
#SBATCH --job-name=distilbert_cpu_bench
#SBATCH --partition=i64m512u   # confirmed via sinfo: 106 nodes, 64 CPUs
                                # each, no GRES, 7-day limit -- much lower
                                # contention risk than squeezing into
                                # debug's 5 CPU-only nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cpu_bench_%j.out
#SBATCH --error=logs/cpu_bench_%j.err

# No --gres=gpu:... here on purpose: this is a CPU-only job.

set -e   # stop immediately on first failure, don't silently benchmark
         # a broken/missing checkpoint

mkdir -p logs results

source ~/.bashrc
conda activate your_env_name   # <-- EDIT: replace with your actual env name

cd "$SLURM_SUBMIT_DIR"          # project ROOT, not src/ -- see note below
export HF_HUB_OFFLINE=1         # avoid retrying the known-broken HF network path

CHECKPOINT=models/distilbert_imdb_baseline.pt

echo "=== $(date) : job started on $(hostname) ==="
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT."
    echo "Transfer it from the local Windows machine first (scp), then resubmit."
    exit 1
fi

echo "=== Step 1: harness validation (FP32-CPU, small, fast) ==="
python -m src.benchmark.benchmark \
    --device cpu \
    --batch-sizes 1 2 \
    --num-warmup 3 \
    --num-iters 5 \
    --checkpoint "$CHECKPOINT" \
    --tag fp32_cpu_validation \
    --output results/benchmark_fp32_cpu_validation.csv

echo "=== Step 2: full FP32-CPU sweep ==="
python -m src.benchmark.benchmark \
    --device cpu \
    --batch-sizes 1 2 4 8 16 32 64 \
    --num-warmup 10 \
    --num-iters 50 \
    --checkpoint "$CHECKPOINT" \
    --tag fp32_cpu \
    --output results/benchmark_fp32_cpu.csv

echo "=== Step 3: full INT8-CPU sweep (the controlled comparison) ==="
python -m src.benchmark.benchmark \
    --quantized \
    --batch-sizes 1 2 4 8 16 32 64 \
    --num-warmup 10 \
    --num-iters 50 \
    --checkpoint "$CHECKPOINT" \
    --tag int8_cpu \
    --output results/benchmark_int8_cpu.csv

echo "=== $(date) : job finished ==="
echo "Compare results/benchmark_fp32_cpu.csv vs results/benchmark_int8_cpu.csv"
echo "-- same device, same batch sizes, precision is the only variable."
