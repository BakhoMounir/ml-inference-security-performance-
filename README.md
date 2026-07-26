# DistilBERT IMDb Serving-Optimization Study

Archetype 2.2 — how batch size and INT8 quantization move the
latency/throughput/accuracy trade-off for DistilBERT sentiment
classification.

## Status

- Data pipeline, model, training loop: implemented and previously run
  on an A40 GPU (see historical results below).
- Benchmark harness (`src/benchmark/benchmark.py`): implemented,
  **not yet run end-to-end** in this environment (no GPU/internet
  access to huggingface.co here) — run Step 1 below yourself and
  confirm the output before trusting Step 2's numbers.
- Quantization (INT8) and ONNX export: not yet implemented.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Train the baseline

```bash
cd src
python -m train
```
Saves a checkpoint to `models/distilbert_imdb_baseline.pt` (directory
created automatically). Takes ~10 minutes on an A40-class GPU for
1 epoch over 25,000 IMDb training examples.

## Validate the benchmark harness (run this first, ~10 seconds)

```bash
cd src
python -m benchmark.benchmark --batch-sizes 1 2 --num-warmup 3 --num-iters 5 \
    --tag validation --output ../results/validation.csv
```

## Run the full batch-size sweep (requires a trained checkpoint)

```bash
cd src
python -m benchmark.benchmark --batch-sizes 1 2 4 8 16 32 64 \
    --checkpoint ../models/distilbert_imdb_baseline.pt \
    --tag fp32 --output ../results/benchmark_fp32.csv
```

On the HKUST(GZ) HPC cluster, submit as a batch job instead of running
interactively (avoids SSH-disconnect/queue issues):
```bash
sbatch scripts/run_benchmark.slurm
```

## Historical baseline (1 epoch, A40 GPU) — not yet re-verified with a seed set

- Train loss: 0.3576, Train accuracy: 0.8436
- Test loss: 0.3066, Test accuracy: 0.8674
- Epoch time: ~637s

## What's real vs. stubbed (per course honesty requirement)

- **Real:** data pipeline, model, training, benchmark harness (timing
  methodology validated logically; needs one live run to confirm).
- **Stubbed / not yet built:** `src/api/` (serving endpoint — not
  required by the Evaluated deliverable for this archetype, kept as a
  placeholder), quantization variant, ONNX export, plotting scripts,
  systems analysis write-up.
