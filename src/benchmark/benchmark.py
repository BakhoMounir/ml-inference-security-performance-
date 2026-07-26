"""
Serving benchmark for DistilBERT IMDb classifier.

Measures, for each batch size in a sweep:
  - median latency (ms)
  - p95 latency (ms)
  - throughput (samples/sec)
  - peak memory (MB) -- GPU peak via torch.cuda, or None on CPU (not
    meaningfully comparable across processes/OS the way GPU peak alloc is)

Methodology (per course rubric §1 / Phase 4 requirements):
  - model.eval() + torch.inference_mode()
  - warm-up iterations excluded from timing
  - torch.cuda.Event for GPU timing, with torch.cuda.synchronize()
    around each measured region (NOT naive time.time() wall-clock,
    which does not wait for the async CUDA queue to drain)
  - same fixed input shape (seq_len=128) across all batch sizes so
    batch size is the only varying factor
  - peak memory reset per batch size via
    torch.cuda.reset_peak_memory_stats()

NEW: --device and --quantized let you run the SAME script to produce the
controlled FP32-CPU vs INT8-CPU comparison (precision is the only variable
that changes between the two runs, same device, same machine):

    # FP32 on CPU (forces CPU even if a GPU is present)
    python -m src.benchmark.benchmark --device cpu \
        --checkpoint models/distilbert_imdb_baseline.pt \
        --tag fp32_cpu --output results/benchmark_fp32_cpu.csv

    # INT8 (dynamic quantization is CPU-only regardless of --device)
    python -m src.benchmark.benchmark --quantized \
        --checkpoint models/distilbert_imdb_baseline.pt \
        --tag int8_cpu --output results/benchmark_int8_cpu.csv

Usage (original FP32-GPU sweep, unchanged):
    python -m src.benchmark.benchmark --batch-sizes 1 2 4 8 16 32 64 \
        --checkpoint models/distilbert_imdb_baseline.pt \
        --output results/benchmark_fp32.csv

If --checkpoint is omitted, the script still runs (using the
pretrained-but-not-fine-tuned head) so you can validate the harness
itself before you have a trained checkpoint. It will print a warning
that latency/throughput numbers are still valid but accuracy is not.
"""
import argparse
import csv
import json
import os
import statistics
import time

import torch

from src.model.model import build_model
from src.quantize_utils import build_quantized_model
from src.utils.reproducibility import set_seed

SEQ_LEN = 128


def build_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device):
    """Fixed-shape synthetic input. Using synthetic token ids (rather than
    real IMDb text) keeps the input distribution IDENTICAL across every
    batch size and every precision/quantization variant, so batch size
    (or precision) is the only thing that changes between runs -- exactly
    what Phase 5/6 require ('do not change multiple variables at once').
    This also means no HF dataset download is needed to run this script,
    which matters specifically on HPC where the IMDb dataset is not
    reliably reachable."""
    input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len), device=device
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    return input_ids, attention_mask


@torch.inference_mode()
def time_batch_size(
    model,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int,
    num_iters: int,
):
    input_ids, attention_mask = build_synthetic_batch(
        batch_size, seq_len, vocab_size, device
    )

    # --- warm-up (not timed) ---
    for _ in range(num_warmup):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # --- reset peak memory counter AFTER warmup, BEFORE measured loop ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []

    if device.type == "cuda":
        for _ in range(num_iters):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            end_evt.record()

            torch.cuda.synchronize()  # required: events are async until this
            latencies_ms.append(start_evt.elapsed_time(end_evt))  # already ms

        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        # CPU fallback: time.perf_counter is synchronous by nature on CPU,
        # so no cuda.Event equivalent is needed here.
        for _ in range(num_iters):
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
        peak_mem_mb = float("nan")  # not meaningfully comparable on CPU

    latencies_ms.sort()
    median_ms = statistics.median(latencies_ms)
    p95_ms = latencies_ms[int(0.95 * len(latencies_ms)) - 1]
    throughput = batch_size / (median_ms / 1000.0)  # samples/sec

    return {
        "batch_size": batch_size,
        "median_latency_ms": round(median_ms, 4),
        "p95_latency_ms": round(p95_ms, 4),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_memory_mb": round(peak_mem_mb, 2) if peak_mem_mb == peak_mem_mb else None,
        "num_warmup": num_warmup,
        "num_iters": num_iters,
    }


def main():
    parser = argparse.ArgumentParser(description="Serving latency/throughput benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a fine-tuned state_dict (.pt). If omitted, "
                              "runs with the pretrained-but-not-fine-tuned head "
                              "(harness validation only).")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/benchmark_fp32.csv")
    parser.add_argument("--tag", type=str, default="fp32",
                         help="Label for this run, e.g. fp32 / fp32_cpu / int8_cpu. "
                              "Stored in the output so runs can be concatenated later.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                         help="Force a specific device. If omitted, auto-selects "
                              "cuda if available, else cpu. Use --device cpu to "
                              "force a CPU run on a machine that also has a GPU, "
                              "so you get a controlled FP32-CPU baseline to compare "
                              "against --quantized (which is CPU-only anyway).")
    parser.add_argument("--quantized", action="store_true",
                         help="Build a dynamically INT8-quantized model instead of "
                              "the plain FP32 model. Requires --checkpoint (quantization "
                              "is applied to a fine-tuned checkpoint, not a random head). "
                              "Always runs on CPU regardless of --device, since PyTorch "
                              "dynamic quantization has no CUDA kernels.")
    args = parser.parse_args()

    set_seed(args.seed)

    # --- resolve device ---
    if args.quantized:
        if args.device == "cuda":
            print("WARNING: --quantized forces CPU (dynamic quantization has no "
                  "CUDA kernels). Ignoring --device cuda.")
        device = torch.device("cpu")
    elif args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available here.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}" + (" (quantized: CPU forced)" if args.quantized else ""))
    if device.type != "cuda" and not args.quantized:
        print("WARNING: running on CPU. Latency/throughput/memory numbers from a "
              "CPU run are NOT comparable to GPU numbers -- keep them in a "
              "separate results file and label it clearly (this is exactly what "
              "--tag is for).")

    # --- build model ---
    if args.quantized:
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                "--quantized requires a valid --checkpoint (quantizing an "
                "untrained head is not a meaningful measurement)."
            )
        print(f"Building dynamically-quantized INT8 model from: {args.checkpoint}")
        model = build_quantized_model(args.checkpoint)
        # build_quantized_model already puts the model on CPU and calls .eval()
    else:
        model = build_model()
        if args.checkpoint is not None:
            if not os.path.exists(args.checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint not found at {args.checkpoint}. "
                    f"Train first (python -m src.train) or pass --checkpoint correctly."
                )
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded fine-tuned checkpoint: {args.checkpoint}")
        else:
            print("WARNING: no --checkpoint given. Running with an UNTRAINED "
                  "classification head. Latency/throughput/memory numbers are "
                  "still valid (they don't depend on trained weights), but do "
                  "NOT report accuracy from this run.")
        model.to(device)
        model.eval()  # required before inference_mode timing

    vocab_size = model.config.vocab_size

    results = []
    for bs in args.batch_sizes:
        print(f"Benchmarking batch_size={bs} ...")
        try:
            result = time_batch_size(
                model, bs, args.seq_len, vocab_size, device,
                args.num_warmup, args.num_iters,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs} -- stopping sweep here. "
                  f"This IS a valid result: it's your memory ceiling.")
            break
        result["tag"] = args.tag
        results.append(result)
        print(f"  median={result['median_latency_ms']} ms | "
              f"p95={result['p95_latency_ms']} ms | "
              f"throughput={result['throughput_samples_per_sec']} samples/sec | "
              f"peak_mem={result['peak_memory_mb']} MB")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = ["tag", "batch_size", "median_latency_ms", "p95_latency_ms",
                  "throughput_samples_per_sec", "peak_memory_mb",
                  "num_warmup", "num_iters"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved {len(results)} rows to {args.output}")

    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Also saved {json_path}")


if __name__ == "__main__":
    main()
