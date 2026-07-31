# DistilBERT IMDb Serving-Optimization Study

**Course:** DSAA 4012 — Machine Learning Systems, HKUST(GZ)
**Archetype:** 2.2 — Serving-optimization study
**Team:** Bakhom Ramzy, Fatma Haddad

## 1. Systems question

How do **batch size** and **INT8 dynamic quantization** move the
latency–throughput frontier for a fine-tuned DistilBERT sentiment
classifier, and where is the **batching knee** — the point past which
a larger batch stops raising throughput but keeps raising tail
latency?

**Headline result:** on the HPC CPU node, INT8 dynamic quantization
lifts peak throughput from **63.32 samples/s (FP32, batch 64)** to
**100.58 samples/s (INT8, batch 16)** — a **1.59×** peak-throughput
improvement — while test accuracy moves from **87.02% → 87.16%**
(+0.14 pp, i.e. no measurable accuracy cost). See §6 for the full
frontier and the batching-knee analysis.

## 2. Hardware & environment

This project targets a single-GPU / single-node budget as required by
the course (§1 of the handout), split across two machines:

| Role | Resource |
|---|---|
| GPU training + FP32 GPU inference benchmarking | **NVIDIA GTX 1650** (4 GB VRAM), local workstation |
| HPC CPU node — FP32 and INT8 CPU inference benchmarking | **Intel Xeon Gold 6348H @ 2.30 GHz**, 96 CPUs, 3 TB RAM |

The CPU sweeps were submitted as a batch job on the HPC cluster
(`scripts/benchmark_cpu.slurm`) rather than run interactively, so they
survive SSH disconnects and don't contend with an interactive session.
GPU training, GPU FP32 benchmarking, and both accuracy-evaluation runs
were run locally on the GTX 1650 workstation.

> **Note on `scripts/run_train.slurm`, `run_benchmark.slurm`,
> `run_full_pipeline.slurm`:** these three job scripts request an
> HPC A40 GPU partition (`--gres=gpu:a40:1`). They were early
> scaffolding for a plan to run GPU work on the HPC cluster and were
> **not used to produce any of the numbers in this report** — the
> GPU work reported here ran locally on the GTX 1650 instead. They are
> kept in the repo for provenance/history, not as part of the
> reproduction path below. The only SLURM script actually used for
> the submitted results is `scripts/benchmark_cpu.slurm`.

Reliability practices followed throughout (per the course's measuring
guidance): warm-up iterations excluded from timing, median (and p95
for latency) reported instead of a single reading, `torch.cuda.Event`
+ `torch.cuda.synchronize()` used for GPU timing (not wall-clock),
peak-memory counters reset per batch size, and FP32-vs-INT8 comparisons
run on the same machine with precision as the only varying factor.

## 3. Model & dataset

| | |
|---|---|
| Model | DistilBERT (`distilbert-base-uncased`), ~66M parameters |
| Task | Binary sentiment classification, fine-tuned 1 epoch |
| Dataset | IMDb Movie Reviews (`stanfordnlp/imdb`) — 25,000 train / 25,000 test |

## 4. Software / frameworks

- Python, PyTorch 2.0.1
- Hugging Face Transformers, Hugging Face Datasets
- PyTorch dynamic INT8 quantization (`torch.quantization.quantize_dynamic`, `nn.Linear` only — embeddings/LayerNorm stay FP32, standard practice)
- CUDA 11.8 (GPU environment)
- matplotlib, pandas (analysis/plotting)
- ONNX Runtime is listed in `requirements.txt` for a possible runtime-choice extension but is **not used** in the submitted results — see §7.

## 5. Repository structure

```
src/
  data/           dataset loading, tokenization (max_length=128), dataloaders
  model/          DistilBERT classifier construction
  train.py        1-epoch fine-tuning loop, saves models/distilbert_imdb_baseline.pt
  evaluate.py              FP32 accuracy on the IMDb test set (real data)
  evaluate_quantized.py    INT8 accuracy on the IMDb test set (real data, identical eval loop)
  quantize_utils.py        builds the dynamically-quantized INT8 model (CPU-only)
  benchmark/benchmark.py   latency/throughput/memory sweep over batch sizes
                           (synthetic fixed-shape inputs — see §7 for why)
  analysis/       roofline_analysis.py, compare_benchmarks.py,
                  generate_report.py, final_summary.py, plot_results.py
  api/            placeholder — no serving endpoint implemented (not required
                  by the Archetype 2.2 evaluated deliverable; see §7)
scripts/          SLURM job scripts (see the note in §2 on which ones were
                  actually used)
results/          all CSV/JSON results and PNG plots referenced in this README
```

## 6. Setup & reproduction

### 6.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 6.2 Get the dataset (once)

`src/data/preprocess.py` loads from a local `imdb_dataset/` directory
(`datasets.load_from_disk`) rather than hitting the network on every
run — useful on an HPC node where outbound access to huggingface.co
is often unreliable. Fetch and cache it once:

```bash
python -c "from datasets import load_dataset; ds = load_dataset('stanfordnlp/imdb'); ds.save_to_disk('imdb_dataset')"
"
```

Do **not** commit `imdb_dataset/` to the repository — it's a public
dataset, so per the course's artifact rule it is referenced (this
script), not uploaded.

### 6.3 Train the baseline (GTX 1650)

```bash
python -m src.train
```
Saves a checkpoint to `models/distilbert_imdb_baseline.pt` (created
automatically). Fine-tunes 1 epoch over the 25,000-example IMDb
training split.

### 6.4 Accuracy evaluation (real IMDb test set, not synthetic)

```bash
python -m src.evaluate              # FP32 -> results/baseline_accuracy.json
python -m src.evaluate_quantized    # INT8 -> results/int8_accuracy.json
```

### 6.5 Latency/throughput/memory benchmarks

The benchmark harness (`src/benchmark/benchmark.py`) uses **synthetic,
fixed-shape token batches** (see §7 for why) rather than real IMDb
text, so batch size — or precision — is the only variable that
changes between runs.

GPU FP32 sweep (run on the GTX 1650):
```bash
python -m src.benchmark.benchmark --batch-sizes 1 2 4 8 16 32 64 \
    --checkpoint models/distilbert_imdb_baseline.pt \
    --tag fp32 --output results/benchmark_fp32.csv
```

CPU FP32 and CPU INT8 sweeps (run on the HPC node, as a batch job):
```bash
sbatch scripts/benchmark_cpu.slurm
```
which internally runs:
```bash
python -m src.benchmark.benchmark --device cpu --batch-sizes 1 2 4 8 16 32 64 \
    --checkpoint models/distilbert_imdb_baseline.pt \
    --tag fp32_cpu --output results/benchmark_fp32_cpu.csv

python -m src.benchmark.benchmark --quantized --batch-sizes 1 2 4 8 16 32 64 \
    --checkpoint models/distilbert_imdb_baseline.pt \
    --tag int8_cpu --output results/benchmark_int8_cpu.csv
```
(the `results/*_final.csv` files in this repo are the confirmed runs
used for the report and plots below.)

### 6.6 Analysis & plots

```bash
python -m src.analysis.compare_benchmarks   # -> results/int8_speedup_summary.csv
python -m src.analysis.generate_report      # -> results/final_fp32_vs_int8_summary.csv
python -m src.analysis.roofline_analysis    # -> results/roofline_summary.json
python -m src.analysis.final_summary        # -> results/final_summary.json
python -m src.analysis.plot_results         # -> results/*.png
```

## 7. What's real vs. stubbed (honesty per course requirement)

**Real, run end-to-end, and reproducible from this repo:**
- Data loading/tokenization, DistilBERT fine-tuning (1 epoch on the GTX 1650), and both accuracy evaluations — all on the real IMDb train/test split.
- INT8 dynamic quantization of the fine-tuned checkpoint.
- The full latency/throughput/peak-memory batch-size sweep, in three controlled configurations: GPU FP32, HPC-CPU FP32, HPC-CPU INT8.
- All comparison tables and the roofline-style analysis (`results/roofline_summary.json`, `results/final_summary.json`) are computed directly from the CSVs above, not hand-entered.

**Deliberately simplified:**
- The benchmark harness measures inference on **synthetic, fixed-shape token batches**, not real IMDb text — deliberately, so that batch size (or precision) is the only variable that changes between sweeps, and so the harness needs no dataset access on a node where HF Hub access is unreliable. Accuracy is measured separately on the real test set (§6.4); the two are never mixed in one run.
- CPU peak-memory is not reported (`peak_memory_mb` is empty for CPU rows) — `torch.cuda.max_memory_allocated` has no meaningfully-comparable CPU equivalent across processes/OS, so we didn't fabricate a number for it. GPU peak memory (526–785 MB across batch sizes) is real, from `torch.cuda.max_memory_allocated`.

**Not built (out of scope for this archetype's evaluated deliverable):**
- `src/api/` — a serving endpoint. The Archetype 2.2 evaluated deliverable is the latency–throughput–accuracy frontier and roofline analysis, not a deployed service, so this was left as a placeholder rather than padded out.
- ONNX Runtime export/runtime-choice comparison — listed as a possible third knob in the handout and in `requirements.txt`, but only two knobs (batch size × precision) were carried through to a full measured comparison, per the course's "depth beats breadth" guidance.
- The three legacy A40-targeting SLURM scripts described in §2 — scaffolding from an earlier plan, not used for the submitted results.

## 8. Results

### 8.1 Accuracy (real IMDb test set, 25,000 examples)

| | FP32 | INT8 (dynamic) | Δ |
|---|---|---|---|
| Test accuracy | 87.02% | 87.16% | **+0.14 pp** |
| Test loss | 0.2993 | 0.2993 | ≈0 |

### 8.2 GPU FP32 (GTX 1650) — latency/throughput/peak memory

| Batch | Median latency (ms) | p95 (ms) | Throughput (samples/s) | Peak mem (MB) |
|---|---|---|---|---|
| 1  | 8.96   | 9.78   | 111.55 | 526.24 |
| 2  | 15.88  | 16.19  | 125.93 | 529.87 |
| 4  | 26.46  | 26.83  | **151.20 (peak)** | 539.62 |
| 8  | 56.08  | 56.20  | 142.67 | 554.63 |
| 16 | 136.76 | 138.75 | 116.99 | 587.65 |
| 32 | 311.06 | 314.64 | 102.87 | 653.18 |
| 64 | 654.91 | 660.24 | 97.72  | 785.24 |

**GPU batching knee: batch 4.** Throughput peaks at batch 4 and then
*declines* as batch size keeps growing — on this 4 GB card, latency
grows faster than batch size beyond the ridge point, so bigger
batches buy nothing and cost tail latency. See `results/gpu_fp32_throughput.png` and `results/gpu_fp32_latency.png`.

### 8.3 HPC-CPU FP32 vs. INT8 — latency/throughput

| Batch | FP32 throughput | INT8 throughput | Speedup | FP32 p95 (ms) | INT8 p95 (ms) |
|---|---|---|---|---|---|
| 1  | 3.43  | 17.29  | **5.04×** | 765.46 | 139.01 |
| 2  | 15.70 | 25.18  | 1.60× | 312.64 | 188.64 |
| 4  | 40.49 | 51.76  | 1.28× | 203.55 | 148.10 |
| 8  | 61.45 | 74.32  | 1.21× | 232.13 | 179.50 |
| 16 | 60.11 | **100.58 (INT8 peak)** | 1.67× | 403.67 | 242.18 |
| 32 | 58.80 | 83.53  | 1.42× | 688.52 | 533.99 |
| 64 | **63.32 (FP32 peak)** | 54.31  | 0.86× | 1279.28 | 1460.75 |

**CPU batching knee — FP32:** throughput is essentially flat from
batch 8 to batch 64 (61.45 → 63.32 samples/s) while p95 latency grows
6×, so batches above ~8 buy almost no additional throughput on this
CPU for almost 6× the tail latency. **INT8 knee: batch 16** —
throughput peaks there and then *drops* at 32 and 64, while latency
keeps climbing; INT8 also **stops helping past batch 32** (0.86×,
i.e. INT8 is slower than FP32 at batch 64), the one place in the
sweep where quantization does not pay off. See
`results/throughput_fp32_vs_int8.png`,
`results/latency_fp32_vs_int8.png`,
`results/p95_latency_fp32_vs_int8.png`.

### 8.4 Compute- vs. memory-bound reading (Roofline vocabulary)

- **GPU:** a 66M-parameter model is tiny relative to even a 4 GB
  card's compute budget, so at batch 1–4 the GPU is likely
  latency/launch-overhead-bound (per-call fixed costs dominate) —
  arithmetic intensity per call is too low to hide those costs. Batch
  4 is close to the ridge point; past it, added compute work per call
  grows faster than the throughput gain, consistent with the model
  becoming compute-bound on a small GPU sooner than a larger card
  would.
- **CPU:** INT8 dynamic quantization mainly cuts the size and
  bandwidth cost of the linear-layer weights, which is exactly the
  lever that should matter if CPU inference is memory-bandwidth-bound
  rather than compute-bound — consistent with the large low-batch
  speedup (5.04× at batch 1, where weight-loading cost dominates
  per-sample cost) shrinking as batch size grows and compute cost
  starts to dominate. The INT8 regression at batch 64 suggests the
  quantized kernel's per-call overhead or reduced parallelism
  eventually outweighs its bandwidth savings once the batch is large
  enough to be compute-bound instead.

These are our interpretations from the measured curves, not separately
verified with a hardware profiler — a natural next step (see §9).

## 9. Limitations & possible next steps

- Only two knobs were swept (batch size, precision); a runtime-choice
  knob (ONNX Runtime) was scoped out per §7.
- The compute- vs. memory-bound explanation in §8.4 is inferred from
  the shape of the latency/throughput curves, not confirmed with a
  hardware profiler (e.g., `nsight-compute` on the GPU or `perf` on
  the CPU) — a natural extension.
- Single-run medians per batch size (no repeated multi-run
  confidence interval across separate process launches) — within-run
  we do take the median of many iterations (10 GPU repeats × 50
  timed calls; CPU 20 warm-up + 100 timed calls per batch size), but
  we did not repeat the entire sweep across multiple process launches.
- CPU peak memory is not reported (§7); a `/proc`-based RSS
  measurement would let us report a CPU-side memory table too.

## 10. Individual-contribution statement

Both members contributed to the design, implementation, evaluation,
and write-up. Contribution was not perfectly even; by mutual
agreement, **Bakhom Ramzy** carried a larger share of the work overall.

| Area | Bakhom Ramzy | Fatma Haddad |
|---|---|---|
| System design & systems question | Lead | Contributed |
| Implementation (data/model/train/benchmark/quantization/analysis code) | Lead | Contributed |
| Evaluation & interpretation (roofline reasoning, plots) | Lead | Contributed |
| Report writing | Contributed | Contributed |
| Presentation | Contributed | Contributed |

*Agreed and signed by both members: Bakhom Ramzy, Fatma Haddad.*

## 11. AI-use acknowledgment

Generative AI (Claude) was used during this project, consistent with
the course AI policy: for **debugging specific, localized errors**
(e.g., stack traces from tokenization/dataloader/quantization calls)
and for **writing repetitive, boilerplate code** (e.g., CSV/JSON
serialization in the benchmark harness, matplotlib plotting
boilerplate across the three comparison plots, argparse scaffolding).
It was not used to generate the system design, the choice of
archetype, the systems question, the evaluation methodology, or the
analysis/conclusions in §8–9, which are the team's own. Both members
can explain every line of the submitted code.

## 12. References

- Wolf et al., *Transformers: State-of-the-Art Natural Language
  Processing*, Hugging Face.
- Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL
  2011 (IMDb dataset).
- Sanh et al., *DistilBERT, a distilled version of BERT*, arXiv:1910.01108.
- PyTorch dynamic quantization documentation, https://pytorch.org
