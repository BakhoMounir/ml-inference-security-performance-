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
raises the observed peak throughput from **63.32 samples/s (FP32,
batch 64)** to **100.58 samples/s (INT8, batch 16)** — a **1.59×**
improvement in peak throughput — while test accuracy moves from
**87.02% → 87.16%** (**+0.14 percentage points**). See §6 for the
full frontier and the batching-knee analysis.

The peak-throughput comparison is intentionally made between the
best-performing batch size for each precision: batch 64 for FP32 and
batch 16 for INT8.

## 2. Hardware & environment

This project targets a single-GPU / single-node budget as required by
the course (§1 of the handout), split across two machines:

| Role                                                    | Resource                                                |
| ------------------------------------------------------- | ------------------------------------------------------- |
| GPU training + FP32 GPU inference benchmarking          | **NVIDIA GTX 1650** (4 GB VRAM), local workstation      |
| HPC CPU node — FP32 and INT8 CPU inference benchmarking | **Intel Xeon Gold 6348H @ 2.30 GHz**, 96 CPUs, 3 TB RAM |

The CPU sweeps were submitted as a batch job on the HPC cluster
(`scripts/benchmark_cpu.slurm`) rather than run interactively, so they
survive SSH disconnects and don't contend with an interactive session.
GPU training, GPU FP32 benchmarking, and both accuracy-evaluation runs
were run locally on the GTX 1650 workstation.

> **Note on `scripts/run_train.slurm`, `run_benchmark.slurm`, and
> `run_full_pipeline.slurm`:** these three job scripts request an HPC
> A40 GPU partition (`--gres=gpu:a40:1`). They were early scaffolding
> for a plan to run GPU work on the HPC cluster and were **not used to
> produce any of the numbers in this report** — the GPU work reported
> here ran locally on the GTX 1650 instead. They are kept in the repo
> for provenance/history, not as part of the reproduction path below.
> The only SLURM script actually used for the submitted CPU results is
> `scripts/benchmark_cpu.slurm`.

Reliability practices followed throughout (per the course's measuring
guidance) include excluding warm-up iterations from timing, reporting
median and p95 latency rather than relying on a single reading,
using `torch.cuda.Event` and `torch.cuda.synchronize()` for GPU timing
rather than naive wall-clock timing, resetting GPU peak-memory
counters per batch size, and running FP32-vs-INT8 comparisons on the
same CPU hardware with precision as the principal experimental factor.

## 3. Model & dataset

|             |                                                                      |
| ----------- | -------------------------------------------------------------------- |
| **Model**   | DistilBERT (`distilbert-base-uncased`), ~66M parameters              |
| **Task**    | Binary sentiment classification, fine-tuned for 1 epoch              |
| **Dataset** | IMDb Movie Reviews (`stanfordnlp/imdb`) — 25,000 train / 25,000 test |

## 4. Software / frameworks

* Python, PyTorch 2.0.1
* Hugging Face Transformers, Hugging Face Datasets
* PyTorch dynamic INT8 quantization
  (`torch.quantization.quantize_dynamic`, `nn.Linear` only —
  embeddings and LayerNorm remain FP32)
* CUDA 11.8 (GPU environment)
* matplotlib, pandas (analysis/plotting)
* ONNX Runtime is listed in `requirements.txt` for a possible
  runtime-choice extension but is **not used in the submitted results**
  — see §7.

## 5. Repository structure

```text
src/
  data/           dataset loading, tokenization (max_length=128), dataloaders
  model/          DistilBERT classifier construction
  train.py        1-epoch fine-tuning loop, saves models/distilbert_imdb_baseline.pt
  evaluate.py     FP32 accuracy on the IMDb test set (real data)
  evaluate_quantized.py
                  INT8 accuracy on the IMDb test set (real data, identical eval loop)
  quantize_utils.py
                  builds the dynamically-quantized INT8 model (CPU-only)
  benchmark/benchmark.py
                  latency/throughput/memory sweep over batch sizes
                  (synthetic fixed-shape inputs — see §7 for why)
  analysis/       roofline_analysis.py, compare_benchmarks.py,
                  generate_report.py, final_summary.py, plot_results.py
  api/            placeholder — no serving endpoint implemented
                  (not required by the Archetype 2.2 evaluated deliverable)
scripts/          SLURM job scripts (see §2 on which ones were actually used)
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
may be unreliable. Fetch and cache it once:

```bash
python -c "from datasets import load_dataset; ds = load_dataset('stanfordnlp/imdb'); ds.save_to_disk('imdb_dataset')"
```

Do **not** commit `imdb_dataset/` to the repository — it is a public
dataset, so per the course's artifact rule it is referenced by this
script rather than uploaded.

### 6.3 Train the baseline (GTX 1650)

```bash
python -m src.train
```

Saves a checkpoint to `models/distilbert_imdb_baseline.pt` (created
automatically). The model is fine-tuned for 1 epoch over the
25,000-example IMDb training split.

### 6.4 Accuracy evaluation (real IMDb test set, not synthetic)

```bash
python -m src.evaluate              # FP32 -> results/baseline_accuracy.json
python -m src.evaluate_quantized    # INT8 -> results/int8_accuracy.json
```

### 6.5 Latency/throughput/memory benchmarks

The benchmark harness (`src/benchmark/benchmark.py`) uses
**synthetic, fixed-shape token batches** (see §7 for why) rather than
real IMDb text, so batch size — or precision — is the controlled
experimental variable between runs.

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

The `results/*_final.csv` files in this repository are the confirmed
runs used for the report and plots below.

### 6.6 Analysis & plots

```bash
python -m src.analysis.compare_benchmarks   # -> results/int8_speedup_summary.csv
python -m src.analysis.generate_report      # -> results/final_fp32_vs_int8_summary.csv
python -m src.analysis.roofline_analysis    # -> results/roofline_summary.json
python -m src.analysis.final_summary        # -> results/final_summary.json
python -m src.analysis.plot_results         # -> results/*.png
```

## 7. What's real vs. stubbed (honesty per course requirement)

### Real, run end-to-end, and reproducible from this repo

* Data loading/tokenization, DistilBERT fine-tuning (1 epoch on the
  GTX 1650), and both accuracy evaluations — all on the real IMDb
  train/test split.
* INT8 dynamic quantization of the fine-tuned checkpoint.
* The full latency/throughput/peak-memory batch-size sweep in three
  controlled configurations: GPU FP32, HPC-CPU FP32, and HPC-CPU INT8.
* All comparison tables and the roofline-style analysis
  (`results/roofline_summary.json`, `results/final_summary.json`) are
  computed directly from the CSVs above rather than being
  hand-entered.

### Deliberately simplified

* The benchmark harness measures inference on **synthetic, fixed-shape
  token batches**, not real IMDb text. This was deliberate so that
  batch size (or precision) is the controlled experimental variable
  and so that the benchmark harness does not require dataset access
  on a node where Hugging Face Hub access may be unreliable.
  Accuracy is measured separately on the real test set (§6.4); the
  two evaluations are never mixed in one run.
* CPU peak memory is not reported (`peak_memory_mb` is empty for CPU
  rows). `torch.cuda.max_memory_allocated` has no directly comparable
  CPU equivalent in this benchmark, so no CPU memory value was
  fabricated.
* GPU peak memory (526–785 MB across the tested batch sizes) is real
  and comes from `torch.cuda.max_memory_allocated`.

### Not built (out of scope for this archetype's evaluated deliverable)

* `src/api/` — a serving endpoint. The Archetype 2.2 evaluated
  deliverable is the latency–throughput–accuracy frontier and
  roofline-style analysis, not a deployed service, so this was left
  as a placeholder rather than padded out.
* ONNX Runtime export/runtime-choice comparison — listed as a possible
  third knob in the handout and in `requirements.txt`, but only two
  knobs (batch size × precision) were carried through to a full
  measured comparison, following the course's "depth beats breadth"
  guidance.
* The three legacy A40-targeting SLURM scripts described in §2 —
  scaffolding from an earlier plan, not used for the submitted
  results.

## 8. Results

### 8.1 Accuracy (real IMDb test set, 25,000 examples)

|               |   FP32 | INT8 (dynamic) |            Δ |
| ------------- | -----: | -------------: | -----------: |
| Test accuracy | 87.02% |         87.16% | **+0.14 pp** |
| Test loss     | 0.2993 |         0.2993 |           ≈0 |

The observed difference is small and should be interpreted as **no
observed accuracy degradation in this evaluation**, rather than as
proof that quantization can never affect accuracy.

### 8.2 GPU FP32 (GTX 1650) — latency/throughput/peak memory

| Batch | Median latency (ms) | p95 (ms) | Throughput (samples/s) | Peak mem (MB) |
| ----: | ------------------: | -------: | ---------------------: | ------------: |
|     1 |                8.96 |     9.78 |                 111.55 |        526.24 |
|     2 |               15.88 |    16.19 |                 125.93 |        529.87 |
|     4 |               26.46 |    26.83 |      **151.20 (peak)** |        539.62 |
|     8 |               56.08 |    56.20 |                 142.67 |        554.63 |
|    16 |              136.76 |   138.75 |                 116.99 |        587.65 |
|    32 |              311.06 |   314.64 |                 102.87 |        653.18 |
|    64 |              654.91 |   660.24 |                  97.72 |        785.24 |

**GPU batching knee: batch 4.** Throughput peaks at batch 4 and then
declines as batch size continues to grow, while p95 latency increases
substantially. Thus, for this GTX 1650 configuration, larger batches
beyond the knee do not improve throughput and instead impose a
significant tail-latency cost.

See `results/gpu_fp32_throughput.png` and
`results/gpu_fp32_latency.png`.

### 8.3 HPC-CPU FP32 vs. INT8 — latency/throughput

| Batch |       FP32 throughput |        INT8 throughput |   Speedup | FP32 p95 (ms) | INT8 p95 (ms) |
| ----: | --------------------: | ---------------------: | --------: | ------------: | ------------: |
|     1 |                  3.43 |                  17.29 | **5.04×** |        765.46 |        139.01 |
|     2 |                 15.70 |                  25.18 |     1.60× |        312.64 |        188.64 |
|     4 |                 40.49 |                  51.76 |     1.28× |        203.55 |        148.10 |
|     8 |                 61.45 |                  74.32 |     1.21× |        232.13 |        179.50 |
|    16 |                 60.11 | **100.58 (INT8 peak)** |     1.67× |        403.67 |        242.18 |
|    32 |                 58.80 |                  83.53 |     1.42× |        688.52 |        533.99 |
|    64 | **63.32 (FP32 peak)** |                  54.31 |     0.86× |       1279.28 |       1460.75 |

**CPU batching knee — FP32:** throughput is approximately flat from
batch 8 through batch 64 (61.45 → 63.32 samples/s), while p95 latency
increases from 232.13 ms to 1279.28 ms. Thus, batches above roughly 8
provide little additional throughput for a large tail-latency cost.

**INT8 knee: batch 16.** Throughput peaks at 100.58 samples/s at batch
16 and then falls at batches 32 and 64 while latency continues to
increase. INT8 provides its largest relative speedup at batch 1
(5.04×) and eventually becomes slower than FP32 at batch 64
(0.86×).

See `results/throughput_fp32_vs_int8.png`,
`results/latency_fp32_vs_int8.png`, and
`results/p95_latency_fp32_vs_int8.png`.

### 8.4 Compute- vs. memory-bound reading (Roofline vocabulary)

The following interpretation is based on the observed
latency–throughput curves. It is **not independently verified by a
hardware performance profiler**.

* **GPU:** At small batch sizes, the measured increase in throughput
  from batch 1 to batch 4 is consistent with fixed per-inference
  overheads being amortized across more samples. Batch 4 corresponds
  to the observed throughput maximum. Beyond this point, increasing
  batch size substantially increases latency while reducing
  throughput, suggesting that additional computation and resource
  pressure are no longer being compensated by improved utilization.
  A hardware profiler would be required to determine the exact
  compute/memory bottleneck.

* **CPU:** Dynamic INT8 quantization reduces the representation size
  and computational cost of the quantized `nn.Linear` layers. The
  strong low-batch improvement — 5.04× throughput at batch 1 —
  is consistent with quantization reducing the cost of CPU inference.
  The decreasing speedup with increasing batch size, followed by the
  INT8 regression at batch 64, indicates that the benefit is
  workload-dependent rather than universal. The available benchmark
  data alone cannot establish whether memory bandwidth, arithmetic
  throughput, kernel overhead, thread scheduling, or another
  microarchitectural factor is responsible for the crossover.

These interpretations are therefore hypotheses about the observed
performance behavior, not profiler-confirmed classifications.

## 9. Limitations & possible next steps

* Only two knobs were swept (batch size and precision); a runtime-choice
  knob (ONNX Runtime) was scoped out per §7.
* The compute- vs. memory-bound explanation in §8.4 is inferred from
  the shape of the latency/throughput curves, not confirmed with a
  hardware profiler (e.g., `nsight-compute` on the GPU or `perf` on
  the CPU). Hardware-counter profiling would be a natural extension.
* The reported results come from a single benchmark sweep per
  configuration rather than repeated independent process launches.
  Within each batch-size measurement, warm-up iterations are excluded
  and the reported latency statistics are computed from repeated
  timed calls. The CPU benchmark uses **10 warm-up iterations and
  50 timed iterations per batch size**.
* CPU peak memory is not reported (§7); a `/proc`-based RSS measurement
  would allow a CPU-side memory comparison.
* The GPU and CPU experiments use different hardware, so the results
  should be interpreted as **within-platform comparisons** rather than
  as a direct CPU-versus-GPU performance ranking.
* The benchmark uses synthetic fixed-shape inputs. Therefore, the
  latency and throughput results characterize a controlled inference
  workload with sequence length 128 rather than the complete
  distribution of sequence lengths found in natural IMDb reviews.
* The accuracy evaluation uses the real IMDb test set, but the
  measured accuracy difference between FP32 and INT8 is small enough
  that repeated evaluation across additional seeds or datasets would
  be needed to establish generality.

## 10. Individual-contribution statement

Both members contributed to the design, implementation, evaluation,
and write-up. Contribution was not perfectly even; by mutual
agreement, **Bakhom Ramzy** carried a larger share of the work overall.

| Area                                                                   | Bakhom Ramzy | Fatma Haddad |
| ---------------------------------------------------------------------- | ------------ | ------------ |
| System design & systems question                                       | Lead         | Contributed  |
| Implementation (data/model/train/benchmark/quantization/analysis code) | Lead         | Contributed  |
| Evaluation & interpretation (roofline reasoning, plots)                | Lead         | Contributed  |
| Report writing                                                         | Contributed  | Contributed  |
| Presentation                                                           | Contributed  | Contributed  |

*Agreed and signed by both members: Bakhom Ramzy, Fatma Haddad.*

## 11. AI-use acknowledgment

Generative AI (Claude) was used during this project, consistent with
the course AI policy: for **debugging specific, localized errors**
(e.g., stack traces from tokenization/dataloader/quantization calls)
and for **writing repetitive, boilerplate code** (e.g., CSV/JSON
serialization in the benchmark harness, matplotlib plotting
boilerplate across the three comparison plots, and argparse
scaffolding).

It was not used to generate the system design, the choice of
archetype, the systems question, the evaluation methodology, or the
analysis/conclusions in §8–9, which are the team's own. Both members
can explain every line of the submitted code.

## 12. References

* Wolf et al., *Transformers: State-of-the-Art Natural Language
  Processing*, Hugging Face.
* Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL
  2011 (IMDb dataset).
* Sanh et al., *DistilBERT, a distilled version of BERT*, arXiv:1910.01108.
* PyTorch dynamic quantization documentation, https://pytorch.org
