import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")

fp32 = pd.read_csv(RESULTS / "benchmark_fp32_cpu_final.csv")
int8 = pd.read_csv(RESULTS / "benchmark_int8_cpu_final.csv")

plt.figure(figsize=(8,5))
plt.plot(fp32["batch_size"], fp32["throughput_samples_per_sec"], marker="o", label="FP32 CPU")
plt.plot(int8["batch_size"], int8["throughput_samples_per_sec"], marker="o", label="INT8 CPU")
plt.xlabel("Batch Size")
plt.ylabel("Throughput (samples/sec)")
plt.title("FP32 vs INT8 Throughput")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS / "throughput_fp32_vs_int8.png", dpi=300)
plt.close()


plt.figure(figsize=(8,5))
plt.plot(fp32["batch_size"], fp32["median_latency_ms"], marker="o", label="FP32 CPU")
plt.plot(int8["batch_size"], int8["median_latency_ms"], marker="o", label="INT8 CPU")
plt.xlabel("Batch Size")
plt.ylabel("Median Latency (ms)")
plt.title("FP32 vs INT8 Latency")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS / "latency_fp32_vs_int8.png", dpi=300)
plt.close()


plt.figure(figsize=(8,5))
plt.plot(fp32["batch_size"], fp32["p95_latency_ms"], marker="o", label="FP32 CPU")
plt.plot(int8["batch_size"], int8["p95_latency_ms"], marker="o", label="INT8 CPU")
plt.xlabel("Batch Size")
plt.ylabel("p95 Latency (ms)")
plt.title("FP32 vs INT8 p95 Latency")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS / "p95_latency_fp32_vs_int8.png", dpi=300)
plt.close()


print("Plots generated successfully.")
