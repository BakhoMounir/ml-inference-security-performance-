import pandas as pd

fp32 = pd.read_csv("results/benchmark_fp32_cpu_final.csv")
int8 = pd.read_csv("results/benchmark_int8_cpu_final.csv")

print("\n===== FP32 Summary =====")
print(fp32)

print("\n===== INT8 Summary =====")
print(int8)

merged = fp32.merge(
    int8,
    on="batch_size",
    suffixes=("_fp32", "_int8")
)

merged["throughput_speedup"] = (
    merged["throughput_samples_per_sec_int8"] /
    merged["throughput_samples_per_sec_fp32"]
)

merged["latency_speedup"] = (
    merged["median_latency_ms_fp32"] /
    merged["median_latency_ms_int8"]
)

print("\n===== INT8 Improvement =====")
print(
    merged[
        [
            "batch_size",
            "throughput_speedup",
            "latency_speedup"
        ]
    ]
)

merged.to_csv(
    "results/final_fp32_vs_int8_summary.csv",
    index=False
)

print("\nSaved results/final_fp32_vs_int8_summary.csv")
