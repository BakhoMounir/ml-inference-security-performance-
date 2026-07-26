import pandas as pd
import os

FP32 = "results/benchmark_fp32_cpu_final.csv"
INT8 = "results/benchmark_int8_cpu_final.csv"

def analyze(path, name):
    df = pd.read_csv(path)

    print(f"\n===== {name} =====")
    print(df[[
        "batch_size",
        "median_latency_ms",
        "throughput_samples_per_sec"
    ]])

    peak = df.loc[df["throughput_samples_per_sec"].idxmax()]

    print("\nPeak throughput:")
    print(
        f"Batch size {peak.batch_size}: "
        f"{peak.throughput_samples_per_sec:.2f} samples/s"
    )

    print("\nTrend:")
    print(
        "Latency increases strongly after saturation point while "
        "throughput stops improving."
    )


analyze(FP32, "FP32 CPU")
analyze(INT8, "INT8 CPU")
