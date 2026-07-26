import pandas as pd


FP32_FILE = "results/benchmark_fp32_cpu_final.csv"
INT8_FILE = "results/benchmark_int8_cpu_final.csv"


def main():

    fp32 = pd.read_csv(FP32_FILE)
    int8 = pd.read_csv(INT8_FILE)

    print("\n=== FP32 CPU Benchmark ===")
    print(fp32.to_string(index=False))

    print("\n=== INT8 CPU Benchmark ===")
    print(int8.to_string(index=False))


    merged = fp32.merge(
        int8,
        on="batch_size",
        suffixes=("_fp32", "_int8")
    )


    merged["throughput_speedup"] = (
        merged["throughput_samples_per_sec_int8"]
        /
        merged["throughput_samples_per_sec_fp32"]
    )

    merged["latency_speedup"] = (
        merged["median_latency_ms_fp32"]
        /
        merged["median_latency_ms_int8"]
    )


    summary = merged[
        [
            "batch_size",
            "throughput_samples_per_sec_fp32",
            "throughput_samples_per_sec_int8",
            "throughput_speedup",
            "median_latency_ms_fp32",
            "median_latency_ms_int8",
            "latency_speedup"
        ]
    ]


    print("\n=== INT8 Speedup Summary ===")
    print(summary.to_string(index=False))


    summary.to_csv(
        "results/int8_speedup_summary.csv",
        index=False
    )


if __name__ == "__main__":
    main()
