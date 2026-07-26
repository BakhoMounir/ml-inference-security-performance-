import json
import pandas as pd

summary = {}

# Load accuracy results
with open("results/baseline_accuracy.json") as f:
    fp32_acc = json.load(f)

with open("results/int8_accuracy.json") as f:
    int8_acc = json.load(f)


# Load benchmark results
gpu = pd.read_csv("results/benchmark_fp32.csv")
fp32_cpu = pd.read_csv("results/benchmark_fp32_cpu_final.csv")
int8_cpu = pd.read_csv("results/benchmark_int8_cpu_final.csv")


# Accuracy comparison
summary["accuracy"] = {
    "FP32_accuracy_percent": fp32_acc["test_accuracy_percent"],
    "INT8_accuracy_percent": int8_acc["test_accuracy_percent"],
    "accuracy_difference_percent": (
        int8_acc["test_accuracy_percent"]
        - fp32_acc["test_accuracy_percent"]
    )
}


# Performance comparison
gpu_best = gpu.loc[gpu["throughput_samples_per_sec"].idxmax()]
fp32_cpu_best = fp32_cpu.loc[fp32_cpu["throughput_samples_per_sec"].idxmax()]
int8_cpu_best = int8_cpu.loc[int8_cpu["throughput_samples_per_sec"].idxmax()]


summary["performance"] = {
    "GPU_FP32": {
        "best_batch_size": int(gpu_best["batch_size"]),
        "peak_throughput_samples_per_sec": float(
            gpu_best["throughput_samples_per_sec"]
        ),
        "latency_ms": float(
            gpu_best["median_latency_ms"]
        )
    },

    "CPU_FP32": {
        "best_batch_size": int(fp32_cpu_best["batch_size"]),
        "peak_throughput_samples_per_sec": float(
            fp32_cpu_best["throughput_samples_per_sec"]
        ),
        "latency_ms": float(
            fp32_cpu_best["median_latency_ms"]
        )
    },

    "CPU_INT8": {
        "best_batch_size": int(int8_cpu_best["batch_size"]),
        "peak_throughput_samples_per_sec": float(
            int8_cpu_best["throughput_samples_per_sec"]
        ),
        "latency_ms": float(
            int8_cpu_best["median_latency_ms"]
        )
    }
}


# INT8 speedup calculation
summary["quantization"] = {
    "cpu_throughput_speedup_factor": round(
        int8_cpu_best["throughput_samples_per_sec"]
        / fp32_cpu_best["throughput_samples_per_sec"],
        3
    )
}


# Save summary
with open("results/final_summary.json", "w") as f:
    json.dump(summary, f, indent=4)


print(json.dumps(summary, indent=4))
