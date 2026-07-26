# Archetype 2.2 Serving-Optimization Study Results

## Model
- DistilBERT fine-tuned on IMDb sentiment classification
- FP32 accuracy: 0.8702
- INT8 accuracy: 0.8710
- Accuracy difference: -0.0008

## FP32 CPU Benchmark
- Device: CPU
- Warmup iterations: 20
- Measurement iterations: 100
- Batch sizes: 1, 2, 4, 8, 16, 32, 64

## INT8 CPU Benchmark
- Dynamic INT8 quantization
- Same benchmark configuration as FP32

## Main Results

### FP32 Peak Throughput
- Batch size: 64
- Throughput: 63.32 samples/s

### INT8 Peak Throughput
- Batch size: 16
- Throughput: 100.58 samples/s

## INT8 Improvements

Maximum throughput improvement:
- Batch size 1: 5.04x

Best practical throughput:
- Batch size 16: 1.67x improvement

## Observations

Increasing batch size improves throughput until saturation.
After saturation, latency increases significantly while throughput stops improving.
This indicates increasing memory/system bottleneck effects at larger batch sizes.

INT8 quantization improves CPU inference efficiency while maintaining accuracy.
