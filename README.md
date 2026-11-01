# Adaptive Batching for GPU Inference

A C++17 GPU inference serving experiment comparing static batching and two queue-aware adaptive policies.

**Status:** Initial implementation, 84-run comparison, GPU monitoring, and result analysis completed.

## Overview

Batching can improve inference efficiency, but waiting to assemble a batch also increases request latency.

This project explores that trade-off using a small wafer-defect CNN, ONNX Runtime with CUDA, and four synthetic request workloads.

The main finding is that adaptive policies reduced long batch-formation delays relative to large static batches, but did not consistently outperform Static batch 1 at the tested arrival rates.

## Research Question

Under what workload conditions does adaptive batching outperform static batching, and how does batch-formation delay affect tail latency, throughput, and GPU utilization?

## Serving Pipeline

A producer thread generates requests and places them in a thread-safe queue. The serving loop collects a batch, runs inference, and records request-level metrics.

```mermaid
flowchart TD
    A["Generator"] --> B["Request Queue"]
    B --> C["Batcher"]
    C --> D["ORT / CUDA"]
    D --> E["Metrics"]
```

The same preprocessed image is copied into each request. This is an in-process serving simulation rather than an HTTP or gRPC service.

## Batching Policies

| Policy | Batch selection | Dispatch condition |
| --- | --- | --- |
| Static | Fixed size: 1, 4, 8, 16, or 32 | Full batch or queue closure |
| Adaptive wait | Queue-based target, starting at up to 4 and increasing to a maximum of 32 | Target reached, oldest request reaches the configured timeout, or queue closure |
| Adaptive immediate | Selects a supported size from queued requests, up to 32 | No intentional accumulation wait |

Adaptive wait uses a configured maximum wait of **10 ms**. This is not a hard latency guarantee: thread scheduling and other server work can delay dispatch.

## Workloads

Each workload has a configured generator duration of 10 seconds.

| Workload | Target arrival rate | Pattern |
| --- | ---: | --- |
| Low | 2 requests/s | Continuous |
| Medium | 20 requests/s | Continuous |
| High | 200 requests/s | Continuous |
| Bursty | 200 requests/s during active periods | 500 ms active, 1500 ms inactive |

Bursty traffic has an ideal full-window average of 50 requests/s. Its active-phase target should not be interpreted as a continuous 200 requests/s workload.

Requests use absolute scheduled arrival times. Actual arrival time and generation lag are also recorded because Windows scheduling can cause delayed or grouped arrivals.

## Environment

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1660 SUPER, 6 GB |
| Operating system | Windows, WDDM |
| Language | C++17 |
| Compiler | MSVC, Visual Studio Build Tools 2026 |
| Build system | CMake |
| Inference runtime | ONNX Runtime GPU 1.26.0 |
| CUDA | 12.6 |
| cuDNN | 9.8 |
| OpenCV | 4.14.0 |
| Model precision | FP32 |
| Model input | `[N, 1, 24, 24]` |

Preprocessing loads a grayscale image, converts it to FP32, divides by its maximum pixel value when positive, and resizes it to 24 × 24 using linear interpolation.

## Metrics

Each request records:

- Queue wait: actual arrival to service start
- Service time: service start to inference completion
- End-to-end latency: actual arrival to completion
- Arrival offset and generation lag
- Actual batch size

Service time includes batch-input assembly, tensor setup, and `session.Run`. It is not isolated GPU kernel execution time.

Throughput is calculated as:

```text
completed requests / (last completion time - first actual arrival time)
```

This interval can differ from the configured 10-second generator duration.

## Experimental Method

The comparison contains:

- Four workloads
- Five static batch sizes
- Two adaptive policies
- Three repeats per configuration
- **84 runs in total**

Before measured requests, every batch size from 1 through 32 is warmed up three times.

The original Static/Adaptive wait comparison contains 72 runs. Adaptive immediate adds 12 runs collected in a separate session.

Reported latency values are means of run-level percentiles. They are not percentiles calculated by pooling all requests.

## Results

### P95 Latency Across Policies

![P95 latency across all policies](results/analysis/p95_all_policies.png)

The vertical axis uses a logarithmic scale.

### Comparison with Static Batch 1

![Focused P95 comparison](results/analysis/p95_focus.png)

Error bars show sample standard deviation across three runs, not confidence intervals.

| Workload | Static 1 P95 | Adaptive wait P95 | Adaptive immediate P95 |
| --- | ---: | ---: | ---: |
| Low | 1.912 ms | 17.871 ms | 1.890 ms |
| Medium | 1.915 ms | 17.677 ms | 1.790 ms |
| High | 4.569 ms | 21.031 ms | 5.933 ms |
| Bursty | 4.152 ms | 21.898 ms | 6.250 ms |

Adaptive immediate was close to Static 1 on low and medium workloads. These small differences do not establish reliable superiority.

Static 1 had lower mean P95 latency on high and bursty workloads.

### Queue Wait

![Mean queue wait](results/analysis/queue_wait.png)

Large static batches created substantial waiting under sparse arrivals.

For example, Low generated only 20 requests, fewer than Static 32 required for a full batch. Requests waited until queue closure, producing a P95 of approximately **9.53 seconds**. Adaptive wait reduced this to approximately **17.87 ms**.

This comparison includes both batch-selection and timeout differences.

### Arrival Rate and Throughput

![Arrival rate, P95 latency, and throughput](results/analysis/arrival_rate_metrics.png)

At the high workload, Static 1, Adaptive wait, and Adaptive immediate all completed approximately 200 requests/s.

These values follow the offered load. They do not establish the system's maximum processing capacity or a throughput advantage from batching.

Bursty is excluded from the continuous arrival-rate curves because its duty cycle differs.

### GPU Monitoring

![GPU utilization under high workload](results/analysis/gpu_high.png)

GPU monitoring was performed in separate high-workload executions.

| Policy | Mean sampled GPU utilization | Maximum sampled GPU utilization |
| --- | ---: | ---: |
| Static 1 | 2.50% | 10% |
| Adaptive wait | 1.29% | 9% |
| Adaptive immediate | 2.64% | 12% |

These are single monitored runs per policy, not GPU measurements from all 84 benchmark runs.

`nvidia-smi` was polled every 100 ms. Measurements describe the whole GPU and use the device's rolling utilization metric.

The results are consistent with low GPU activity at the tested load. They do not establish a reliable utilization ranking between policies.

### Nsight Systems

Nsight Systems confirmed CUDA kernel execution and visible gaps between kernel activity.

One selected cuDNN convolution kernel took **13.216 μs**. This is one kernel's execution time rather than full-model inference latency.

Application queue waits were not explicitly annotated, so the gaps cannot be attributed to a single source of delay.

## Interpretation

- Adaptive policies reduced long waits relative to large fixed batches.
- Intentional batch accumulation increased latency at the tested loads without a demonstrated throughput benefit.
- Immediate dispatch avoided most accumulation delay but did not consistently beat Static batch 1.
- Sustained GPU saturation was not observed in the monitored high-workload runs.
- A workload regime where adaptive batching outperforms Static 1 has not yet been demonstrated.

The experiment shows why batching decisions must account for workload intensity and waiting costs.

## Initial Hypotheses

**H1: Adaptive batching reduces tail latency relative to large static batches under low or bursty traffic.**

Supported relative to large fixed batches, with an important qualification: timeout behavior also differs between the policies.

**H2: The benefit of adaptation diminishes under sustained high load.**

Not established. The workload named “high” represents 200 requests/s in this experiment; it does not demonstrate system saturation.

## Build and Run

### Prerequisites

Install:

- Visual Studio Build Tools with C++ support
- CMake
- OpenCV
- ONNX Runtime GPU
- Compatible CUDA and cuDNN libraries
- Python for experiment and analysis scripts

Update the local dependency paths in `CMakeLists.txt` before building.

Required input files:

```text
models/wafer_fault_cnn.onnx
models/wafer_fault_cnn.onnx.data
test_images/sample.png
```

Both ONNX files are required because the model uses external tensor data.

### Build

Run from the repository root:

```bat
cmake -S . -B build
cmake --build build --config Release
```

### Run One Configuration

```bat
build\Release\p3_inference.exe high static 1 metrics_static_test.csv
```

```bat
build\Release\p3_inference.exe high adaptive 32 metrics_adaptive_test.csv
```

```bat
build\Release\p3_inference.exe high adaptive_immediate 32 metrics_high_immediate_test.csv
```

Arguments:

```text
p3_inference.exe <workload> <mode> <batch_size> <csv_path>
```

- Workload: `low`, `medium`, `high`, `bursty`
- Mode: `static`, `adaptive`, `adaptive_immediate`
- Batch size: `1`, `4`, `8`, `16`, `32`

Run from the repository root so relative model and image paths resolve correctly.

### Run the Comparison Experiments

```bat
python scripts\run_experiments.py
python scripts\run_immediate_experiments.py
python scripts\merge_results.py
```

Existing result directories may need to be moved before rerunning scripts that prevent overwriting previous experiments.

### Monitor GPU Utilization

```bat
python scripts\measure_gpu.py high static 1
python scripts\measure_gpu.py high adaptive 32
python scripts\measure_gpu.py high adaptive_immediate 32
```

For another run, select an unused run number:

```bat
python scripts\measure_gpu.py high adaptive_immediate 32 --run 3
```

### Regenerate Figures

```bat
python -m pip install --user matplotlib numpy
python scripts\plot_results.py
```

Figures are written to `results/analysis/` as PNG and SVG files. This step does not rerun inference.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `include/` | Request types, queue, and component interfaces |
| `src/` | Generator, batching policies, inference, metrics, and main |
| `models/` | ONNX model and external tensor data |
| `test_images/` | Sample input image |
| `scripts/` | Experiments, GPU monitoring, and plotting |
| `results/comparison/` | Original 72-run comparison |
| `results/immediate/` | Additional 12 immediate-policy runs |
| `results/combined/` | Combined summaries |
| `results/gpu/` | Separate GPU-monitoring outputs |
| `results/analysis/` | Figures and derived GPU summary |
| `docs/day13_results.md` | Detailed analysis and measurement limitations |

## Limitations

- The model is small, and the tested arrival rates did not demonstrate sustained GPU saturation.
- Only three benchmark repeats were collected per setting.
- Low workloads contain only 20 requests per run, limiting interpretation of extreme percentiles.
- Adaptive immediate was tested in a separate session.
- Windows scheduling affects arrival spacing and timeout wake-ups.
- Static and adaptive policies differ in timeout behavior as well as batch selection.
- CSV writing can affect subsequent dispatch timing.
- Preprocessing is performed before request generation and is excluded from serving latency.
- Throughput uses the first-arrival-to-last-completion interval, which can affect comparisons under short or bursty workloads.
- GPU utilization was measured only for three high-workload configurations, once each.
- The pipeline reuses one input image and does not include network serving.
- A precise CPU, transfer, or GPU bottleneck has not been established.

See [the detailed analysis](docs/day13_results.md) for further discussion.

## Connection to Previous Projects

- **P0:** Train a wafer-defect CNN and export it to ONNX.
- **P1:** Run inference using a standalone C++ pipeline.
- **P2:** Profile inference and input-processing costs.
- **P3:** Study request queues, batching policies, latency, and GPU activity.

## Related Work

The project was motivated by reading inference-serving and scheduling research, including Sarathi-Serve.

This implementation is a small CNN serving experiment, not a reproduction of an LLM serving system.

## Future Work

- Define an explicit latency SLO.
- Add request deadlines and slack estimates.
- Implement an SLO-aware batching policy.
- Compare Static, Adaptive, and SLO-aware scheduling.
- Evaluate higher offered loads to investigate saturation.

## Completion Status

- [x] Request generator and thread-safe queue
- [x] Static batching baselines
- [x] Queue-aware adaptive policies
- [x] Maximum-wait dispatch
- [x] Four workload patterns
- [x] Request-level timing and CSV metrics
- [x] Warm-up
- [x] 84-run comparison
- [x] Separate GPU monitoring
- [x] Nsight Systems inspection
- [x] Graphs and result analysis
- [ ] SLO-aware scheduling extension
