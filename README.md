# Batching Policies for ResNet-50 GPU Inference

A C++17 inference-serving experiment that compares four request-batching policies on ResNet-50 with ONNX Runtime and CUDA:

- **Static**: fixed batch size
- **Adaptive wait**: queue-based batch target with a maximum wait
- **Adaptive immediate**: queue-based batch size without waiting to accumulate
- **SLO-aware**: dispatch time computed from request deadlines and estimated service time

**Status:** the ResNet-50 model, a fixed 1,000-image ImageNet validation subset, the per-batch service-time profile, and the image pool connected to the request generator are done, and a 4-run pilot has been executed (`results/resnet50/pilot/`, not formal results; it predates the extra CSV columns and the generator fix below). **The formal comparison is configured but has not been run yet.**

Comparison settings (confirmed; defaults in `include/workload.hpp` and `include/slo_config.hpp`, the only place they live; every CSV row records the values actually used): Low 20 / Medium 100 / High 250 requests/s; Bursty 250 requests/s while active (500 ms on, 1500 ms off); requests are generated for 10 s and all remaining requests are then processed; SLO 100 ms, required attainment 95%, safety margin 10 ms; Static batch 1 / 4 / 8 / 16; Adaptive wait (max wait 10 ms), Adaptive immediate and SLO-aware with max batch 16; 4 workloads x 7 policies x 3 repeats = **84 runs**. The settings are not changed to make a policy pass. Options `--duration`, `--slo-ms`, `--margin-ms`, `--required-percent`, `--max-wait-ms` can override them for experiments such as the pilot.

## Background

The project started with a small wafer-defect CNN. That model is so cheap (about 1.5 ms per request) that batching never beat batch size 1 at the tested loads, so the experiments could not show when batching helps. It was replaced with ResNet-50, whose compute cost makes batching decisions matter. The wafer code, results, and analysis are not kept in the working tree; they remain in Git history (last wafer-era commits: `07052c8` for the first 84-run comparison, `857c8d8` for the SLO-aware extension and its 96-run comparison).

## Pipeline

A producer thread generates requests into a thread-safe queue. The serving loop collects a batch with the selected policy, runs inference, and records request-level metrics.

```mermaid
flowchart TD
    A["Generator"] --> B["Request Queue"]
    B --> C["Batcher"]
    C --> D["ORT / CUDA"]
    D --> E["Metrics"]
```

This is an in-process serving simulation, not an HTTP or gRPC service.

## Model and Data

| Item | Value |
| --- | --- |
| Model | torchvision ResNet-50, `IMAGENET1K_V2` weights, FP32 |
| ONNX | `models/resnet50/resnet50_v2_fp32.onnx` (+ `.onnx.data`), dynamic batch axis, input `[N,3,224,224]`, output `[N,1000]` |
| Data | 1,000 ImageNet validation images (Hugging Face `ILSVRC/imagenet-1k`, revision pinned), chosen with seed 42, preprocessed with the official V2 transform |
| Prepared file | `data/imagenet/val1000_resnet50_v2_nchw_f32.bin`, float32 little-endian, 602,112,000 bytes |

The model and data files are not in Git. ImageNet requires accepting its terms on Hugging Face and logging in locally.

Verification (4 real images, batch 1 and 4): PyTorch CPU, ONNX Runtime CPU, and the C++ ONNX Runtime CUDA path agree to a maximum absolute difference below 1e-5 (`allclose` with rtol 1e-3, atol 1e-4, same top-1). Labels were checked against the dataset's synset list, and the stored tensors equal the official transform output (`scripts/verify_resnet_inputs.py`).

## Batch Profile

Sequential back-to-back batches of the first N prepared images (no arriving traffic), 3 warm-up runs, then 10 timed runs per size, one process per size. Service time covers batch assembly, tensor setup, and `session.Run`. Details and raw timings are in `results/resnet50/batch_profile/`.

| Batch | Mean | Median | P95* | Observed sequential rate | Sampled GPU memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.20 ms | 6.18 ms | 6.38 ms | 161 img/s | 344 MiB |
| 4 | 17.08 ms | 16.87 ms | 18.26 ms | 234 img/s | 342 MiB |
| 8 | 29.25 ms | 29.12 ms | 30.03 ms | 273 img/s | 598 MiB |
| 16 | 52.42 ms | 51.85 ms | 54.66 ms | 305 img/s | 600 MiB |
| 32 | 97.03 ms | 97.04 ms | 99.89 ms | 330 img/s | 1112 MiB |

\*P95 of 10 runs is interpolated near the maximum. The rate is batch size divided by mean service time; it is not the throughput of a serving run with arriving requests. Memory is whole-GPU `nvidia-smi` memory sampled about every 100 ms, so it is a sampled value, not an exact peak. One measurement session on a GTX 1660 SUPER (6 GB); no repeats.

## Environment

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1660 SUPER, 6 GB (Windows, WDDM) |
| Build | C++17, MSVC (Visual Studio Build Tools), CMake |
| Inference runtime | ONNX Runtime GPU 1.26.0, CUDA 12.6, cuDNN 9.8 |
| Python (data/model preparation) | 3.12 environment `.venv-resnet`, versions in `scripts/requirements-resnet.txt` |

## Build and Run

Update the dependency paths at the top of `CMakeLists.txt`, then from the repository root:

```bat
cmake -S . -B build
cmake --build build --config Release
```

Prepare the model and data once (see `scripts/requirements-resnet.txt` for the environment):

```bat
.venv-resnet\Scripts\python.exe scripts\prepare_resnet50.py
.venv-resnet\Scripts\hf.exe auth login
.venv-resnet\Scripts\python.exe scripts\prepare_imagenet_subset.py
.venv-resnet\Scripts\python.exe scripts\verify_resnet_inputs.py
```

Measure the batch profile (runs `batch_profile` once per batch size):

```bat
python scripts\profile_batches.py
```

Command line:

```text
p3_inference.exe <workload> <static|adaptive|adaptive_immediate|slo> <max_batch> <csv_path> [options]
p3_inference.exe <any> model_check <1|4> <logits_path>
p3_inference.exe <any> batch_profile <1|4|8|16|32> <csv_path>
```

Serving loads the whole prepared subset into RAM once, calibrates service time for every batch size from 1 up to `max_batch` (partial batches can occur), then generates requests: request id `k` always carries image `(k-1) % 1000`, so all policies see the same image for the same id. Each request copies its image into `Request.input` after its arrival time is stamped, so the copy and queue push count against its latency. Logits are never copied while serving. Pilot run (4 jobs, explicit order):

```bat
python scripts\run_slo_experiments.py --max-batch 16 --duration 5 --repeats 1 --output results\resnet50\pilot --only low:static:1 high:static:1 high:static:16 high:slo:16
```

Run from the repository root so relative paths resolve. Scheduler logic is covered by `tests/test_slo.cpp`, a dependency-free test that is not part of the CMake build; compile it with the queue, batcher, metrics, and generator sources and pass a temporary CSV path as its argument.

## Measurement Notes

- **Generator**: a planned arrival slot is always generated, even if the thread wakes after the window ended, so request id `k` has the same image and the same planned arrival offset in every run. `scheduled`, `generated`, and `not generated` counts are logged per run; each generated request must complete exactly once.
- **Drain** is `max(0, last completion - end of the generation window)`. The signed value (negative when the last request finished before the window ended) is logged separately as `last_completion_minus_generation_end_ms`.
- **Slack**: `slack_at_service_start_ms` is `deadline - service start - estimated service time`. A negative value means the request was already past that point when service began, which includes queue wait; it is not by itself a service-time estimation error. The estimate is compared with reality separately in `service_minus_estimate_ms` (measured batch service time minus its calibration estimate).
- **Arrival timing**: actual arrivals are not assumed to be the same in every run. Generation lag (actual minus planned arrival) and arrival gaps are summarized per run. In the pilot, lag averaged 7-8 ms with a maximum near 15.7 ms and many arrivals followed each other less than 1 ms apart. That is what was observed; operating-system timer granularity is a hypothesis that has not been tested, and no timer change or busy waiting is used.
- **Aggregation**: percentiles are computed per run, then averaged (with sample SD) over runs; they are not pooled over requests of different runs. GPU monitoring is not part of the 84 runs and is run separately.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `include/`, `src/` | Request queue, generator, Static/Adaptive/SLO batchers, service-time estimator, metrics, inference engine, dataset reader, model config |
| `scripts/prepare_resnet50.py`, `prepare_imagenet_subset.py`, `verify_resnet_inputs.py` | Model export, data preparation, and input verification |
| `scripts/profile_batches.py` | Per-batch service-time profile |
| `scripts/run_slo_experiments.py`, `measure_gpu.py`, `check_slo.py`, `check_metrics.py` | Policy comparison, GPU monitoring, and CSV checks. They read the settings recorded in the CSVs instead of assuming SLO numbers |
| `scripts/plot_slo_results.py`, `plot_slo_gpu.py` | Plots from the wafer stage, **not yet adapted** to the ResNet-50 matrix and CSV columns; rewrite them when the formal results exist |
| `docs/slo_policy.md` | SLO-aware policy definition and margin history |
| `results/resnet50/` | ResNet-50 measurements |
| `models/`, `data/` | Model and dataset files (ignored by Git) |

## Next Steps

- Run the 84-run comparison: `python scripts\run_slo_experiments.py --max-batch 16 --repeats 3` (writes `results/resnet50/comparison/`; refuses to overwrite).
- Adapt the plot scripts to the results, and run GPU monitoring separately.

## Completion Status

- [x] Request generator, thread-safe queue, and four batching policies
- [x] Request-level timing metrics and SLO accounting
- [x] ResNet-50 ONNX export and real-image input verification (PyTorch / ORT CPU / C++ CUDA)
- [x] Fixed 1,000-image ImageNet validation subset
- [x] Per-batch service-time profile (single session)
- [x] Image pool connected to the generator (fixed order, shared by all policies)
- [x] Settings generalized and recorded in every CSV; 4-run pilot executed
- [x] Workloads, maximum batch size, and SLO confirmed for ResNet-50
- [ ] 84-run comparison executed
- [ ] Plot scripts adapted
- [ ] ResNet-50 policy comparison (not started)
