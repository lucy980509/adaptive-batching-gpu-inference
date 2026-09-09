# adaptive-batching-gpu-inference

> Research Project — In Progress

## Overview

GPU inference serving involves a fundamental trade-off between throughput and latency. 
Large batches can improve GPU utilization and throughput, but may increase request latency, 
especially under low or bursty workloads.

This project investigates when adaptive batching provides meaningful benefits over static 
batching under varying request workloads.

## Research Question

**Under what workload conditions does adaptive batching outperform static batching in GPU 
inference serving, and what are the resulting trade-offs between tail latency, throughput, 
and GPU utilization?**

## Hypotheses

### H1 — Adaptive batching under variable workloads

Adaptive batching will reduce tail latency relative to large static batch configurations 
under low or bursty request rates, while maintaining comparable throughput under high 
request rates.

### H2 — Diminishing benefits under sustained high load

The benefit of adaptive batching will diminish when request arrival rates are consistently 
high, because batching opportunities become predictable and static batching can achieve 
similar GPU utilization and throughput without runtime adaptation.

## Experimental Design

The serving pipeline will consist of:

```text
Request Generator
        ↓
Request Queue
        ↓
Batching Policy
        ↓
ONNX Runtime / CUDA
        ↓
GPU
```

Two batching strategies will be compared:

**Static batching**
- Batch sizes: 1, 4, 8, 16, 32

**Adaptive batching**
- Dynamically adjusts batch size based on the current request queue.

Experiments will evaluate both policies under:

- Low request load
- Medium request load
- High request load
- Bursty request load

### Controlled Setup

To isolate the effect of the batching policy, all experiments will use the same model, GPU, runtime configuration, and input processing pipeline. The primary experimental variable will be the batching policy (static vs. adaptive).

## Metrics

The primary evaluation metrics will be:

- Throughput (requests/second)
- P50 latency
- P95/P99 tail latency
- GPU utilization

## Related Work

The experimental design is informed by prior work on ML inference serving and scheduling, 
particularly the throughput–latency trade-offs studied in Sarathi-Serve.

## Status

- [x] Research question
- [x] Hypotheses
- [x] Initial related-work review
- [x] Experimental design
- [ ] Request generator and queue implementation
- [ ] Static batching baseline
- [ ] Adaptive batching policy
- [ ] Workload generation
- [ ] Performance experiments
- [ ] Results and analysis
