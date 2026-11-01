# SLO-aware batching: pre-experiment definition

This extension adds a `slo` mode while retaining Static, Adaptive wait, and Adaptive immediate as comparison baselines.

## Fixed evaluation target

- Request latency budget: **20 ms**, from actual request creation/arrival to inference completion.
- Run-level requirement: **at least 95%** of completed requests finish within their individual deadlines.
- A request finishing exactly at its deadline counts as met.
- Safety margin for scheduling: **2 ms**.

There is no external production requirement. This is an experimental target chosen after observing the original baseline P95 values of roughly 1.9–4.6 ms for Static 1, and before collecting SLO-policy results. It allows room above baseline inference while making batch-formation delay relevant. The target and margin must remain fixed during the initial comparison; changes should be evaluated as separately labeled experiments.

## Margin adjustment history

The text and formulas in this document are the **initial pre-experiment definition**, which used a **2 ms** safety margin. They are kept unchanged as a record.

- The initial 2 ms margin was tried in a Low-workload smoke test, which failed the 95% requirement.
- After that observation the margin was adjusted to **10 ms**. This is exploratory policy development, not an untouched holdout evaluation.
- The 96-run comparison (see [slo_results.md](slo_results.md)) and the four-policy GPU monitoring (see [slo_gpu_results.md](slo_gpu_results.md)) both used the fixed 10 ms margin. The code default in `include/slo_config.hpp` is 10 ms.
- Wherever this document says 2 ms (including the `latest_dispatch` formula below), read 10 ms for the reported experiments.

The queue is FIFO, requests use the same budget, and one producer creates them in arrival order. Therefore the front request has the earliest deadline. This implementation is not a heterogeneous-deadline/EDF scheduler.

## Deadline and slack

```text
deadline = actual_arrival + 20 ms
slack = deadline - now - estimated_service_time
latest_dispatch = deadline - estimated_service_time - 2 ms
```

The safety margin is subtracted when deciding how long to wait, not from the reported latency target or the slack column. A positive raw slack smaller than 2 ms is already considered urgent by this policy.

Deadline origin matches the existing code's `Request` construction time. Copying the input into the request and queue insertion occur afterwards and consume budget. Scheduled generator arrival is only used to measure generation lag; it is not the deadline origin.

## Service-time estimate

For every batch size from 1 through 32:

1. Run the existing three warm-up calls.
2. Collect ten additional service-time observations, excluding all these requests from serving metrics.
3. Retain the maximum observed service time for that batch size.
4. Apply a prefix maximum over sizes: estimate for size N is at least the maximum estimate for all smaller sizes.

This produces a conservative monotonic envelope: reserving time for a target batch also covers the calibrated estimates for a smaller partial batch. It is a heuristic, not a probabilistic latency bound. Ten observations can miss later spikes, and a single calibration outlier can cause overestimation. The model is frozen for the measured run; there is no feedback-based online estimator in this version.

Service time retains the existing definition: batch assembly, tensor creation, and `session.Run`, ending at inference completion. It is not pure GPU kernel time. The profile is saved next to each request CSV as `<request_csv>.service_times.csv`.

All four modes use the same new warm-up/calibration procedure. Existing policy algorithms are unchanged. For Day 17, rerun comparison baselines using this executable rather than mixing new SLO results with the earlier 84-run session as though they were controlled contemporaneous measurements.

## Dispatch policy

After the queue becomes nonempty, select a target using the same ceiling-style queue-size policy as Adaptive wait: start at min(4, configured maximum), double until covering the observed queue length, and cap at the configured maximum.

Reserve the calibrated service time for that target. Under the queue mutex, wait until any of the following:

- The queue contains the target number of requests.
- The oldest request reaches `latest_dispatch`.
- The generator closes the queue.

Pop up to the target size and execute. An already urgent or overdue front request is dispatched without an additional intentional wait. When a partial batch is returned, reporting uses the estimate for its actual size. The reported slack at service start may therefore differ from the target-size slack used to reserve time.

The absolute wait deadline is fixed from the oldest request; incoming requests do not reset it. Condition-variable waiting releases the mutex so the producer can continue to enqueue. No Windows timer-resolution changes or busy waiting are introduced.

## CSV and summary

Original columns remain in their existing order. Appended fields are:

| Field | Meaning |
| --- | --- |
| `mode` | static, adaptive, adaptive_immediate, or slo |
| `slo_target_ms` | Fixed request budget |
| `slo_required_percent` | Required run-level attainment |
| `safety_margin_ms` | Configured scheduling margin (applied only in slo mode) |
| `deadline_offset_ms` | Deadline relative to first actual arrival |
| `estimated_service_ms` | Estimate for the actual dispatched batch size |
| `slack_at_service_start_ms` | Deadline minus recorded service-start time minus estimate |
| `slo_met` | 1 if completion <= deadline, otherwise 0 |

The summary prints target, met count, attainment, violations, and PASS/FAIL against 95%. Every mode is evaluated against the same target, although only `slo` schedules using it. No requests are deliberately dropped; the serving loop drains the queue on normal completion. An aborted run must not be treated as a completed SLO evaluation.

Low produces approximately 20 requests, so one additional miss changes attainment by about five percentage points. This is suitable for checking the implementation, not a strong statistical estimate of an operational SLO.

## Build and smoke checks

Run from the project root after copying the extension files:

```bat
cmake -S . -B build
cmake --build build --config Release
build\Release\p3_inference.exe low slo 32 metrics_low_slo_test.csv
python scripts\check_slo.py metrics_low_slo_test.csv
build\Release\p3_inference.exe high slo 32 metrics_high_slo_test.csv
python scripts\check_slo.py metrics_high_slo_test.csv
build\Release\p3_inference.exe low static 1 metrics_static_slo_test.csv
```

These are smoke checks; repeated controlled comparisons belong to Day 17. The original experiment scripts still enumerate their original policies. Adding `slo` to automated experiments and GPU-monitor argument validation is a separate Day 17 change. Never feed the `.service_times.csv` sidecars into a request-level summary parser.

## Validation scope

The supplied dependency-free C++ tests cover service-time envelopes, urgent partial batches, full-target dispatch, oldest-request wait boundaries, closure/draining, configured caps, baseline batching, negative slack, inclusive deadline equality, and SLO-count output. They compile against the real queue, batching and metrics implementations without ONNX Runtime or OpenCV.

Linux test compilation:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -pthread tests/test_slo.cpp src/slo_batcher.cpp src/static_batcher.cpp src/adaptive_batcher.cpp src/metrics.cpp src/request_generator.cpp -o /tmp/p3_slo_tests
/tmp/p3_slo_tests /tmp/p3_slo_metrics.csv
python scripts/check_slo.py /tmp/p3_slo_metrics.csv
```

Windows/MSVC linking, model loading, CUDA execution, and real SLO attainment must be validated on the target machine. A positive predicted slack or a 2 ms margin cannot guarantee a met deadline under OS delays, service-time estimation error, or backlog.
