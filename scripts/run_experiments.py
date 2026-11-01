import csv
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "Release" / "p3_inference.exe"

RESULTS = ROOT / "results" / "comparison"
RESULTS.mkdir(parents=True, exist_ok=True)

WORKLOADS = ["low", "medium", "high", "bursty"]
POLICIES = [
    ("static", 1),
    ("static", 4),
    ("static", 8),
    ("static", 16),
    ("static", 32),
    ("adaptive", 32),
]

REPEATS = 3


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def summarize(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"No requests recorded: {csv_path}")

    latencies = [
        float(row["e2e_latency_ms"])
        for row in rows
    ]

    waits = [
        float(row["queue_wait_ms"])
        for row in rows
    ]

    first_arrival_ms = min(
        float(row["arrival_offset_ms"])
        for row in rows
    )

    last_finish_ms = max(
        float(row["arrival_offset_ms"])
        + float(row["e2e_latency_ms"])
        for row in rows
    )

    elapsed_seconds = (
        last_finish_ms - first_arrival_ms
    ) / 1000

    return {
        "completed_requests": len(rows),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_queue_wait_ms": statistics.mean(waits),
        "throughput_rps": len(rows) / elapsed_seconds,
    }


def main():
    if not EXE.is_file():
        raise FileNotFoundError(f"Build the executable first: {EXE}")

    rng = random.Random(42)
    total_runs = len(WORKLOADS) * len(POLICIES) * REPEATS
    completed_runs = 0
    all_results = []

    fields = [
        "workload",
        "mode",
        "configured_batch_size",
        "max_wait_ms",
        "repeat",
        "completed_requests",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_queue_wait_ms",
        "throughput_rps",
        "request_csv",
    ]

    summary_path = RESULTS / "summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()

        for repeat in range(1, REPEATS + 1):
            jobs = [
                (workload, mode, batch_size)
                for workload in WORKLOADS
                for mode, batch_size in POLICIES
            ]
            rng.shuffle(jobs)

            for workload, mode, batch_size in jobs:
                name = (
                    f"{workload}_{mode}_b{batch_size}"
                    f"_run{repeat}"
                )

                csv_path = RESULTS / f"{name}.csv"
                log_path = RESULTS / f"{name}.log"

                completed_runs += 1

                print(
                    f"[{completed_runs}/{total_runs}] {name}",
                    flush=True,
                )

                command = [
                    str(EXE),
                    workload,
                    mode,
                    str(batch_size),
                    str(csv_path),
                ]

                with log_path.open(
                    "w", encoding="utf-8"
                ) as log_file:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=180,
                    )

                result = {
                    "workload": workload,
                    "mode": mode,
                    "configured_batch_size": batch_size,
                    "max_wait_ms": 10 if mode == "adaptive" else "",
                    "repeat": repeat,
                    **summarize(csv_path),
                    "request_csv": csv_path.name,
                }

                writer.writerow(result)
                summary_file.flush()
                all_results.append(result)

                print(
                    f"  P95={result['p95_ms']:.3f} ms"
                    f" | throughput="
                    f"{result['throughput_rps']:.3f} requests/sec",
                    flush=True,
                )

    groups = defaultdict(list)

    for result in all_results:
        key = (
            result["workload"],
            result["mode"],
            result["configured_batch_size"],
        )
        groups[key].append(result)

    metric_names = [
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_queue_wait_ms",
        "throughput_rps",
    ]

    comparison_fields = [
        "workload",
        "mode",
        "configured_batch_size",
        "repeats",
    ]

    for metric in metric_names:
        comparison_fields.extend([
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_cv_percent",
        ])

    comparison_path = RESULTS / "comparison.csv"

    with comparison_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=comparison_fields
        )
        writer.writeheader()

        for key, runs in sorted(groups.items()):
            workload, mode, batch_size = key

            row = {
                "workload": workload,
                "mode": mode,
                "configured_batch_size": batch_size,
                "repeats": len(runs),
            }

            for metric in metric_names:
                values = [run[metric] for run in runs]
                mean = statistics.mean(values)
                std = statistics.stdev(values)

                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_cv_percent"] = (
                    std / mean * 100 if mean != 0 else ""
                )

            writer.writerow(row)

    print(f"\nPer-run results: {summary_path}")
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    main()