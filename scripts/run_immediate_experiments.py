import csv
import random
import subprocess

from run_experiments import ROOT, EXE, summarize


def main():
    if not EXE.is_file():
        raise FileNotFoundError(f"Executable not found: {EXE}")

    results_dir = ROOT / "results" / "immediate"
    results_dir.mkdir(parents=True, exist_ok=True)

    workloads = ["low", "medium", "high", "bursty"]
    repeats = 3
    rng = random.Random(42)

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

    summary_path = results_dir / "summary.csv"

    completed_runs = 0
    total_runs = len(workloads) * repeats

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=fields,
        )
        writer.writeheader()

        for repeat in range(1, repeats + 1):
            jobs = workloads.copy()
            rng.shuffle(jobs)

            for workload in jobs:
                name = (
                    f"{workload}_adaptive_immediate"
                    f"_b32_run{repeat}"
                )

                csv_path = results_dir / f"{name}.csv"
                log_path = results_dir / f"{name}.log"

                completed_runs += 1

                print(
                    f"[{completed_runs}/{total_runs}] {name}",
                    flush=True,
                )

                command = [
                    str(EXE),
                    workload,
                    "adaptive_immediate",
                    "32",
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
                    "mode": "adaptive_immediate",
                    "configured_batch_size": 32,
                    "max_wait_ms": 0,
                    "repeat": repeat,
                    **summarize(csv_path),
                    "request_csv": csv_path.name,
                }

                writer.writerow(result)
                summary_file.flush()

                print(
                    f"  Requests={result['completed_requests']}"
                    f" | P95={result['p95_ms']:.3f} ms"
                    f" | Throughput="
                    f"{result['throughput_rps']:.3f} requests/sec",
                    flush=True,
                )

    print(f"\nCompleted: {completed_runs}/{total_runs}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()