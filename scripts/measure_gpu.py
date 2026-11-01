import argparse
import csv
import json
import statistics
import subprocess
from datetime import datetime

from run_experiments import ROOT, EXE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workload",
        choices=["low", "medium", "high", "bursty"],
    )
    parser.add_argument(
        "mode",
        choices=["static", "adaptive", "adaptive_immediate"],
    )
    parser.add_argument("batch_size", type=int, choices=[1, 4, 8, 16, 32])
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    name = (
        f"{args.workload}_{args.mode}"
        f"_b{args.batch_size}_run{args.run}"
    )

    output_dir = ROOT / "results" / "gpu" / name
    output_dir.mkdir(parents=True, exist_ok=False)

    request_path = output_dir / "requests.csv"
    raw_path = output_dir / "gpu_raw.csv"
    samples_path = output_dir / "gpu_samples.csv"
    log_path = output_dir / "inference.log"

    measure_start = None
    measure_end = None
    inference_returncode = None

    with raw_path.open(
        "w", newline="", encoding="utf-8"
    ) as raw_file:
        raw_file.write(
            "timestamp,gpu_util_percent,memory_used_mib\n"
        )
        raw_file.flush()

        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "-i", "0",
                "--query-gpu=timestamp,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-lms", "100",
            ],
            stdout=raw_file,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        try:
            with log_path.open(
                "w", encoding="utf-8"
            ) as log_file:
                process = subprocess.Popen(
                    [
                        str(EXE),
                        args.workload,
                        args.mode,
                        str(args.batch_size),
                        str(request_path),
                    ],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                for line in process.stdout:
                    received_at = datetime.now()
                    log_file.write(line)
                    print(line, end="", flush=True)

                    marker = line.strip()

                    if marker == "MEASURE_START":
                        measure_start = received_at
                    elif marker == "MEASURE_END":
                        measure_end = received_at

                inference_returncode = process.wait()
        finally:
            if monitor.poll() is None:
                monitor.terminate()

            _, monitor_error = monitor.communicate()

    if monitor_error:
        print("GPU monitor message:", monitor_error)

    if inference_returncode != 0:
        raise RuntimeError(
            f"Inference failed. Check {log_path}"
        )

    if measure_start is None or measure_end is None:
        raise RuntimeError(
            "Measurement markers missing. Check main.cpp and rebuild."
        )

    measured_utilization = []

    with raw_path.open(
        newline="", encoding="utf-8"
    ) as raw_file, samples_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        reader = csv.DictReader(raw_file)

        writer = csv.writer(output_file)
        writer.writerow([
            "timestamp",
            "phase",
            "gpu_util_percent",
            "memory_used_mib",
        ])

        for row in reader:
            timestamp = row["timestamp"].strip()

            sample_time = datetime.strptime(
                timestamp,
                "%Y/%m/%d %H:%M:%S.%f",
            )

            if sample_time < measure_start:
                phase = "initialization_or_warmup"
            elif sample_time <= measure_end:
                phase = "measurement"
            else:
                phase = "shutdown"

            utilization = row["gpu_util_percent"].strip()
            memory = row["memory_used_mib"].strip()

            writer.writerow([
                timestamp,
                phase,
                utilization,
                memory,
            ])

            if phase == "measurement":
                try:
                    measured_utilization.append(float(utilization))
                except ValueError:
                    pass

    metadata = {
        "workload": args.workload,
        "mode": args.mode,
        "configured_batch_size": args.batch_size,
        "max_wait_ms": 10 if args.mode == "adaptive" else 0,
        "repeat": args.run,
        "gpu_index": 0,
        "query_interval_ms": 100,
        "measurement_start_host": measure_start.isoformat(),
        "measurement_end_host": measure_end.isoformat(),
        "valid_measurement_samples": len(measured_utilization),
        "gpu_measurement_enabled": True,
        "scope": "whole_gpu",
    }

    with (output_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    if not measured_utilization:
        raise RuntimeError(
            f"No valid GPU samples. Check {raw_path}"
        )

    print(
        "Mean sampled GPU utilization: "
        f"{statistics.mean(measured_utilization):.2f}%"
    )
    print(
        "Max sampled GPU utilization: "
        f"{max(measured_utilization):.2f}%"
    )
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()