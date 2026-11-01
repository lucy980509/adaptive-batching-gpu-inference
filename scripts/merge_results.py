import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCES = [
    RESULTS / "comparison" / "summary.csv",
    RESULTS / "immediate" / "summary.csv",
]

OUTPUT = RESULTS / "combined"


def main():
    all_rows = []
    fields = None

    for source in SOURCES:
        with source.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise ValueError(f"CSV columns differ: {source}")

            for row in reader:
                request_path = source.parent / row["request_csv"]

                if not request_path.is_file():
                    raise FileNotFoundError(request_path)

                row["request_csv"] = request_path.relative_to(
                    RESULTS
                ).as_posix()

                all_rows.append(row)

    if len(all_rows) != 84:
        raise ValueError(
            f"Expected 84 runs, found {len(all_rows)}"
        )

    groups = defaultdict(list)
    seen = set()

    for row in all_rows:
        key = (
            row["workload"],
            row["mode"],
            int(row["configured_batch_size"]),
        )

        run_key = (*key, int(row["repeat"]))

        if run_key in seen:
            raise ValueError(f"Duplicate run: {run_key}")

        seen.add(run_key)
        groups[key].append(row)

    if len(groups) != 28:
        raise ValueError(
            f"Expected 28 configurations, found {len(groups)}"
        )

    for key, runs in groups.items():
        repeats = {int(row["repeat"]) for row in runs}

        if repeats != {1, 2, 3}:
            raise ValueError(f"Missing repeats: {key}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT / "summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

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

    comparison_path = OUTPUT / "comparison.csv"

    with comparison_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_fields,
        )
        writer.writeheader()

        for key, runs in sorted(groups.items()):
            workload, mode, batch_size = key

            result = {
                "workload": workload,
                "mode": mode,
                "configured_batch_size": batch_size,
                "repeats": len(runs),
            }

            for metric in metric_names:
                values = [
                    float(row[metric])
                    for row in runs
                ]

                mean = statistics.mean(values)
                std = statistics.stdev(values)

                result[f"{metric}_mean"] = mean
                result[f"{metric}_std"] = std
                result[f"{metric}_cv_percent"] = (
                    std / mean * 100 if mean != 0 else ""
                )

            writer.writerow(result)

    print(f"Validated runs: {len(all_rows)}")
    print(f"Configurations: {len(groups)}")
    print(f"Summary: {summary_path}")
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    main()