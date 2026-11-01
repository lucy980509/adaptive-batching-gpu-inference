import csv
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / "metrics.csv"

with csv_path.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print(f"Completed requests: {len(rows)}")

for row in rows:
    print(
        f"Request {row['request_id']} | "
        f"Wait: {row['queue_wait_ms']} ms | "
        f"Service: {row['service_time_ms']} ms | "
        f"Total: {row['e2e_latency_ms']} ms"
    )