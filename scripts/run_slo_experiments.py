import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ['low', 'medium', 'high', 'bursty']
POLICIES = [('static', size) for size in (1, 4, 8, 16, 32)] + [
    ('adaptive', 32), ('adaptive_immediate', 32), ('slo', 32)]
METRICS = ['p50_ms', 'p95_ms', 'p99_ms', 'mean_queue_wait_ms',
           'mean_service_time_ms', 'throughput_rps', 'slo_attainment_percent',
           'mean_actual_batch_size', 'mean_generation_lag_ms']
FIELDS = ['workload', 'mode', 'configured_batch_size', 'max_wait_ms',
          'slo_target_ms', 'safety_margin_ms', 'slo_required_percent',
          'repeat', 'completed_requests', 'slo_met_requests', 'slo_violations',
          'slo_pass', 'observed_interval_seconds'] + METRICS + [
          'gpu_measurement_enabled', 'request_csv', 'calibration_csv']


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError('Non-finite value: ' + key)
    return value


def summarize(path, workload, mode, batch_size):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('No requests: ' + str(path))
    ids = set()
    for row in rows:
        if row['workload'] != workload or row['mode'] != mode:
            raise ValueError('Unexpected workload or mode in ' + str(path))
        if int(row['max_batch_size']) != batch_size:
            raise ValueError('Unexpected configured batch size')
        if (number(row, 'slo_target_ms') != 20 or
                number(row, 'safety_margin_ms') != 10 or
                number(row, 'slo_required_percent') != 95):
            raise ValueError('Expected target 20 ms, margin 10 ms, attainment 95%. Rebuild first.')
        if row['request_id'] in ids:
            raise ValueError('Duplicate request ID')
        ids.add(row['request_id'])
        if row['slo_met'] not in ('0', '1'):
            raise ValueError('Invalid SLO flag')
        size = int(row['actual_batch_size'])
        if not 1 <= size <= batch_size:
            raise ValueError('Invalid actual batch size')
        latency = number(row, 'e2e_latency_ms')
        if latency < 0 or number(row, 'queue_wait_ms') < 0 or number(row, 'service_time_ms') < 0:
            raise ValueError('Negative request duration')
        if abs(number(row, 'queue_wait_ms') + number(row, 'service_time_ms') - latency) > .001:
            raise ValueError('Request timing does not reconcile')
        if abs(latency - 20) > .000001 and int(row['slo_met']) != int(latency <= 20):
            raise ValueError('SLO flag does not agree with latency')
    latency = [number(row, 'e2e_latency_ms') for row in rows]
    start = min(number(row, 'arrival_offset_ms') for row in rows)
    end = max(number(row, 'arrival_offset_ms') + number(row, 'e2e_latency_ms') for row in rows)
    elapsed = (end - start) / 1000.0
    if elapsed <= 0:
        raise ValueError('Invalid observation interval')
    batch_count = sum(1.0 / int(row['actual_batch_size']) for row in rows)
    if abs(batch_count - round(batch_count)) > .00001:
        raise ValueError('Partial batch records in request CSV')
    met = sum(int(row['slo_met']) for row in rows)
    attainment = 100.0 * met / len(rows)
    return {
        'completed_requests': len(rows), 'slo_met_requests': met,
        'slo_violations': len(rows) - met, 'slo_pass': int(attainment >= 95),
        'observed_interval_seconds': elapsed,
        'p50_ms': percentile(latency, 50), 'p95_ms': percentile(latency, 95),
        'p99_ms': percentile(latency, 99),
        'mean_queue_wait_ms': statistics.mean(number(r, 'queue_wait_ms') for r in rows),
        'mean_service_time_ms': statistics.mean(number(r, 'service_time_ms') for r in rows),
        'throughput_rps': len(rows) / elapsed,
        'slo_attainment_percent': attainment,
        'mean_actual_batch_size': len(rows) / batch_count,
        'mean_generation_lag_ms': statistics.mean(number(r, 'generation_lag_ms') for r in rows),
    }


def validate_calibration(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if [int(r['batch_size']) for r in rows] != list(range(1, 33)):
        raise ValueError('Missing batch-size calibration: ' + str(path))
    estimates = [number(r, 'estimated_service_ms') for r in rows]
    if any(x <= 0 for x in estimates) or estimates != sorted(estimates):
        raise ValueError('Invalid service-time estimate envelope')
    if any(int(r['calibration_runs']) != 10 or number(r, 'safety_margin_ms') != 10 for r in rows):
        raise ValueError('Calibration configuration mismatch')


def write_comparison(rows, output, repeats):
    fields = ['workload', 'mode', 'configured_batch_size', 'repeats',
              'slo_target_ms', 'safety_margin_ms', 'slo_required_percent',
              'runs_meeting_slo', 'gpu_measurement_enabled']
    for metric in METRICS:
        fields.extend([metric + '_mean', metric + '_std', metric + '_cv_percent'])
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for workload in WORKLOADS:
            for mode, batch_size in POLICIES:
                group = [r for r in rows if r['workload'] == workload and
                         r['mode'] == mode and r['configured_batch_size'] == batch_size]
                if sorted(r['repeat'] for r in group) != list(range(1, repeats + 1)):
                    raise ValueError('Missing or duplicate repeat')
                result = dict(workload=workload, mode=mode, configured_batch_size=batch_size,
                              repeats=repeats, slo_target_ms=20, safety_margin_ms=10,
                              slo_required_percent=95,
                              runs_meeting_slo=sum(r['slo_pass'] for r in group),
                              gpu_measurement_enabled=False)
                for metric in METRICS:
                    values = [r[metric] for r in group]
                    mean = statistics.mean(values)
                    std = statistics.stdev(values) if len(values) > 1 else None
                    result[metric + '_mean'] = mean
                    result[metric + '_std'] = std
                    result[metric + '_cv_percent'] = 100 * std / mean if std is not None and mean > 0 else None
                writer.writerow(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'slo_comparison')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    executable = ROOT / 'build' / 'Release' / 'p3_inference.exe'
    output = args.output.resolve()
    rng = random.Random(args.seed)
    plan = []
    for repeat in range(1, args.repeats + 1):
        jobs = [(w, m, b) for w in WORKLOADS for m, b in POLICIES]
        rng.shuffle(jobs)
        for workload, mode, batch_size in jobs:
            plan.append(dict(workload=workload, mode=mode, batch_size=batch_size, repeat=repeat))
    if args.dry_run:
        print('Planned runs:', len(plan))
        print('Output:', output)
        print('Target: 20 ms | Margin: 10 ms | Required attainment: 95%')
        print('GPU monitoring: disabled (separate experiments)')
        return
    if not executable.is_file():
        raise FileNotFoundError('Executable not found: ' + str(executable))
    if output.exists():
        raise FileExistsError('Output already exists. Preserve it and choose another --output: ' + str(output))
    output.mkdir(parents=True)
    manifest = dict(plan=plan, seed=args.seed, repeats=args.repeats,
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    python_version=sys.version, target_ms=20, safety_margin_ms=10,
                    required_attainment_percent=95, gpu_measurement_enabled=False,
                    throughput_definition='completed / (last finish - first actual arrival)',
                    aggregation='mean and sample SD of run-level metrics; no pooled percentiles')
    (output / 'experiment_config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    results = []
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for index, job in enumerate(plan, 1):
            workload, mode, size, repeat = job['workload'], job['mode'], job['batch_size'], job['repeat']
            stem = '{}_{}_b{}_run{}'.format(workload, mode, size, repeat)
            csv_path = output / (stem + '.csv')
            calibration = output / (stem + '.csv.service_times.csv')
            print('[{}/{}] {}'.format(index, len(plan), stem), flush=True)
            try:
                with (output / (stem + '.log')).open('w', encoding='utf-8') as log:
                    subprocess.run([str(executable), workload, mode, str(size), str(csv_path)],
                                   cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=180)
                validate_calibration(calibration)
                result = dict(workload=workload, mode=mode, configured_batch_size=size,
                              max_wait_ms=10 if mode == 'adaptive' else (0 if mode == 'adaptive_immediate' else ''),
                              slo_target_ms=20, safety_margin_ms=10, slo_required_percent=95,
                              repeat=repeat, gpu_measurement_enabled=False,
                              request_csv=csv_path.name, calibration_csv=calibration.name,
                              **summarize(csv_path, workload, mode, size))
            except Exception as error:
                (output / 'failure.json').write_text(json.dumps(dict(job=job, error=str(error)), indent=2), encoding='utf-8')
                raise
            results.append(result)
            writer.writerow(result)
            f.flush()
            print('  Requests={} | P95={:.3f} ms | SLO={:.3f}% | {}'.format(
                result['completed_requests'], result['p95_ms'], result['slo_attainment_percent'],
                'PASS' if result['slo_pass'] else 'FAIL'), flush=True)
    write_comparison(results, output, args.repeats)
    print('\nCompleted: {}/{}'.format(len(results), len(plan)))
    print('Summary:', output / 'summary.csv')
    print('Comparison:', output / 'comparison.csv')


if __name__ == '__main__':
    main()
