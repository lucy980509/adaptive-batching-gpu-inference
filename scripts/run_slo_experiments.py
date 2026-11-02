"""Run policy-comparison experiments with p3_inference and validate every result.

Request rates and the default SLO numbers live in the C++ code (workload.hpp,
slo_config.hpp). This script only passes overrides that you give explicitly, then reads
the settings that were actually applied from each request CSV (they are recorded in every
row) and checks them: constant inside a run, equal to any override you passed, and the
same in all runs of the experiment.

Per run it also checks that every generated request completed exactly once (ids 1..N),
that generated + not-generated equals the scheduled count (recomputed here from the
recorded schedule), and that the image order is (id - 1) % pool size. After each run it
compares image_index and the planned arrival offset with the first run of the same
workload: the same request id must have the same image and the same planned arrival in
every policy. Actual arrivals are NOT assumed to be identical; generation lag and arrival
gaps are summarized instead.

Formal comparison (run from the repository root, after building and committing):
    python scripts\\run_slo_experiments.py --max-batch 16 --repeats 3
Dry run (prints the job list only):
    python scripts\\run_slo_experiments.py --max-batch 16 --repeats 3 --dry-run

Aggregation: every metric, including P50/P95/P99, is computed per run first; comparison.csv
then reports the mean and sample standard deviation of those per-run values. Percentiles
are never pooled over requests of different runs. GPU monitoring is not part of this run;
use scripts\\measure_gpu.py separately.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'build' / 'Release' / 'p3_inference.exe'
MANIFEST = ROOT / 'data' / 'imagenet' / 'val1000_manifest.json'
DEFAULT_OUTPUT = ROOT / 'results' / 'resnet50' / 'comparison'
WORKLOADS = ['low', 'medium', 'high', 'bursty']  # names only; rates are in workload.hpp
STATIC_SIZES = [1, 4, 8, 16, 32]
MODES = ('static', 'adaptive', 'adaptive_immediate', 'slo')
SETTING_COLUMNS = ['target_rps', 'duration_seconds', 'burst_on_ms', 'burst_off_ms',
                   'slo_target_ms', 'slo_required_percent', 'safety_margin_ms', 'model']
METRICS = ['p50_ms', 'p95_ms', 'p99_ms', 'mean_queue_wait_ms', 'mean_service_time_ms',
           'throughput_rps', 'slo_attainment_percent', 'mean_actual_batch_size',
           'mean_generation_lag_ms', 'p95_generation_lag_ms', 'max_generation_lag_ms',
           'arrival_gap_p95_ms', 'share_arrival_gaps_below_1ms', 'drain_ms',
           'mean_service_minus_estimate_ms', 'max_service_minus_estimate_ms']
# (summary column, regex on the log); drain_ms is max(0, signed offset)
LOG_FIELDS = [('scheduled_requests', r'Scheduled requests: (\d+)'),
              ('generated_requests', r'Generated requests: (\d+)'),
              ('not_generated_requests', r'Not generated at the window boundary: (\d+)'),
              ('prepare_mean_us', r'preparation .*?mean ([\d.]+) us'),
              ('prepare_max_us', r'max ([\d.]+) us'),
              ('drain_ms', r'Drain after generation end: ([\d.]+) ms'),
              ('last_completion_minus_generation_end_ms',
               r'Last completion minus generation end \(signed\): (-?[\d.]+) ms'),
              ('peak_working_set_mib', r'Peak working set: ([\d.]+) MiB')]
FIELDS = (['workload', 'mode', 'configured_max_batch', 'repeat', 'completed_requests',
           'slo_met_requests', 'slo_violations', 'slo_pass', 'observed_interval_seconds',
           'actual_arrival_rate_rps', 'expected_scheduled_requests',
           'requests_negative_slack_at_start', 'share_batches_service_above_estimate']
          + METRICS + SETTING_COLUMNS[:-1] + ['max_wait_ms', 'request_csv', 'calibration_csv']
          + [n for n, _ in LOG_FIELDS if n != 'drain_ms'])


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


def pool_size():
    return int(json.loads(MANIFEST.read_text(encoding='utf-8'))['selection']['count'])


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(ROOT),
                                   universal_newlines=True).strip()


def expected_scheduled(interval_ns, duration_s, on_ms, off_ms):
    """Arrival slots planned by the workload (slot times t < window end)."""
    def slots(length_ns):
        return -(-length_ns // interval_ns)
    total = int(duration_s) * 10 ** 9
    if on_ms > 0:
        on, cycle = on_ms * 10 ** 6, (on_ms + off_ms) * 10 ** 6
        count, start = 0, 0
        while start < total:
            count += slots(min(on, total - start))
            start += cycle
        return count
    return slots(total)


def read_settings(rows, expected_mode, expected_batch, expected_workload):
    """The settings recorded in the CSV; they must be constant inside the run."""
    keys = SETTING_COLUMNS + ['max_wait_ms']
    settings = {c: rows[0][c] for c in keys}
    for row in rows:
        if any(row[c] != settings[c] for c in keys):
            raise ValueError('Settings change inside one run')
        if row['workload'] != expected_workload or row['mode'] != expected_mode:
            raise ValueError('Unexpected workload or mode')
        if int(row['max_batch_size']) != expected_batch:
            raise ValueError('Unexpected configured max batch size')
    return settings


def summarize(path, workload, mode, max_batch, images):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('No requests: ' + str(path))
    settings = read_settings(rows, mode, max_batch, workload)
    target_ms = number(rows[0], 'slo_target_ms')
    required = number(rows[0], 'slo_required_percent')
    ids = []
    for row in rows:
        request_id = int(row['request_id'])
        ids.append(request_id)
        if int(row['image_index']) != (request_id - 1) % images:
            raise ValueError('Image order broken at request {}'.format(request_id))
        if row['slo_met'] not in ('0', '1'):
            raise ValueError('Invalid SLO flag')
        if not 1 <= int(row['actual_batch_size']) <= max_batch:
            raise ValueError('Invalid actual batch size')
        latency = number(row, 'e2e_latency_ms')
        if latency < 0 or number(row, 'queue_wait_ms') < 0 or number(row, 'service_time_ms') < 0:
            raise ValueError('Negative request duration')
        if abs(number(row, 'queue_wait_ms') + number(row, 'service_time_ms') - latency) > .001:
            raise ValueError('Request timing does not reconcile')
        if abs(number(row, 'service_time_ms') - number(row, 'estimated_service_ms')
               - number(row, 'service_minus_estimate_ms')) > .001:
            raise ValueError('service_minus_estimate_ms does not match its parts')
        if abs(latency - target_ms) > 1e-6 and int(row['slo_met']) != int(latency <= target_ms):
            raise ValueError('SLO flag does not agree with latency')
    # Each generated request must complete exactly once: ids 1..N, no gaps, no repeats.
    if sorted(ids) != list(range(1, len(rows) + 1)):
        raise ValueError('Requests do not complete exactly once (ids are not 1..N)')
    by_id = sorted(rows, key=lambda r: int(r['request_id']))
    arrivals = [number(r, 'arrival_offset_ms') for r in by_id]
    scheduled = [number(r, 'scheduled_offset_ms') for r in by_id]
    if any(b < a for a, b in zip(scheduled, scheduled[1:])):
        raise ValueError('Planned arrivals are not ordered')
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    latency = [number(row, 'e2e_latency_ms') for row in rows]
    start = min(arrivals)
    end = max(number(row, 'arrival_offset_ms') + number(row, 'e2e_latency_ms') for row in rows)
    elapsed = (end - start) / 1000.0
    if elapsed <= 0:
        raise ValueError('Invalid observation interval')
    batch_count = sum(1.0 / int(row['actual_batch_size']) for row in rows)
    if abs(batch_count - round(batch_count)) > .00001:
        raise ValueError('Partial batch records in request CSV')
    met = sum(int(row['slo_met']) for row in rows)
    attainment = 100.0 * met / len(rows)
    lags = [number(r, 'generation_lag_ms') for r in rows]
    differences = [number(r, 'service_minus_estimate_ms') for r in rows]

    # planned schedule, rebuilt from the recorded spacing of requests 1 and 2
    expected = ''
    if len(scheduled) > 1:
        expected = expected_scheduled(
            int(round(scheduled[1] * 1e6)), float(settings['duration_seconds']),
            int(float(settings['burst_on_ms'])), int(float(settings['burst_off_ms'])))
    result = {
        'completed_requests': len(rows), 'slo_met_requests': met,
        'slo_violations': len(rows) - met, 'slo_pass': int(attainment >= required),
        'observed_interval_seconds': elapsed,
        'actual_arrival_rate_rps': (len(rows) - 1) / (max(arrivals) / 1000.0),
        'expected_scheduled_requests': expected,
        'requests_negative_slack_at_start': sum(
            1 for r in rows if number(r, 'slack_at_service_start_ms') < 0),
        'share_batches_service_above_estimate': (
            sum(1 for d in differences if d > 0) / len(differences)),
        'p50_ms': percentile(latency, 50), 'p95_ms': percentile(latency, 95),
        'p99_ms': percentile(latency, 99),
        'mean_queue_wait_ms': statistics.mean(number(r, 'queue_wait_ms') for r in rows),
        'mean_service_time_ms': statistics.mean(number(r, 'service_time_ms') for r in rows),
        'throughput_rps': len(rows) / elapsed,
        'slo_attainment_percent': attainment,
        'mean_actual_batch_size': len(rows) / batch_count,
        'mean_generation_lag_ms': statistics.mean(lags),
        'p95_generation_lag_ms': percentile(lags, 95),
        'max_generation_lag_ms': max(lags),
        'arrival_gap_p95_ms': percentile(gaps, 95) if gaps else '',
        'share_arrival_gaps_below_1ms': (
            sum(1 for g in gaps if g < 1.0) / len(gaps) if gaps else ''),
        'mean_service_minus_estimate_ms': statistics.mean(differences),
        'max_service_minus_estimate_ms': max(differences),
    }
    result.update({c: settings[c] for c in SETTING_COLUMNS[:-1]})
    result['max_wait_ms'] = settings['max_wait_ms']
    plan = {'image_index': [int(r['image_index']) for r in by_id], 'scheduled_ms': scheduled}
    return result, settings, plan


def validate_calibration(path, max_batch, margin_ms):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if [int(r['batch_size']) for r in rows] != list(range(1, max_batch + 1)):
        raise ValueError('Calibration must cover batch sizes 1..{}: {}'.format(max_batch, path))
    estimates = [number(r, 'estimated_service_ms') for r in rows]
    if any(x <= 0 for x in estimates) or estimates != sorted(estimates):
        raise ValueError('Invalid service-time estimate envelope')
    if any(number(r, 'safety_margin_ms') != margin_ms for r in rows):
        raise ValueError('Calibration margin differs from the request CSV')
    return {r['calibration_runs'] + '|' + r['estimator'] for r in rows}


def parse_log(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    found = {}
    for name, pattern in LOG_FIELDS:
        match = re.search(pattern, text)
        found[name] = float(match.group(1)) if match else ''
    return found


def check_counts(result):
    """generated = completed, and generated + not generated = scheduled = recomputed."""
    generated = int(result['generated_requests'])
    if generated != result['completed_requests']:
        raise ValueError('Generated {} requests but {} completed'.format(
            generated, result['completed_requests']))
    if generated + int(result['not_generated_requests']) != int(result['scheduled_requests']):
        raise ValueError('generated + not generated != scheduled')
    if int(result['scheduled_requests']) != int(result['expected_scheduled_requests']):
        raise ValueError('Scheduled count {} differs from the recomputed {}'.format(
            result['scheduled_requests'], result['expected_scheduled_requests']))


def check_same_plan(reference, plan, name):
    """Same request id => same image and same planned arrival, across policies."""
    n = min(len(reference['image_index']), len(plan['image_index']))
    if reference['image_index'][:n] != plan['image_index'][:n]:
        raise ValueError('Image order differs from the first run of this workload: ' + name)
    if any(abs(a - b) > 1e-6 for a, b in zip(reference['scheduled_ms'][:n], plan['scheduled_ms'][:n])):
        raise ValueError('Planned arrival offsets differ from the first run of this workload: ' + name)


def write_comparison(rows, output):
    groups = sorted({(r['workload'], r['mode'], r['configured_max_batch']) for r in rows})
    fields = ['workload', 'mode', 'configured_max_batch', 'repeats', 'runs_meeting_slo',
              'slo_target_ms', 'safety_margin_ms', 'slo_required_percent', 'target_rps',
              'duration_seconds']
    for metric in METRICS:
        fields.extend([metric + '_mean', metric + '_std', metric + '_cv_percent'])
    fields.append('aggregation')
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for workload, mode, batch in groups:
            group = [r for r in rows if (r['workload'], r['mode'], r['configured_max_batch'])
                     == (workload, mode, batch)]
            result = dict(workload=workload, mode=mode, configured_max_batch=batch,
                          repeats=len(group), runs_meeting_slo=sum(r['slo_pass'] for r in group),
                          aggregation='per-run values first (percentiles included), then mean and '
                                      'sample SD over runs; percentiles are not pooled')
            for key in ['slo_target_ms', 'safety_margin_ms', 'slo_required_percent',
                        'target_rps', 'duration_seconds']:
                result[key] = group[0][key]
            for metric in METRICS:
                values = [float(r[metric]) for r in group]
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else None
                result[metric + '_mean'] = mean
                result[metric + '_std'] = std
                result[metric + '_cv_percent'] = 100 * std / mean if std is not None and mean > 0 else None
            writer.writerow(result)


def build_identity(allow_dirty):
    """Commit and executable that produced the results."""
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty and not allow_dirty:
        raise RuntimeError('Tracked files have uncommitted changes; commit first '
                           '(or pass --allow-dirty):\n' + dirty)
    newest = max(p.stat().st_mtime for pattern in ('src/*.cpp', 'include/*.hpp')
                 for p in ROOT.glob(pattern))
    newest = max(newest, (ROOT / 'CMakeLists.txt').stat().st_mtime)
    if EXE.stat().st_mtime < newest:
        raise RuntimeError('Executable is older than the sources; rebuild it first.')
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(dirty),
            'git_dirty_files': dirty.splitlines(),
            'executable': str(EXE.relative_to(ROOT)),
            'executable_sha256': hashlib.sha256(EXE.read_bytes()).hexdigest(),
            'executable_modified': datetime.fromtimestamp(EXE.stat().st_mtime).isoformat(timespec='seconds')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-batch', type=int, required=True, choices=STATIC_SIZES,
                        help='maximum batch size for Adaptive/Immediate/SLO; Static runs every size up to it')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--only', nargs='+', metavar='WORKLOAD:MODE:BATCH',
                        help='run only these jobs (in the given order) instead of the full matrix')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-dirty', action='store_true',
                        help='allow uncommitted tracked changes (recorded in experiment_config.json)')
    for option in ['duration', 'slo-ms', 'margin-ms', 'required-percent', 'max-wait-ms']:
        parser.add_argument('--' + option, type=float, default=None,
                            help='override the C++ default; if omitted the default is used and recorded')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')

    flags = [('--duration', args.duration), ('--slo-ms', args.slo_ms),
             ('--margin-ms', args.margin_ms), ('--required-percent', args.required_percent),
             ('--max-wait-ms', args.max_wait_ms)]
    cli = [x for flag, value in flags if value is not None for x in (flag, str(value))]
    overrides = {'duration_seconds': args.duration, 'slo_target_ms': args.slo_ms,
                 'safety_margin_ms': args.margin_ms, 'slo_required_percent': args.required_percent,
                 'max_wait_ms': args.max_wait_ms}
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if args.only:
        matrix = []
        for item in args.only:
            workload, mode, batch = item.split(':')
            matrix.append((workload, mode, int(batch)))
    else:
        policies = [('static', s) for s in STATIC_SIZES if s <= args.max_batch] + [
            (m, args.max_batch) for m in ('adaptive', 'adaptive_immediate', 'slo')]
        matrix = [(w, m, b) for w in WORKLOADS for m, b in policies]
    for workload, mode, batch in matrix:
        if workload not in WORKLOADS or batch > args.max_batch or mode not in MODES:
            parser.error('Invalid job {}:{}:{}'.format(workload, mode, batch))

    # The job order inside every repeat is shuffled with the seed; the stream continues
    # from one repeat to the next, so repeats get different, reproducible orders.
    rng = random.Random(args.seed)
    plan = []
    for repeat in range(1, args.repeats + 1):
        jobs = list(matrix)
        if not args.only:
            rng.shuffle(jobs)
        plan.extend(dict(workload=w, mode=m, batch_size=b, repeat=repeat) for w, m, b in jobs)
    output = args.output.resolve()

    if args.dry_run:
        print('Planned runs: {} ({} settings x {} repeats)'.format(len(plan), len(matrix), args.repeats))
        print('Output:', output, '| exists:', output.exists())
        print('Overrides passed to p3_inference:', overrides or 'none (C++ defaults, recorded in the CSVs)')
        print('Settings per workload:', {w: sum(1 for j in matrix if j[0] == w) for w in WORKLOADS})
        print('Settings:', sorted({(m, b) for _, m, b in matrix}))
        for repeat in range(1, args.repeats + 1):
            print('Repeat', repeat, 'order:', ' '.join(
                '{}:{}:{}'.format(j['workload'], j['mode'], j['batch_size'])
                for j in plan if j['repeat'] == repeat))
        return 0

    if not EXE.is_file():
        raise FileNotFoundError('Executable not found: ' + str(EXE))
    if output.exists():
        raise FileExistsError('Output already exists. Preserve it and choose another --output: ' + str(output))
    identity = build_identity(args.allow_dirty)
    output.mkdir(parents=True)

    images = pool_size()
    applied, calibration_methods = None, set()
    references = {}
    results = []
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for index, job in enumerate(plan, 1):
            workload, mode, size, repeat = job['workload'], job['mode'], job['batch_size'], job['repeat']
            stem = '{}_{}_b{}_run{}'.format(workload, mode, size, repeat)
            csv_path = output / (stem + '.csv')
            calibration = output / (stem + '.csv.service_times.csv')
            log_path = output / (stem + '.log')
            print('[{}/{}] {}'.format(index, len(plan), stem), flush=True)
            try:
                with log_path.open('w', encoding='utf-8') as log:
                    subprocess.run([str(EXE), workload, mode, str(size), str(csv_path)] + cli,
                                   cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=900)
                summary, settings, run_plan = summarize(csv_path, workload, mode, size, images)
                # max_wait_ms is recorded as what the policy applies: the wait in use for
                # Adaptive wait, 0 for Adaptive immediate, blank for Static and SLO-aware.
                wait = settings['max_wait_ms']
                if (mode == 'adaptive' and wait == '') or \
                        (mode == 'adaptive_immediate' and wait != '0') or \
                        (mode in ('static', 'slo') and wait != ''):
                    raise ValueError('Unexpected max_wait_ms {!r} for {}'.format(wait, mode))
                for key, value in overrides.items():
                    if key == 'max_wait_ms' and mode != 'adaptive':
                        continue
                    if settings[key] != '' and float(settings[key]) != value:
                        raise ValueError('CSV {}={} differs from the override {}'.format(
                            key, settings[key], value))
                common = {k: settings[k] for k in ('slo_target_ms', 'slo_required_percent',
                                                   'safety_margin_ms', 'duration_seconds')}
                if applied is None:
                    applied = common
                elif applied != common:
                    raise ValueError('SLO or duration settings differ between runs')
                calibration_methods |= validate_calibration(
                    calibration, size, float(settings['safety_margin_ms']))
                if len(calibration_methods) != 1:
                    raise ValueError('Calibration method differs between runs: ' + str(calibration_methods))
                result = dict(workload=workload, mode=mode, configured_max_batch=size,
                              repeat=repeat, request_csv=csv_path.name,
                              calibration_csv=calibration.name, **summary)
                result.update(parse_log(log_path))
                check_counts(result)
                if workload not in references:
                    references[workload] = run_plan
                else:
                    check_same_plan(references[workload], run_plan, stem)
            except Exception as error:
                (output / 'failure.json').write_text(json.dumps(dict(job=job, error=str(error)), indent=2), encoding='utf-8')
                raise
            results.append(result)
            writer.writerow(result)
            f.flush()
            print('  scheduled={} generated={} completed={} | P95={:.3f} ms | SLO={:.3f}% (need {}%) | {}'.format(
                int(result['scheduled_requests']), int(result['generated_requests']),
                result['completed_requests'], result['p95_ms'], result['slo_attainment_percent'],
                result['slo_required_percent'], 'PASS' if result['slo_pass'] else 'FAIL'), flush=True)

    manifest = dict(
        created=datetime.now().isoformat(timespec='seconds'), **identity,
        command=' '.join(sys.argv), plan=plan, seed=args.seed, repeats=args.repeats,
        max_batch=args.max_batch, overrides=overrides, applied_common_settings=applied,
        per_workload_settings={w: {k: r[k] for k in ('target_rps', 'duration_seconds',
                                                      'burst_on_ms', 'burst_off_ms')}
                               for w in {r['workload'] for r in results}
                               for r in [next(x for x in results if x['workload'] == w)]},
        calibration='warm-up 3 + 10 timed runs per batch size, estimate = prefix maximum of '
                    'per-size maxima ({}); each run calibrates sizes 1..its own max batch with '
                    'the same method for every policy'.format(sorted(calibration_methods)[0]),
        python_version=sys.version, image_pool_size=images,
        image_order='request id k carries image (k-1) % pool size; checked in every row and against '
                    'the first run of each workload (together with the planned arrival offsets)',
        gpu_measurement_enabled=False,
        definitions=dict(
            throughput='completed / (last finish - first actual arrival)',
            drain_ms='max(0, last completion - end of the generation window)',
            last_completion_minus_generation_end_ms='signed; negative = the last request finished before the window ended',
            slack_at_service_start_ms='deadline - start of service - estimated service time; '
                                      'negative means the request was already past that point when service began, '
                                      'which includes queue wait; it is not by itself an estimation error',
            service_minus_estimate_ms='measured service time of the batch minus its calibration estimate',
            generation_lag_ms='actual arrival - planned arrival; arrivals of one run are summarized '
                              '(lag, arrival gaps), not assumed identical across runs',
            aggregation='per-run percentiles, then mean/SD over runs; not pooled'))
    (output / 'experiment_config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    write_comparison(results, output)
    print('\nCompleted: {}/{}'.format(len(results), len(plan)))
    print('Summary:', output / 'summary.csv')
    print('Comparison:', output / 'comparison.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
