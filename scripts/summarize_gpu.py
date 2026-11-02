"""Check and summarize the GPU-monitored runs in results/resnet50/gpu/.

Run from the repository root (any Python 3.6+, standard library only):
    python scripts\\summarize_gpu.py

For every setting there must be one run per repeat. Each run folder is checked for
consistency between metadata.json, gpu_samples.csv, requests.csv and gpu_raw.csv; folders
whose name ends in ".incomplete" are ignored. A run contributes ONE number: the mean
sampled GPU utilization over its measurement samples (between the host-received
MEASURE_START and MEASURE_END markers; warm-up and calibration samples are excluded).
Per setting the table reports the mean and sample standard deviation of those run means.
The roughly 100 ms samples inside a run are not independent repeats and are never used as
the sample size.

These are separate monitored runs, not the GPU measurements of the 84 performance runs in
results/resnet50/comparison. Utilization is whole-GPU nvidia-smi utilization (a rolling
device metric), so it says how busy the GPU was, not which policy is more efficient.

Output: results/resnet50/gpu/summary.csv (refuses to overwrite).
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / 'results' / 'resnet50' / 'gpu'
COMPARISON_CONFIG = ROOT / 'results' / 'resnet50' / 'comparison' / 'experiment_config.json'
EXE = ROOT / 'build' / 'Release' / 'p3_inference.exe'
WORKLOAD = 'high'
SETTINGS = [('static', 1), ('static', 4), ('static', 8), ('static', 16),
            ('adaptive', 16), ('adaptive_immediate', 16), ('slo', 16)]
REPEATS = [1, 2, 3]
MIN_SAMPLES = 30


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_run(folder, expected, problems):
    """Return the per-run record, appending anything inconsistent to problems."""
    name = folder.name
    need = ['metadata.json', 'gpu_samples.csv', 'gpu_raw.csv', 'requests.csv', 'inference.log']
    missing = [n for n in need if not (folder / n).is_file()]
    if missing:
        problems.append('{}: missing {}'.format(name, missing))
        return None
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))

    # GPU samples: measurement phase only
    with (folder / 'gpu_samples.csv').open(newline='', encoding='utf-8') as f:
        samples = list(csv.DictReader(f))
    phases = {}
    values = []
    for row in samples:
        phases[row['phase']] = phases.get(row['phase'], 0) + 1
        if row['phase'] == 'measurement':
            value = float(row['gpu_util_percent'])
            if not 0 <= value <= 100:
                problems.append('{}: utilization out of range'.format(name))
            values.append(value)
    if not values:
        problems.append('{}: no valid measurement samples'.format(name))
        return None
    if len(values) < MIN_SAMPLES:
        problems.append('{}: only {} measurement samples'.format(name, len(values)))
    if not phases.get('initialization_or_warmup'):
        problems.append('{}: no warm-up/calibration samples to exclude (unexpected)'.format(name))
    mean = statistics.mean(values)
    if meta['valid_measurement_samples'] != len(values):
        problems.append('{}: metadata sample count differs'.format(name))
    if abs(meta['mean_sampled_gpu_util_percent'] - mean) > 1e-9 or \
            meta['max_sampled_gpu_util_percent'] != max(values):
        problems.append('{}: metadata mean/max differ from gpu_samples.csv'.format(name))
    with (folder / 'gpu_raw.csv').open(encoding='utf-8') as f:
        raw_rows = sum(1 for _ in f) - 1
    if raw_rows != len(samples):
        problems.append('{}: gpu_raw.csv has {} rows, gpu_samples.csv {}'.format(
            name, raw_rows, len(samples)))

    # requests: generated = completed, each once
    with (folder / 'requests.csv').open(newline='', encoding='utf-8-sig') as f:
        requests = list(csv.DictReader(f))
    ids = sorted(int(r['request_id']) for r in requests)
    if ids != list(range(1, len(requests) + 1)):
        problems.append('{}: requests do not complete exactly once'.format(name))
    for key in ('completed_requests', 'generated_requests', 'scheduled_requests'):
        if meta[key] != len(requests):
            problems.append('{}: {}={} but requests.csv has {}'.format(
                name, key, meta[key], len(requests)))
    met = sum(int(r['slo_met']) for r in requests)
    attainment = 100.0 * met / len(requests)
    if abs(attainment - meta['slo_attainment_percent']) > 1e-9:
        problems.append('{}: SLO attainment differs from requests.csv'.format(name))

    # settings identical to the formal comparison
    for key, value in expected.items():
        if float(meta[key]) != float(value):
            problems.append('{}: {}={} but the comparison used {}'.format(name, key, meta[key], value))
    if not meta.get('separate_from_comparison_runs'):
        problems.append('{}: not marked as separate from the comparison runs'.format(name))
    if meta['workload'] != WORKLOAD:
        problems.append('{}: workload {}'.format(name, meta['workload']))
    markers = folder / 'markers.json'
    if markers.is_file():
        m = json.loads(markers.read_text(encoding='utf-8'))
        if m['measurement_start_host'] != meta['measurement_start_host'] or \
                m['measurement_end_host'] != meta['measurement_end_host']:
            problems.append('{}: markers.json differs from metadata.json'.format(name))
    return {'mean': mean, 'samples': len(values), 'max': max(values),
            'attainment': attainment, 'seconds': meta['measurement_seconds'],
            'commit': meta['git_commit'], 'dirty': meta['git_dirty'],
            'exe_sha256': meta['executable_sha256'],
            'has_markers_file': markers.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=GPU_DIR / 'summary.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output exists, not overwriting: ' + str(args.output))

    config = json.loads(COMPARISON_CONFIG.read_text(encoding='utf-8'))
    common = config['applied_common_settings']
    high = config['per_workload_settings'][WORKLOAD]
    expected = {'duration_seconds': common['duration_seconds'], 'slo_target_ms': common['slo_target_ms'],
                'safety_margin_ms': common['safety_margin_ms'],
                'slo_required_percent': common['slo_required_percent'],
                'target_rps': high['target_rps']}

    folders = sorted(p for p in GPU_DIR.iterdir() if p.is_dir())
    ignored = [p.name for p in folders if p.name.endswith('.incomplete')]
    folders = [p for p in folders if not p.name.endswith('.incomplete')]
    wanted = {'{}_{}_b{}_run{}'.format(WORKLOAD, m, b, r): (m, b, r)
              for m, b in SETTINGS for r in REPEATS}
    problems = []
    unexpected = [p.name for p in folders if p.name not in wanted]
    if unexpected:
        problems.append('Unexpected folders: {}'.format(unexpected))
    runs = {}
    for name, key in wanted.items():
        folder = GPU_DIR / name
        if not folder.is_dir():
            problems.append('{}: folder missing'.format(name))
            continue
        record = check_run(folder, expected, problems)
        if record:
            runs[key] = record

    # provenance: every run must use the executable of the formal comparison
    comparison_sha = config['executable_sha256']
    current_sha = sha256(EXE)
    shas = {r['exe_sha256'] for r in runs.values()}
    commits = {}
    for r in runs.values():
        commits[r['commit'][:7]] = commits.get(r['commit'][:7], 0) + 1
    if shas != {comparison_sha}:
        problems.append('Executable SHA-256 differs: runs {} vs comparison {}'.format(
            sorted(s[:12] for s in shas), comparison_sha[:12]))
    if current_sha != comparison_sha:
        problems.append('Current build/Release exe differs from the comparison exe')
    if any(r['dirty'] for r in runs.values()):
        problems.append('Some runs were recorded with uncommitted tracked changes')

    print('Runs found: {}/{} | ignored: {}'.format(len(runs), len(wanted), ignored))
    print('Executable SHA-256 of all runs == comparison == current build: {}  ({})'.format(
        shas == {comparison_sha} and current_sha == comparison_sha, comparison_sha[:12]))
    print('Script commits recorded in the runs:', commits)
    if problems:
        print('\nPROBLEMS ({}):'.format(len(problems)))
        for p in problems:
            print(' -', p)
        return 1

    fields = ['workload', 'mode', 'configured_max_batch', 'runs', 'mean_of_run_means_percent',
              'sd_of_run_means_percent', 'min_run_mean_percent', 'max_run_mean_percent',
              'observed_max_percent_over_runs']
    for r in REPEATS:
        fields += ['run{}_mean_percent'.format(r), 'run{}_valid_samples'.format(r),
                   'run{}_observed_max_percent'.format(r), 'run{}_slo_attainment_percent'.format(r)]
    fields += ['mean_measurement_seconds', 'executable_sha256', 'scope']
    scope = ('Separate GPU-monitored runs (workload high), NOT measured during the 84 performance '
             'runs of results/resnet50/comparison. Whole-GPU nvidia-smi utilization sampled about '
             'every 100 ms; each run counts once (mean of its measurement samples); mean and '
             'sample SD are over the 3 run means; samples inside a run are not independent repeats.')
    table = []
    for mode, batch in SETTINGS:
        group = [runs[(mode, batch, r)] for r in REPEATS]
        means = [g['mean'] for g in group]
        row = {'workload': WORKLOAD, 'mode': mode, 'configured_max_batch': batch,
               'runs': len(group), 'mean_of_run_means_percent': statistics.mean(means),
               'sd_of_run_means_percent': statistics.stdev(means),
               'min_run_mean_percent': min(means), 'max_run_mean_percent': max(means),
               'observed_max_percent_over_runs': max(g['max'] for g in group),
               'mean_measurement_seconds': statistics.mean(g['seconds'] for g in group),
               'executable_sha256': comparison_sha, 'scope': scope}
        for r, g in zip(REPEATS, group):
            row['run{}_mean_percent'.format(r)] = g['mean']
            row['run{}_valid_samples'.format(r)] = g['samples']
            row['run{}_observed_max_percent'.format(r)] = g['max']
            row['run{}_slo_attainment_percent'.format(r)] = g['attainment']
        table.append(row)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    print('\n{:<20}{:>6}{:>10}{:>8}{:>20}{:>12}'.format(
        'setting', 'runs', 'mean %', 'SD %', 'run means %', 'obs max %'))
    for row in table:
        print('{:<20}{:>6}{:>10.2f}{:>8.2f}{:>20}{:>12.0f}'.format(
            '{} {}'.format(row['mode'], row['configured_max_batch']), row['runs'],
            row['mean_of_run_means_percent'], row['sd_of_run_means_percent'],
            '/'.join('{:.1f}'.format(row['run{}_mean_percent'.format(r)]) for r in REPEATS),
            row['observed_max_percent_over_runs']))
    print('\nSaved:', args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
