"""Measure ResNet-50 service time per batch size (sequential batches, no arriving traffic).

Run from the repository root after building p3_inference (any Python 3.6+):
    python scripts\\profile_batches.py

For each batch size 1, 4, 8, 16, 32 a separate p3_inference process runs
`batch_profile`: it loads the model and the first N prepared ImageNet images (fixed
indices 0..N-1, fixed order), runs 3 untimed warm-up batches, then times 10 batches.
If a size fails (for example out of GPU memory) the failure is recorded and no larger
size is attempted.

Output (results/resnet50/batch_profile/):
    batch_profile.csv   summary per batch size
    batch_<N>.csv       the 10 raw service times of each size
    metadata.json       settings, GPU info and caveats
Throughput here is batch_size / mean service time of back-to-back batches. It is NOT the
throughput of a serving run with arriving requests. GPU memory is whole-GPU used memory
sampled about every 100 ms with nvidia-smi: a sampled value, not an exact peak.
"""
import argparse
import csv
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'build' / 'Release' / 'p3_inference.exe'
SIZES = [1, 4, 8, 16, 32]
WARMUP_RUNS = 3
MEASURED_RUNS = 10


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def gpu_memory_mib():
    text = subprocess.check_output(
        ['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
         '--format=csv,noheader,nounits'], universal_newlines=True)
    return float(text.strip())


def gpu_info():
    text = subprocess.check_output(
        ['nvidia-smi', '-i', '0', '--query-gpu=name,driver_version,memory.total',
         '--format=csv,noheader'], universal_newlines=True)
    name, driver, total = [part.strip() for part in text.strip().split(',')]
    return {'name': name, 'driver': driver, 'memory_total': total}


def run_size(size, output_dir):
    csv_path = output_dir / 'batch_{}.csv'.format(size)
    log_path = output_dir / 'batch_{}.log'.format(size)
    baseline = gpu_memory_mib()
    peak = baseline
    samples = 0

    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [str(EXE), 'none', 'batch_profile', str(size), str(csv_path)],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            peak = max(peak, gpu_memory_mib())
            samples += 1
            time.sleep(0.1)

    if process.returncode != 0 or not csv_path.is_file():
        tail = log_path.read_text(encoding='utf-8', errors='replace').strip().splitlines()[-5:]
        return None, {'batch_size': size, 'returncode': process.returncode,
                      'log_tail': tail}

    with csv_path.open(newline='', encoding='utf-8') as f:
        times = [float(row['service_ms']) for row in csv.DictReader(f)]
    if len(times) != MEASURED_RUNS:
        raise RuntimeError('Expected {} timings for batch {}, found {}'.format(
            MEASURED_RUNS, size, len(times)))

    mean_ms = statistics.mean(times)
    return {
        'batch_size': size,
        'measured_runs': len(times),
        'mean_service_ms': mean_ms,
        'median_service_ms': statistics.median(times),
        'p95_service_ms': percentile(times, 95),
        'min_service_ms': min(times),
        'max_service_ms': max(times),
        'observed_sequential_images_per_s': size / (mean_ms / 1000.0),
        'gpu_memory_before_mib': baseline,
        'gpu_memory_peak_sampled_mib': peak,
        'gpu_memory_rise_sampled_mib': peak - baseline,
        'gpu_memory_samples': samples,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results' / 'resnet50' / 'batch_profile')
    args = parser.parse_args()

    if not EXE.is_file():
        raise FileNotFoundError('Executable not found: ' + str(EXE))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output not empty; keep it and choose --output: ' + str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)

    rows, failure = [], None
    for size in SIZES:
        print('Batch {} ...'.format(size), flush=True)
        row, failure = run_size(size, args.output)
        if failure:
            print('  FAILED (return code {}); stopping before larger sizes.'.format(
                failure['returncode']))
            break
        rows.append(row)
        print('  mean {:.2f} ms | median {:.2f} ms | P95 {:.2f} ms | {:.1f} img/s'.format(
            row['mean_service_ms'], row['median_service_ms'], row['p95_service_ms'],
            row['observed_sequential_images_per_s']), flush=True)

    if rows:
        with (args.output / 'batch_profile.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'model': 'resnet50 (torchvision IMAGENET1K_V2, FP32, ONNX Runtime 1.26.0 CUDA EP)',
        'command_per_size': 'p3_inference none batch_profile <N> batch_<N>.csv',
        'batch_sizes_requested': SIZES,
        'batch_sizes_completed': [r['batch_size'] for r in rows],
        'failure': failure,
        'warmup_runs': WARMUP_RUNS,
        'measured_runs': MEASURED_RUNS,
        'inputs': 'prepared ImageNet validation subset, images 0..N-1 in file order, '
                  'the same images for every run of a size; the images of a smaller size '
                  'are a prefix of the larger ones',
        'service_time_definition': 'batch assembly + tensor setup + session.Run; model '
                                   'loading, image loading and warm-up excluded; no logits '
                                   'copied, no per-run console output',
        'throughput_definition': 'batch_size / mean service time of back-to-back batches '
                                 '(observed sequential rate), not serving throughput with '
                                 'arriving traffic',
        'percentile_note': 'P95 of 10 runs is linearly interpolated between the two '
                           'largest values and is close to the maximum',
        'gpu_memory_note': 'whole-GPU memory.used from nvidia-smi sampled about every '
                           '100 ms (WDDM, other processes included); sampled value, not '
                           'an exact peak',
        'gpu': gpu_info(),
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved:', args.output)
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
