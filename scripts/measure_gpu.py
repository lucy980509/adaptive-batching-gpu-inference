import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'build' / 'Release' / 'p3_inference.exe'
MANIFEST = ROOT / 'data' / 'imagenet' / 'val1000_manifest.json'
DEFAULT_OUTPUT_ROOT = ROOT / 'results' / 'resnet50' / 'gpu'


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(ROOT),
                                   universal_newlines=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workload', choices=['low', 'medium', 'high', 'bursty'])
    parser.add_argument('mode', choices=['static', 'adaptive', 'adaptive_immediate', 'slo'])
    parser.add_argument('batch_size', type=int, choices=[1, 4, 8, 16, 32])
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    # Optional diagnostic fields polled together with utilization and memory, for example
    # pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,
    # temperature.gpu,clocks_throttle_reasons.active. Without it the recorded columns and
    # the behavior are exactly as before. The fields only observe; no GPU setting is changed.
    parser.add_argument('--extra-fields', default='')
    # Any other option (--duration, --slo-ms, --margin-ms, --required-percent,
    # --max-wait-ms) is passed to p3_inference unchanged and checked against the CSV.
    args, extra = parser.parse_known_args()
    extra_fields = [f.strip() for f in args.extra_fields.split(',') if f.strip()]
    extra_names = [f.replace('.', '_') for f in extra_fields]

    if not EXE.is_file():
        raise FileNotFoundError('Executable not found: ' + str(EXE))
    if args.run < 1:
        parser.error('--run must be positive')

    name = '{}_{}_b{}_run{}'.format(
        args.workload, args.mode, args.batch_size, args.run
    )
    output_dir = args.output_root / name
    output_dir.mkdir(parents=True, exist_ok=False)

    request_path = output_dir / 'requests.csv'
    raw_path = output_dir / 'gpu_raw.csv'
    samples_path = output_dir / 'gpu_samples.csv'
    log_path = output_dir / 'inference.log'

    measure_start = None
    measure_end = None
    inference_returncode = None
    process = None

    with raw_path.open('w', newline='', encoding='utf-8') as raw_file:
        raw_file.write(','.join(['timestamp', 'gpu_util_percent', 'memory_used_mib'] + extra_names) + '\n')
        raw_file.flush()

        monitor = subprocess.Popen(
            ['nvidia-smi', '-i', '0',
             '--query-gpu=' + ','.join(['timestamp', 'utilization.gpu', 'memory.used'] + extra_fields),
             '--format=csv,noheader,nounits', '-lms', '100'],
            stdout=raw_file, stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        try:
            with log_path.open('w', encoding='utf-8') as log_file:
                process = subprocess.Popen(
                    [str(EXE), args.workload, args.mode,
                     str(args.batch_size), str(request_path)] + extra,
                    cwd=str(ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, universal_newlines=True,
                    encoding='utf-8', errors='replace', bufsize=1,
                )

                for line in process.stdout:
                    received_at = datetime.now()
                    log_file.write(line)
                    print(line, end='', flush=True)
                    marker = line.strip()
                    if marker == 'MEASURE_START':
                        measure_start = received_at
                    elif marker == 'MEASURE_END':
                        measure_end = received_at

                inference_returncode = process.wait()
                process.stdout.close()
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            if monitor.poll() is None:
                monitor.terminate()
            _, monitor_error = monitor.communicate()

    # Saved before any validation: if a later check fails, the measurement window can
    # still be recovered from this file and the raw files.
    (output_dir / 'markers.json').write_text(json.dumps({
        'measurement_start_host': measure_start.isoformat() if measure_start else None,
        'measurement_end_host': measure_end.isoformat() if measure_end else None,
        'inference_returncode': inference_returncode,
        'cli_options': extra}, indent=2), encoding='utf-8')

    if monitor_error:
        print('GPU monitor message:', monitor_error)
    if inference_returncode != 0:
        raise RuntimeError('Inference failed. Check ' + str(log_path))
    if measure_start is None or measure_end is None or measure_end <= measure_start:
        raise RuntimeError('Measurement markers missing or invalid. Check main.cpp and rebuild.')

    with request_path.open(newline='', encoding='utf-8-sig') as f:
        requests = list(csv.DictReader(f))
    if not requests:
        raise RuntimeError('No completed requests')
    for row in requests:
        if (row['mode'] != args.mode or row['workload'] != args.workload or
                int(row['max_batch_size']) != args.batch_size):
            raise RuntimeError('Request CSV does not match the command')
        if any(row[c] != requests[0][c] for c in (
                'slo_target_ms', 'safety_margin_ms', 'slo_required_percent',
                'target_rps', 'duration_seconds', 'max_wait_ms')):
            raise RuntimeError('Settings change inside one run')
    # Options are passed to every policy, but only some policies use them.
    # --max-wait-ms: only Adaptive wait uses it (recorded as the value in use).
    # Adaptive immediate never waits (recorded as 0), Static and SLO-aware do not
    # use a maximum wait (recorded blank). So the recorded value is checked against
    # what the policy really applies, and the option is compared only for Adaptive wait.
    recorded_wait = requests[0]['max_wait_ms']
    passed_wait = None
    for index in range(0, len(extra) - 1, 2):
        if extra[index] == '--max-wait-ms':
            passed_wait = float(extra[index + 1])
    if args.mode == 'adaptive_immediate':
        if recorded_wait != '0':
            raise RuntimeError('Adaptive immediate must record max_wait_ms=0, found ' + repr(recorded_wait))
    elif args.mode == 'adaptive':
        if recorded_wait == '' or (passed_wait is not None and float(recorded_wait) != passed_wait):
            raise RuntimeError('Adaptive wait recorded max_wait_ms={!r}, passed {}'.format(
                recorded_wait, passed_wait))
    elif recorded_wait != '':
        raise RuntimeError('{} does not use a maximum wait but recorded {!r}'.format(
            args.mode, recorded_wait))

    # The remaining options apply to every policy and must equal the recorded value.
    override_columns = {'--duration': 'duration_seconds', '--slo-ms': 'slo_target_ms',
                        '--margin-ms': 'safety_margin_ms',
                        '--required-percent': 'slo_required_percent'}
    for index in range(0, len(extra) - 1, 2):
        column = override_columns.get(extra[index])
        if column and float(requests[0][column]) != float(extra[index + 1]):
            raise RuntimeError('CSV {}={} differs from {} {}'.format(
                column, requests[0][column], extra[index], extra[index + 1]))

    # Every generated request must complete exactly once, in the planned image order.
    images = int(json.loads(MANIFEST.read_text(encoding='utf-8'))['selection']['count'])
    ids = sorted(int(r['request_id']) for r in requests)
    if ids != list(range(1, len(requests) + 1)):
        raise RuntimeError('Requests do not complete exactly once (ids are not 1..N)')
    if any(int(r['image_index']) != (int(r['request_id']) - 1) % images for r in requests):
        raise RuntimeError('Image order broken')
    log_text = log_path.read_text(encoding='utf-8', errors='replace')
    counts = {}
    for key, label in (('scheduled', 'Scheduled requests'), ('generated', 'Generated requests')):
        match = re.search(label + r': (\d+)', log_text)
        counts[key] = int(match.group(1)) if match else None
    if counts['generated'] != len(requests):
        raise RuntimeError('Generated {} requests but {} completed'.format(
            counts['generated'], len(requests)))

    measured_utilization = []
    with raw_path.open(newline='', encoding='utf-8') as raw_file, samples_path.open(
        'w', newline='', encoding='utf-8'
    ) as output_file:
        reader = csv.DictReader(raw_file)
        writer = csv.writer(output_file)
        writer.writerow(['timestamp', 'phase', 'gpu_util_percent', 'memory_used_mib'] + extra_names)
        for row in reader:
            timestamp = row['timestamp'].strip()
            sample_time = datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S.%f')
            if sample_time < measure_start:
                phase = 'initialization_or_warmup'
            elif sample_time <= measure_end:
                phase = 'measurement'
            else:
                phase = 'shutdown'
            utilization = row['gpu_util_percent'].strip()
            memory = row['memory_used_mib'].strip()
            writer.writerow([timestamp, phase, utilization, memory] + [
                (row[n] or '').strip() for n in extra_names])
            if phase == 'measurement':
                try:
                    value = float(utilization)
                    if 0 <= value <= 100:
                        measured_utilization.append(value)
                except ValueError:
                    pass

    if not measured_utilization:
        raise RuntimeError('No valid GPU samples. Check ' + str(raw_path))

    met = sum(int(row['slo_met']) for row in requests)
    metadata = {
        'workload': args.workload,
        'mode': args.mode,
        'configured_batch_size': args.batch_size,
        'max_wait_ms': (float(requests[0]['max_wait_ms'])
                        if requests[0]['max_wait_ms'] != '' else None),
        'target_rps': float(requests[0]['target_rps']),
        'duration_seconds': float(requests[0]['duration_seconds']),
        'slo_target_ms': float(requests[0]['slo_target_ms']),
        'safety_margin_ms': float(requests[0]['safety_margin_ms']),
        'slo_required_percent': float(requests[0]['slo_required_percent']),
        'repeat': args.run,
        'gpu_index': 0,
        'query_interval_ms': 100,
        'measurement_start_host': measure_start.isoformat(),
        'measurement_end_host': measure_end.isoformat(),
        'valid_measurement_samples': len(measured_utilization),
        'gpu_measurement_enabled': True,
        'scope': 'whole_gpu',
        'mean_sampled_gpu_util_percent': statistics.mean(measured_utilization),
        'max_sampled_gpu_util_percent': max(measured_utilization),
        'completed_requests': len(requests),
        'scheduled_requests': counts['scheduled'],
        'generated_requests': counts['generated'],
        'measurement_seconds': (measure_end - measure_start).total_seconds(),
        'separate_from_comparison_runs': True,
        'purpose': 'GPU utilization of one run per policy and repeat; collected separately '
                   'from the 84 performance runs in results/resnet50/comparison and not '
                   'measured at the same time as them',
        'samples_scope': 'only samples between the host-received MEASURE_START and '
                         'MEASURE_END markers; model loading, warm-up and calibration are '
                         'labelled initialization_or_warmup in gpu_samples.csv and excluded',
        'cli_options': extra,
        'extra_gpu_fields': extra_fields,
        'max_wait_ms_passed': passed_wait,
        'max_wait_ms_semantics': 'recorded value is what the policy applied: Adaptive wait = the wait in use, Adaptive immediate = 0 (never waits), Static and SLO-aware = not applicable (null)',
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
        'executable_sha256': hashlib.sha256(EXE.read_bytes()).hexdigest(),
        'slo_met_requests': met,
        'slo_attainment_percent': 100.0 * met / len(requests),
        'calibration_csv': request_path.name + '.service_times.csv',
    }
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)

    print('Mean sampled GPU utilization: {:.2f}%'.format(statistics.mean(measured_utilization)))
    print('Max sampled GPU utilization: {:.2f}%'.format(max(measured_utilization)))
    print('Saved to:', output_dir)


if __name__ == '__main__':
    main()
