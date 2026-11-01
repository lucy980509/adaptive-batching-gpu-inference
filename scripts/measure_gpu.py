import argparse
import csv
import json
import statistics
import subprocess
from datetime import datetime

from run_experiments import ROOT, EXE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workload', choices=['low', 'medium', 'high', 'bursty'])
    parser.add_argument('mode', choices=['static', 'adaptive', 'adaptive_immediate', 'slo'])
    parser.add_argument('batch_size', type=int, choices=[1, 4, 8, 16, 32])
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()

    if not EXE.is_file():
        raise FileNotFoundError('Executable not found: ' + str(EXE))
    if args.run < 1:
        parser.error('--run must be positive')

    name = '{}_{}_b{}_run{}'.format(
        args.workload, args.mode, args.batch_size, args.run
    )
    output_dir = ROOT / 'results' / 'gpu_slo' / name
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
        raw_file.write('timestamp,gpu_util_percent,memory_used_mib\n')
        raw_file.flush()

        monitor = subprocess.Popen(
            ['nvidia-smi', '-i', '0',
             '--query-gpu=timestamp,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits', '-lms', '100'],
            stdout=raw_file, stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        try:
            with log_path.open('w', encoding='utf-8') as log_file:
                process = subprocess.Popen(
                    [str(EXE), args.workload, args.mode,
                     str(args.batch_size), str(request_path)],
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
        if (float(row['slo_target_ms']) != 20 or
                float(row['safety_margin_ms']) != 10 or
                float(row['slo_required_percent']) != 95):
            raise RuntimeError('Expected SLO 20 ms, margin 10 ms, requirement 95%. Check configuration and rebuild.')

    measured_utilization = []
    with raw_path.open(newline='', encoding='utf-8') as raw_file, samples_path.open(
        'w', newline='', encoding='utf-8'
    ) as output_file:
        reader = csv.DictReader(raw_file)
        writer = csv.writer(output_file)
        writer.writerow(['timestamp', 'phase', 'gpu_util_percent', 'memory_used_mib'])
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
            writer.writerow([timestamp, phase, utilization, memory])
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
        'max_wait_ms': 10 if args.mode == 'adaptive' else (
            0 if args.mode == 'adaptive_immediate' else None),
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
