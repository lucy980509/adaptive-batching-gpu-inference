import argparse
import csv
from collections import Counter
from pathlib import Path


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError('No completed requests in CSV')

    required = ['mode', 'slo_target_ms', 'slo_required_percent',
                'slack_at_service_start_ms', 'slo_met', 'estimated_service_ms']
    if any(column not in rows[0] for column in required):
        raise ValueError('Use a request CSV from the SLO extension, not a calibration CSV')

    groups = {(r['mode'], r['slo_target_ms'], r['slo_required_percent']) for r in rows}
    if len(groups) != 1:
        raise ValueError('Expected one policy and one SLO configuration per file')
    if any(r['slo_met'] not in ('0', '1') for r in rows):
        raise ValueError('Invalid slo_met flag')

    met = sum(int(r['slo_met']) for r in rows)
    rate = 100.0 * met / len(rows)
    target = float(rows[0]['slo_required_percent'])
    latencies = [float(r['e2e_latency_ms']) for r in rows]
    slack = [float(r['slack_at_service_start_ms']) for r in rows]
    sizes = Counter(int(r['actual_batch_size']) for r in rows)

    print('Mode:', rows[0]['mode'])
    print('Workload:', rows[0]['workload'])
    print('Completed requests:', len(rows))
    print('SLO target: {} ms'.format(rows[0]['slo_target_ms']))
    print('SLO met requests: {}/{}'.format(met, len(rows)))
    print('SLO attainment: {:.3f}%'.format(rate))
    print('SLO status:', 'PASS' if rate >= target else 'FAIL')
    for p in (50, 95, 99):
        print('P{}: {:.3f} ms'.format(p, percentile(latencies, p)))
    print('Minimum slack at service start: {:.3f} ms'.format(min(slack)))
    print('Request-weighted batch counts:', dict(sorted(sizes.items())))


if __name__ == '__main__':
    main()
