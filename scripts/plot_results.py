import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = [('static', 1), ('static', 4), ('static', 8), ('static', 16),
            ('static', 32), ('adaptive', 32), ('adaptive_immediate', 32)]
LABELS = ['Static 1', 'Static 4', 'Static 8', 'Static 16', 'Static 32',
          'Adaptive wait', 'Adaptive immediate']
COLORS = ['#23618c', '#5487a5', '#7aa7bd', '#9cb8c7', '#b5c9d1',
          '#d57930', '#198579']
WORKLOADS = ['low', 'medium', 'high', 'bursty']
FOCUS = [0, 5, 6]


def read_comparison(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 28:
        raise ValueError('Expected 28 workload/policy groups')
    data = {}
    for r in rows:
        key = (r['workload'], r['mode'], int(r['configured_batch_size']))
        if key in data or int(r['repeats']) != 3:
            raise ValueError('Duplicate group or repeat count other than 3')
        for k in ('p50_ms_mean', 'p95_ms_mean', 'p95_ms_std', 'p99_ms_mean',
                  'mean_queue_wait_ms_mean', 'throughput_rps_mean', 'throughput_rps_std'):
            r[k] = float(r[k])
            if not np.isfinite(r[k]) or r[k] < 0:
                raise ValueError('Invalid metric: ' + k)
        data[key] = r
    for w in WORKLOADS:
        for m, b in POLICIES:
            if (w, m, b) not in data:
                raise ValueError('Missing workload/policy group')
    return data


def save(fig, out, name):
    fig.savefig(str(out / (name + '.png')), dpi=180, bbox_inches='tight')
    fig.savefig(str(out / (name + '.svg')), bbox_inches='tight')
    plt.close(fig)


def policy_panels(data, out, metric, name, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, w in zip(axes.flat, WORKLOADS):
        vals = [data[(w, m, b)][metric] for m, b in POLICIES]
        ax.bar(range(7), vals, color=COLORS, width=.65)
        ax.set_yscale('log')
        ax.set_ylim(.01 if 'wait' in metric else .5, 30000)
        ax.set_xticks(range(7))
        ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
        ax.set_title(w.capitalize(), loc='left', fontweight='bold')
        ax.set_ylabel('Milliseconds (log scale)')
        for i, v in enumerate(vals):
            ax.text(i, v * 1.15, '{:.2f}'.format(v), ha='center', fontsize=8)
    fig.suptitle(title, fontweight='bold', fontsize=15)
    fig.text(.5, .01, '84 unprofiled runs; mean of 3 run-level metrics per group. GPU samples are separate.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    save(fig, out, name)


def focus_latency(data, out):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, w in zip(axes, WORKLOADS):
        for x, i in enumerate(FOCUS):
            m, b = POLICIES[i]
            r = data[(w, m, b)]
            ax.bar(x, r['p95_ms_mean'], yerr=r['p95_ms_std'], capsize=4,
                   color=COLORS[i], width=.65)
        ax.set_title(w.capitalize(), fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_xticklabels([LABELS[i] for i in FOCUS], rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 26)
    axes[0].set_ylabel('P95 end-to-end latency (ms)')
    fig.suptitle('Waiting costs latency at these tested loads', fontweight='bold')
    fig.text(.5, .015, 'Mean +/- sample SD across 3 runs; error bars are not confidence intervals.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    save(fig, out, 'p95_focus')


def rate_curves(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rates = [2, 20, 200]
    for i, (m, b) in enumerate(POLICIES):
        rows = [data[(w, m, b)] for w in WORKLOADS[:3]]
        for ax, metric in zip(axes, ['p95_ms_mean', 'throughput_rps_mean']):
            ax.plot(rates, [r[metric] for r in rows], marker='o', markersize=4,
                    color=COLORS[i], label=LABELS[i], linewidth=1.4,
                    linestyle='--' if i == 5 else '-')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(rates)
            ax.set_xticklabels(['2', '20', '200'])
            ax.set_xlabel('Configured target arrival rate (requests/s)')
    axes[0].set_ylabel('P95 end-to-end latency (ms, log scale)')
    axes[1].set_ylabel('Observed completion throughput (requests/s)')
    axes[1].plot(rates, rates, ':', color='#333333', alpha=.7, label='Target rate reference')
    axes[0].set_title('Latency vs configured arrival rate', loc='left', fontweight='bold')
    axes[1].set_title('Throughput follows offered load', loc='left', fontweight='bold')
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Continuous workloads only: low / medium / high', fontweight='bold')
    fig.text(.5, .12, '3-run means. Connecting lines guide the eye; these are not saturation/capacity curves. Bursty is excluded.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, .93))
    save(fig, out, 'arrival_rate_metrics')


def gpu_results(gpu_dir, out):
    rows = []
    series = []
    skipped = []
    for folder in sorted(gpu_dir.iterdir()):
        if not folder.is_dir():
            continue
        meta_path, samples_path = folder / 'metadata.json', folder / 'gpu_samples.csv'
        if not meta_path.is_file() or not samples_path.is_file():
            skipped.append(folder.name)
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if not meta.get('gpu_measurement_enabled'):
            continue
        with samples_path.open(newline='', encoding='utf-8-sig') as f:
            samples = [r for r in csv.DictReader(f) if r['phase'] == 'measurement']
        if len(samples) != meta['valid_measurement_samples'] or not samples:
            raise ValueError('GPU sample count mismatch: ' + folder.name)
        values = [float(r['gpu_util_percent']) for r in samples]
        start = datetime.strptime(meta['measurement_start_host'], '%Y-%m-%dT%H:%M:%S.%f')
        offsets = [(datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f') - start).total_seconds()
                   for r in samples]
        with (folder / 'requests.csv').open(newline='', encoding='utf-8-sig') as f:
            requests = list(csv.DictReader(f))
        elapsed_ms = max(float(r['arrival_offset_ms']) + float(r['e2e_latency_ms']) for r in requests) - min(float(r['arrival_offset_ms']) for r in requests)
        p95 = float(np.percentile([float(r['e2e_latency_ms']) for r in requests], 95))
        counts = Counter(int(r['actual_batch_size']) for r in requests)
        result = dict(run=folder.name, workload=meta['workload'], mode=meta['mode'],
                      configured_batch_size=meta['configured_batch_size'], samples=len(samples),
                      mean_gpu_util_percent=float(np.mean(values)), max_gpu_util_percent=max(values),
                      completed_requests=len(requests), throughput_rps=len(requests) / (elapsed_ms / 1000),
                      p95_ms=p95, scope=meta['scope'])
        rows.append(result)
        series.append((result, offsets, values))
        print(folder.name, 'request-weighted batch counts', dict(sorted(counts.items())))
    if not rows:
        raise ValueError('No successful GPU measurements found')
    with (out / 'gpu_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    order = {'static': 0, 'adaptive': 1, 'adaptive_immediate': 2}
    series.sort(key=lambda x: order[x[0]['mode']])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(series))
    mean = [s[0]['mean_gpu_util_percent'] for s in series]
    peak = [s[0]['max_gpu_util_percent'] for s in series]
    a = ax.bar(x - .18, mean, .36, color='#23618c', label='Mean sampled utilization')
    b = ax.bar(x + .18, peak, .36, color='#9cb8c7', label='Maximum sampled utilization')
    for bars in [a, b]:
        for v in bars:
            ax.text(v.get_x() + v.get_width()/2, v.get_height()+.18,
                    '{:.2f}%'.format(v.get_height()), ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Static 1', 'Adaptive wait', 'Adaptive immediate'])
    ax.set_ylabel('Whole-device GPU utilization (%)')
    ax.set_ylim(0, max(peak)*1.35)
    ax.set_title('High workload: low sampled GPU utilization', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    fig.text(.5, .015, 'One successful monitored run per policy; 100 ms polling, rolling device metric. No significance claim.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, out, 'gpu_high')
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True, sharey=True)
    for ax, (r, offsets, values), label in zip(axes, series, ['Static 1', 'Adaptive wait', 'Adaptive immediate']):
        ax.plot(offsets, values, color='#23618c', linewidth=1.2)
        ax.set_title('{} | mean {:.2f}% | {} samples'.format(label, r['mean_gpu_util_percent'], r['samples']),
                     loc='left', fontsize=10)
        ax.set_ylabel('GPU %')
        ax.set_ylim(0, 14)
    axes[-1].set_xlabel('Seconds since host received MEASURE_START (each run starts separately)')
    fig.suptitle('Separate GPU monitoring runs: measurement phase only', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, out, 'gpu_timeline')
    print('Skipped incomplete folders:', skipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--gpu-dir', type=Path, default=ROOT / 'results' / 'gpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'analysis')
    args = parser.parse_args()
    source = args.comparison or ROOT / 'results' / 'combined' / 'comparison.csv'
    if args.comparison is None and not source.is_file():
        source = ROOT / 'results' / 'analysis' / 'comparison_input.csv'
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'axes.axisbelow': True, 'grid.alpha': .18, 'font.size': 10})
    data = read_comparison(source)
    policy_panels(data, args.output, 'p95_ms_mean', 'p95_all_policies', 'P95 latency by workload and batching policy')
    policy_panels(data, args.output, 'mean_queue_wait_ms_mean', 'queue_wait', 'Mean queue wait explains the latency penalty')
    focus_latency(data, args.output)
    rate_curves(data, args.output)
    gpu_results(args.gpu_dir, args.output)
    print('Validated 28 groups / 84 benchmark runs; wrote 6 PNG + 6 SVG charts.')
    print('Output:', args.output)


if __name__ == '__main__':
    main()
