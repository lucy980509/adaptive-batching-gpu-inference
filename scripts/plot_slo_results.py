import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ['low', 'medium', 'high', 'bursty']
POLICIES = [('static', n) for n in (1, 4, 8, 16, 32)] + [
    ('adaptive', 32), ('adaptive_immediate', 32), ('slo', 32)]
LABELS = ['Static 1', 'Static 4', 'Static 8', 'Static 16', 'Static 32',
          'Adaptive wait', 'Adaptive immediate', 'SLO-aware']
COLORS = ['#23618c', '#5487a5', '#7aa7bd', '#9cb8c7', '#b5c9d1',
          '#d57930', '#198579', '#87559b']


def save(fig, directory, name):
    fig.savefig(str(directory / (name + '.png')), dpi=180, bbox_inches='tight')
    fig.savefig(str(directory / (name + '.svg')), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'results' / 'resnet50' / 'policy_comparison')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'resnet50' / 'analysis')
    args = parser.parse_args()
    with (args.input / 'comparison.csv').open(newline='', encoding='utf-8-sig') as f:
        comparison = list(csv.DictReader(f))
    with (args.input / 'summary.csv').open(newline='', encoding='utf-8-sig') as f:
        summary = list(csv.DictReader(f))
    if len(comparison) != 32 or len(summary) != 96:
        raise ValueError('Expected 32 comparison groups and 96 runs')
    data = {}
    for row in comparison:
        key = (row['workload'], row['mode'], int(row['configured_batch_size']))
        if key in data:
            raise ValueError('Duplicate comparison group')
        if row['repeats'] != '3' or float(row['slo_target_ms']) != 20 or float(row['safety_margin_ms']) != 10:
            raise ValueError('Unexpected experiment settings')
        data[key] = row
    for w in WORKLOADS:
        for m, b in POLICIES:
            data[(w, m, b)]
            runs = [r for r in summary if r['workload'] == w and r['mode'] == m and int(r['configured_batch_size']) == b]
            if sorted(int(r['repeat']) for r in runs) != [1, 2, 3]:
                raise ValueError('Missing or duplicate repeat')
            for metric in ['p95_ms', 'slo_attainment_percent', 'throughput_rps']:
                observed = sum(float(r[metric]) for r in runs) / 3
                if abs(observed - float(data[(w, m, b)][metric + '_mean'])) > .000001:
                    raise ValueError('Summary/comparison mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'axes.axisbelow': True, 'grid.alpha': .18, 'font.size': 10})

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, w in zip(axes.flat, WORKLOADS):
        rows = [data[(w, m, b)] for m, b in POLICIES]
        vals = [float(r['slo_attainment_percent_mean']) for r in rows]
        ax.bar(range(8), vals, color=COLORS)
        ax.axhline(95, color='#ad3030', linestyle='--', linewidth=1)
        for i, v in enumerate(vals):
            ax.text(i, v + 2, '{:.1f}'.format(v), ha='center', fontsize=8)
        ax.set_ylim(0, 112)
        ax.set_title(w.capitalize(), loc='left', fontweight='bold')
        ax.set_ylabel('Requests meeting 20 ms deadline (%)')
        ax.set_xticks(range(8))
        ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
    fig.suptitle('SLO attainment: means across three runs', fontweight='bold', fontsize=15)
    fig.text(.5, .01, 'Dashed line: 95% requirement. Means are not pooled request percentages. No GPU monitoring in these runs.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    save(fig, args.output, 'slo_attainment')

    focus = [0, 5, 6, 7]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)
    for ax, w in zip(axes, WORKLOADS):
        for x, i in enumerate(focus):
            m, b = POLICIES[i]
            r = data[(w, m, b)]
            ax.bar(x, float(r['p95_ms_mean']), yerr=float(r['p95_ms_std']),
                   color=COLORS[i], capsize=4, width=.7)
        ax.axhline(20, color='#ad3030', linestyle='--', linewidth=1)
        ax.set_ylim(0, 29)
        ax.set_title(w.capitalize(), fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels([LABELS[i] for i in focus], rotation=40, ha='right', fontsize=8)
    axes[0].set_ylabel('P95 end-to-end latency (ms)')
    fig.suptitle('Waiting policies vs immediate dispatch', fontweight='bold')
    fig.text(.5, .01, 'Mean +/- sample SD of 3 run-level P95s; not confidence intervals. Dashed line: 20 ms reference.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93))
    save(fig, args.output, 'slo_p95_focus')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, w in zip(axes.flat, WORKLOADS):
        vals = [float(data[(w, m, b)]['p95_ms_mean']) for m, b in POLICIES]
        ax.bar(range(8), vals, color=COLORS)
        ax.set_yscale('log')
        ax.set_ylim(.5, 30000)
        ax.axhline(20, color='#ad3030', linestyle='--', linewidth=1)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.15, '{:.1f}'.format(v), ha='center', fontsize=8)
        ax.set_title(w.capitalize(), loc='left', fontweight='bold')
        ax.set_ylabel('P95 latency (ms, log scale)')
        ax.set_xticks(range(8))
        ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
    fig.suptitle('All eight policies: P95 latency', fontweight='bold', fontsize=15)
    fig.text(.5, .01, '96 runs; mean of 3 run-level P95s. Dashed line: 20 ms reference.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    save(fig, args.output, 'slo_p95_all')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, w in zip(axes.flat, WORKLOADS):
        rows = [data[(w, m, b)] for m, b in POLICIES]
        vals = [float(r['throughput_rps_mean']) for r in rows]
        ax.bar(range(8), vals, yerr=[float(r['throughput_rps_std']) for r in rows],
               color=COLORS, capsize=3)
        ax.set_ylim(0, max(vals)*1.2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals)*.025, '{:.2f}'.format(v), ha='center', fontsize=8)
        ax.set_title(w.capitalize(), loc='left', fontweight='bold')
        ax.set_ylabel('Observed completion throughput (requests/s)')
        ax.set_xticks(range(8))
        ax.set_xticklabels(LABELS, rotation=30, ha='right', fontsize=8)
    fig.suptitle('Observed throughput at the tested offered loads', fontweight='bold', fontsize=15)
    fig.text(.5, .01, 'Denominator: first actual arrival to last finish; different panel scales. Not maximum capacity. Mean +/- sample SD.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    save(fig, args.output, 'slo_throughput')
    print('Validated 96 run summaries / 32 comparison groups.')
    print('Saved 4 PNG + 4 SVG charts:', args.output)


if __name__ == '__main__':
    main()
