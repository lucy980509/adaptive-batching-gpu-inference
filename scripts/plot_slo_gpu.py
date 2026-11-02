"""Draw the separate GPU-utilization chart (High workload, 21 monitored runs).

Run from the repository root after scripts\\summarize_gpu.py:
    python scripts\\plot_slo_gpu.py

Input : results/resnet50/gpu/summary.csv
Output: results/resnet50/analysis/gpu_utilization_high.png / .svg (not overwritten)

A bar is the mean of the 3 per-run means (each run's mean sampled utilization over its
MEASURE_START..MEASURE_END samples); the whisker is the sample SD of those 3 run means; the
dots are the 3 run means. The ~100 ms samples inside a run are not used as repeats. These
runs are separate from the 84 performance runs, and only the High workload was measured.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_slo_results import COLORS, GRID, INK, INK2, LABELS, POLICIES, SURFACE, TICKS, style, legend

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results' / 'resnet50' / 'gpu' / 'summary.csv'
OUTPUT = ROOT / 'results' / 'resnet50' / 'analysis'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()

    with args.input.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    index = {(r['mode'], int(r['configured_max_batch'])): r for r in rows}
    if len(rows) != 7 or any(p not in index for p in POLICIES) or {r['runs'] for r in rows} != {'3'}:
        raise ValueError('Expected 7 settings with 3 runs each')
    means, sds, runs, seconds = [], [], [], []
    for p in POLICIES:
        row = index[p]
        values = [float(row['run{}_mean_percent'.format(r)]) for r in (1, 2, 3)]
        if abs(statistics.mean(values) - float(row['mean_of_run_means_percent'])) > 1e-9 or \
                abs(statistics.stdev(values) - float(row['sd_of_run_means_percent'])) > 1e-9:
            raise ValueError('summary.csv mean/SD do not match its run means: ' + str(p))
        means.append(statistics.mean(values))
        sds.append(statistics.stdev(values))
        runs.append(values)
        seconds.append(float(row['mean_measurement_seconds']))

    style()
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    xs = range(7)
    ax.bar(xs, means, width=.72, color=COLORS, edgecolor='none', zorder=2)
    ax.errorbar(xs, means, yerr=sds, fmt='none', ecolor=INK, elinewidth=1, capsize=3,
                capthick=1, zorder=3)
    for x, values in zip(xs, runs):
        ax.scatter([x - .12, x, x + .12], values, s=14, color=SURFACE, edgecolor=INK,
                   linewidth=.9, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(TICKS, fontsize=8.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Mean sampled GPU utilization (%)')
    ax.text(0, means[0] + sds[0] + 4, '{:.1f} s window*'.format(seconds[0]), ha='center',
            fontsize=8, color=INK2)
    fig.suptitle('GPU utilization, High workload (separate monitored runs)', x=0.01, ha='left',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.text(0.01, 0.925, 'Whole-GPU nvidia-smi utilization sampled about every 100 ms between '
             'MEASURE_START and MEASURE_END. Bar: mean of 3 run means; whisker: sample SD; dots: the run means.',
             ha='left', fontsize=8.6, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.25)
    legend(fig, 0.065)
    fig.text(0.01, 0.01, '21 runs (7 settings x 3), run separately from the 84 performance runs; only High was '
             'measured. *Static 1 drains a backlog, so its window is longer than the ~10 s of the others.\n'
             'Utilization shows how busy the GPU was, not which policy is more efficient or faster.',
             ha='left', fontsize=7.8, color=INK2)
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        path = args.output / 'gpu_utilization_high.{}'.format(ext)
        if path.exists():
            raise FileExistsError('Not overwriting ' + str(path))
        fig.savefig(str(path), dpi=170 if ext == 'png' else None, bbox_inches='tight')
    print('Saved GPU chart to', args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
