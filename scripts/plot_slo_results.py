"""Draw the performance charts of the 84-run ResNet-50 comparison.

Run from the repository root (needs matplotlib):
    python scripts\\plot_slo_results.py

Inputs : results/resnet50/comparison/summary.csv  (84 per-run results)
         results/resnet50/comparison/comparison.csv (28 settings, mean and sample SD)
Outputs: results/resnet50/analysis/*.png and *.svg  (existing files are not overwritten)

Every bar is the mean of three PER-RUN values (the P95 of each run, the SLO attainment of
each run, ...) and every whisker is the sample standard deviation of those three values.
Requests of different runs are never pooled and no percentile is recomputed here. Before
drawing, the means and SDs are recomputed from summary.csv and must equal comparison.csv.
Each panel has its own y-axis unless stated; the focused P95 chart shares one clipped axis.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results' / 'resnet50' / 'comparison'
OUTPUT = ROOT / 'results' / 'resnet50' / 'analysis'

WORKLOADS = ['low', 'medium', 'high', 'bursty']
TITLES = {'low': 'Low (20 req/s)', 'medium': 'Medium (100 req/s)',
          'high': 'High (250 req/s)', 'bursty': 'Bursty (250 req/s while active)'}
POLICIES = [('static', 1), ('static', 4), ('static', 8), ('static', 16),
            ('adaptive', 16), ('adaptive_immediate', 16), ('slo', 16)]
LABELS = ['Static 1', 'Static 4', 'Static 8', 'Static 16',
          'Adaptive wait 16', 'Adaptive immediate 16', 'SLO-aware 16']
TICKS = ['Static\n1', 'Static\n4', 'Static\n8', 'Static\n16',
         'Adaptive\nwait', 'Adaptive\nimmed.', 'SLO-\naware']
# Categorical slots 1-7 of the validated reference palette, fixed order (a policy keeps its
# color in every chart). The three slots below 3:1 contrast on the surface are covered by
# the tick labels under every bar and by the legend.
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load():
    summary = read(INPUT / 'summary.csv')
    comparison = read(INPUT / 'comparison.csv')
    if len(summary) != 84 or len(comparison) != 28:
        raise ValueError('Expected 84 runs and 28 settings, found {} and {}'.format(
            len(summary), len(comparison)))
    data = {(r['workload'], r['mode'], int(r['configured_max_batch'])): r for r in comparison}
    if len(data) != 28 or any((w,) + p not in data for w in WORKLOADS for p in POLICIES):
        raise ValueError('comparison.csv does not hold the 4 workloads x 7 policies')
    if {r['repeats'] for r in comparison} != {'3'}:
        raise ValueError('Every setting must have 3 repeats')
    # means and SDs must be reproducible from the per-run rows
    for key, row in data.items():
        runs = [r for r in summary if (r['workload'], r['mode'], int(r['configured_max_batch'])) == key]
        if len(runs) != 3:
            raise ValueError('Setting {} has {} runs'.format(key, len(runs)))
        for metric in ('p95_ms', 'slo_attainment_percent', 'throughput_rps', 'mean_queue_wait_ms'):
            values = [float(r[metric]) for r in runs]
            if abs(statistics.mean(values) - float(row[metric + '_mean'])) > 1e-6 * (1 + abs(values[0])) or \
                    abs(statistics.stdev(values) - float(row[metric + '_std'])) > 1e-6 * (1 + abs(values[0])):
                raise ValueError('comparison.csv differs from summary.csv: {} {}'.format(key, metric))
        passes = sum(1 for r in runs if r['slo_pass'] == '1')
        if passes != int(row['runs_meeting_slo']):
            raise ValueError('Pass count differs: ' + str(key))
    return summary, data


def mean_sd(data, w, metric):
    rows = [data[(w,) + p] for p in POLICIES]
    return ([float(r[metric + '_mean']) for r in rows], [float(r[metric + '_std']) for r in rows])


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': .7,
        'axes.grid.axis': 'y', 'font.size': 9, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold'})


def save(fig, name, output):
    for ext in ('png', 'svg'):
        path = output / '{}.{}'.format(name, ext)
        if path.exists():
            raise FileExistsError('Not overwriting ' + str(path))
        fig.savefig(str(path), dpi=170 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def legend(fig, y=0.0):
    handles = [Patch(facecolor=c, label=l) for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, y), labelcolor=INK2)


def panels(title, subtitle, note, draw, ylabel, sharey=False):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), sharey=sharey)
    for ax, w in zip(axes.flat, WORKLOADS):
        draw(ax, w)
        ax.set_title(TITLES[w], loc='left')
        ax.set_xticks(range(7))
        ax.set_xticklabels(TICKS, fontsize=7.6)
        ax.tick_params(axis='x', length=0)
        if w in ('low', 'high'):
            ax.set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.01, 0.945, subtitle, ha='left', fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.88, bottom=0.17, hspace=0.42, wspace=0.30)
    legend(fig, 0.045)
    fig.text(0.01, 0.01, note, ha='left', fontsize=7.8, color=INK2)
    return fig


def bars(ax, w, data, metric, clip=None, fmt='{:,.0f}'):
    means, sds = mean_sd(data, w, metric)
    ax.bar(range(7), means, width=.72, color=COLORS, edgecolor='none', zorder=2)
    ax.errorbar(range(7), means, yerr=sds, fmt='none', ecolor=INK, elinewidth=1,
                capsize=2.5, capthick=1, zorder=3)
    if clip is not None:
        ax.set_ylim(0, clip)
        for i, m in enumerate(means):
            if m + sds[i] > clip:
                ax.text(i, clip * 0.985, '▲ ' + fmt.format(m), ha='center', va='top',
                        fontsize=7.6, color=INK, fontweight='bold',
                        bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.2, alpha=.85), zorder=5)
    return means, sds


def hline(ax, y, text, color=INK2):
    """Dashed reference line; its label sits outside the panel so it never covers a bar."""
    ax.axhline(y, color=color, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
    ax.text(1.012, y, text, transform=ax.get_yaxis_transform(), ha='left', va='center',
            fontsize=7.4, color=color, clip_on=False, linespacing=1.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary, data = load()
    any_row = next(iter(data.values()))
    target = float(any_row['slo_target_ms'])
    required = float(any_row['slo_required_percent'])
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    note = ('84 runs = 4 workloads x 7 policies x 3 repeats. Bar = mean of 3 per-run values, whisker = '
            'sample SD of those 3 values. Percentiles are computed per run, then averaged.')

    # 1. P95, full range (own y-axis per panel)
    def p95_full(ax, w):
        bars(ax, w, data, 'p95_ms')
        hline(ax, target, 'SLO' + chr(10) + '{:.0f} ms'.format(target))
        ax.set_ylim(bottom=0)
    save(panels('P95 end-to-end latency, full range',
                'Each panel has its own y-axis. Dashed line: the {:.0f} ms SLO.'.format(target),
                note, p95_full, 'P95 latency (ms)'), 'p95_latency_full', args.output)

    # 2. P95, zoomed (shared clipped axis)
    clip = 1.5 * target

    def p95_zoom(ax, w):
        bars(ax, w, data, 'p95_ms', clip=clip)
        hline(ax, target, 'SLO' + chr(10) + '{:.0f} ms'.format(target))
    save(panels('P95 end-to-end latency, zoomed to small values',
                'All panels share one y-axis cut at {:.0f} ms; bars above the cut show their value '
                '(▲). Dashed line: the {:.0f} ms SLO.'.format(clip, target),
                note, p95_zoom, 'P95 latency (ms)', sharey=True), 'p95_latency_zoom', args.output)

    # 3. SLO attainment with the 95% requirement and per-setting pass counts
    def attainment(ax, w):
        means, sds = bars(ax, w, data, 'slo_attainment_percent')
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        hline(ax, required, '{:.0f}%'.format(required) + chr(10) + 'required')
        for i, p in enumerate(POLICIES):
            passes = data[(w,) + p]['runs_meeting_slo']
            ax.text(i, 104, '{}/3'.format(passes), ha='center', va='bottom', fontsize=8,
                    color=INK2, zorder=5)
    save(panels('SLO attainment: requests finishing within {:.0f} ms'.format(target),
                'Labels above the bars: runs (of 3) that individually met the {:.0f}% requirement. '
                'Dashed line: the requirement.'.format(required),
                note, attainment, 'Requests meeting the SLO (%)', sharey=True), 'slo_attainment', args.output)

    # 4. throughput (own y-axis per panel)
    def throughput(ax, w):
        bars(ax, w, data, 'throughput_rps')
        rate = float(data[(w,) + POLICIES[0]]['target_rps'])
        if w != 'bursty':
            hline(ax, rate, 'offered' + chr(10) + '{:.0f} req/s'.format(rate))
        else:
            runs = [r for r in summary if r['workload'] == 'bursty']
            average = statistics.mean(float(r['generated_requests']) for r in runs) / float(
                runs[0]['duration_seconds'])
            hline(ax, average, 'planned' + chr(10) + '{:.1f} req/s'.format(average) + chr(10) + '(requests / 10 s)')
        ax.set_ylim(bottom=0)
    save(panels('Observed throughput',
                'Completed requests / (last completion − first arrival). Not a maximum capacity. '
                'Bursty: the interval is not the 10 s window (see docs/results.md).',
                note, throughput, 'Throughput (req/s)'), 'throughput', args.output)

    # 5. mean queue wait (own y-axis per panel)
    def queue_wait(ax, w):
        bars(ax, w, data, 'mean_queue_wait_ms')
        ax.set_ylim(bottom=0)
    save(panels('Mean queue wait',
                'Time from arrival to the start of service, averaged per run. Each panel has its own y-axis.',
                note, queue_wait, 'Mean queue wait (ms)'), 'queue_wait', args.output)

    print('Validated 84 runs / 28 settings against comparison.csv. Saved to', args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
