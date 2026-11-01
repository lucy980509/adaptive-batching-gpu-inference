import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
POLICIES = [('static', 1, 'Static 1'), ('adaptive', 32, 'Adaptive wait'),
            ('adaptive_immediate', 32, 'Adaptive immediate'), ('slo', 32, 'SLO-aware')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'results' / 'resnet50' / 'gpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'resnet50' / 'analysis')
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()
    results = []
    for mode, size, label in POLICIES:
        folder = args.input / 'high_{}_b{}_run{}'.format(mode, size, args.run)
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        if (meta['mode'] != mode or meta['workload'] != 'high' or
                meta['configured_batch_size'] != size or not meta['gpu_measurement_enabled'] or
                meta['scope'] != 'whole_gpu' or meta['slo_target_ms'] != 20 or meta['safety_margin_ms'] != 10):
            raise ValueError('Unexpected configuration: ' + str(folder))
        with (folder / 'gpu_samples.csv').open(newline='', encoding='utf-8-sig') as f:
            samples = [float(r['gpu_util_percent']) for r in csv.DictReader(f) if r['phase'] == 'measurement']
        if not samples or len(samples) != meta['valid_measurement_samples']:
            raise ValueError('Invalid GPU sample count')
        with (folder / 'requests.csv').open(newline='', encoding='utf-8-sig') as f:
            requests = list(csv.DictReader(f))
        met = sum(int(r['slo_met']) for r in requests)
        if len(requests) != meta['completed_requests'] or met != meta['slo_met_requests']:
            raise ValueError('Request/metadata count mismatch')
        violations = [r for r in requests if r['slo_met'] == '0']
        positive_slack_misses = sum(float(r['slack_at_service_start_ms']) > 0 for r in violations)
        mean_gpu = statistics.mean(samples)
        if abs(mean_gpu - meta['mean_sampled_gpu_util_percent']) > .000001:
            raise ValueError('GPU mean/metadata mismatch')
        results.append(dict(mode=mode, configured_batch_size=size, repeat=args.run,
                            valid_samples=len(samples), mean_gpu_percent=mean_gpu,
                            max_gpu_percent=max(samples), completed_requests=len(requests),
                            slo_met_requests=met, slo_attainment_percent=100.0 * met / len(requests),
                            slo_violations=len(violations), positive_slack_violations=positive_slack_misses))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'slo_gpu_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.axisbelow': True,
                         'axes.grid': True, 'grid.alpha': .18})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(4))
    means = [r['mean_gpu_percent'] for r in results]
    maxima = [r['max_gpu_percent'] for r in results]
    a = ax.bar([v-.18 for v in x], means, .36, color='#23618c', label='Mean sampled utilization')
    b = ax.bar([v+.18 for v in x], maxima, .36, color='#9cb8c7', label='Maximum sampled utilization')
    for bars in (a, b):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.25,
                    '{:.2f}%'.format(bar.get_height()), ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in POLICIES])
    ax.set_ylabel('Whole-device GPU utilization (%)')
    ax.set_ylim(0, max(maxima)*1.3)
    ax.set_title('High workload: separate GPU-monitoring runs', loc='left', fontweight='bold')
    ax.legend(frameon=False)
    fig.text(.5, .025, 'One run per policy; 100 ms polling of a rolling device metric. These are separate from the 96 benchmark runs.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(str(args.output / ('slo_gpu_high.' + extension)), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Verified all four monitoring runs. Saved:', args.output)


if __name__ == '__main__':
    main()
