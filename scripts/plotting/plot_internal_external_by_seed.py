"""Plot paired Internal/External TS RMSE reductions for individual seeds."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'results/visualization/internal_external_ts_seed42_43_44'
    provenance = json.loads((folder / 'validation_and_provenance.json').read_text(encoding='utf-8'))
    source = folder / 'per_seed_rmse_changes.csv'
    protected = {**provenance['protected_sha256'], source.relative_to(root).as_posix(): sha(source)}
    for extension in ['png', 'pdf']:
        p = folder / f'internal_external_ts_mean_sd.{extension}'
        protected[p.relative_to(root).as_posix()] = sha(p)
    for name, digest in protected.items():
        assert sha(root / name) == digest, name
    data = pd.read_csv(source)
    assert len(data) == 12 and not data.duplicated(['target', 'distribution', 'seed']).any()
    # Reverse the earlier change metric so positive values consistently mean improvement.
    data['rmse_reduction_percent'] = -data.mean_change_percent
    for seed in [42, 43, 44]:
        original = pd.read_csv(root / f'results/visualization/four_model_rmse_visualization_seed{seed}/four_model_rmse_values.csv')
        original = original[(original.evaluation == 'TS') & original.model.isin(['Gain', 'Difficult-GradNorm'])]
        wide = original.pivot(index=['case', 'target'], columns='model', values='RMSE').reset_index()
        wide['distribution'] = np.where(wide['case'].isin(['R5', 'R5-1', 'R5-2', 'R5-6']), 'Internal', 'External')
        wide['reduction'] = 100 * (1 - wide['Difficult-GradNorm'] / wide.Gain)
        for row in data[data.seed == seed].itertuples(index=False):
            values = wide[(wide.target == row.target) & (wide.distribution == row.distribution)].reduction
            assert len(values) == row.cases
            np.testing.assert_allclose(values.mean(), row.rmse_reduction_percent, rtol=1e-13, atol=1e-13)
    data.to_csv(folder / 'by_seed_rmse_reductions.csv', index=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 10,
                         'axes.labelsize': 11, 'axes.titlesize': 13, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True)
    for index, (axis, target) in enumerate(zip(axes, ['B35_H2S', 'B35_SO2'])):
        for distribution, offset, color, hatch in [('Internal', -0.18, '#0072B2', '//'), ('External', 0.18, '#E69F00', '')]:
            values = data[(data.target == target) & (data.distribution == distribution)].set_index('seed').loc[[42, 43, 44], 'rmse_reduction_percent'].to_numpy()
            bars = axis.bar(np.arange(3) + offset, values, width=0.31, color=color, edgecolor='white', linewidth=0.7, hatch=hatch, label=distribution, zorder=3)
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width()/2, value + (1.2 if value >= 0 else -1.2), f'{value:+.2f}%',
                          ha='center', va='bottom' if value >= 0 else 'top', fontsize=9.5, fontweight='bold')
        axis.axhline(0, color='#444444', linewidth=1.2, zorder=4)
        axis.set_title(f"{'AB'[index]}  " + (r'H$_2$S' if target == 'B35_H2S' else r'SO$_2$'), loc='left', fontweight='bold')
        axis.set_xticks([0, 1, 2], ['Seed 42', 'Seed 43', 'Seed 44'])
        axis.set_xlim(-0.6, 2.6)
        axis.set_ylim(-23, 39)
        axis.set_yticks(np.arange(-20, 40, 10))
        axis.grid(axis='y', color='#DDDDDD', linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(length=0, pad=7)
    axes[0].set_ylabel('RMSE reduction relative to Gain (%)')
    figure.suptitle('Internal vs external prediction performance', fontsize=15, fontweight='bold', y=0.99)
    figure.text(0.5, 0.923, 'TS rolling prediction: Difficult-GradNorm relative to Gain', ha='center', fontsize=11)
    figure.text(0.5, 0.868, 'Positive = improvement     Negative = worse', ha='center', fontsize=10, color='#444444')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.835), ncol=2, frameon=False)
    figure.subplots_adjust(top=0.72, bottom=0.18, left=0.09, right=0.99, wspace=0.15)
    figure.text(0.5, 0.055, 'Each bar: case-wise RMSE reductions averaged within one seed (4 internal test cases; 8 external cases).', ha='center', fontsize=9)
    stem = 'internal_external_ts_by_seed'
    for extension in ['png', 'pdf']:
        figure.savefig(folder / f'{stem}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    for name, digest in protected.items():
        assert sha(root / name) == digest, name
    output = {'passed': True, 'metric': '100 * (1 - Difficult-GradNorm RMSE / Gain RMSE)', 'positive_values': 'improvement',
              'aggregation': 'Equal-weight case mean within each individual seed', 'error_bars': False, 'seeds': [42, 43, 44],
              'bars_verified': 12, 'protected_sha256': protected, 'visual_review': 'pending',
              'outputs_sha256': {p.name: sha(p) for p in [folder / f'{stem}.png', folder / f'{stem}.pdf', folder / 'by_seed_rmse_reductions.csv']}}
    (folder / 'by_seed_validation_and_provenance.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('PASS: 12 bars reconciled with original source CSVs; previous mean/SD figures preserved.')
    print(folder / f'{stem}.png')


if __name__ == '__main__':
    main()
