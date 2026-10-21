"""Compare TS RMSE changes using individual training seeds and sample SD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SEEDS = [42, 43, 44]
INTERNAL = ['R5', 'R5-1', 'R5-2', 'R5-6']
EXTERNAL = [f'R5-{i}' for i in range(19, 27)]
TARGETS = ['B35_H2S', 'B35_SO2']
DISTRIBUTIONS = ['Internal', 'External']
STYLES = {42: ('#0072B2', 'o'), 43: ('#E69F00', '^'), 44: ('#009E73', 's')}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results/visualization/internal_external_ts_seed42_43_44')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protected = {}
    frames = []
    verified = 0
    for seed in SEEDS:
        source = root / f'results/visualization/four_model_rmse_visualization_seed{seed}/four_model_rmse_values.csv'
        protected[source.relative_to(root).as_posix()] = sha(source)
        data = pd.read_csv(source)
        data = data[(data.evaluation == 'TS') & data.model.isin(['Gain', 'Difficult-GradNorm'])]
        assert len(data) == 48 and not data.duplicated(['model', 'case', 'target']).any()
        for model, suffix in [('Gain', ''), ('Difficult-GradNorm', '_difficult_group_gradnorm_no_ss')]:
            experiment = f'transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed{seed}{suffix}'
            for case in INTERNAL + EXTERNAL:
                folder = case + ('_test_split' if case in INTERNAL else '')
                path = root / 'results' / experiment / folder / 'evaluation_metrics.csv'
                protected[path.relative_to(root).as_posix()] = sha(path)
                original = pd.read_csv(path).set_index('Variable')
                for target in TARGETS:
                    selected = data[(data.model == model) & (data['case'] == case) & (data.target == target)]
                    assert len(selected) == 1
                    np.testing.assert_allclose(selected.RMSE.iloc[0], original.loc[target, 'RMSE'], rtol=1e-12, atol=1e-16)
                    verified += 1
        wide = data.pivot(index=['case', 'target'], columns='model', values='RMSE').reset_index()
        assert set(wide['case']) == set(INTERNAL + EXTERNAL)
        assert not wide.isna().any().any()
        assert (wide[['Gain', 'Difficult-GradNorm']] > 0).all().all()
        wide['seed'] = seed
        wide['distribution'] = np.where(wide['case'].isin(INTERNAL), 'Internal', 'External')
        wide['change_percent'] = 100 * (wide['Difficult-GradNorm'] / wide.Gain - 1)
        wide['improved'] = wide.change_percent < 0
        frames.append(wide)
    cases = pd.concat(frames, ignore_index=True)
    per_seed = cases.groupby(['target', 'distribution', 'seed'], sort=False).agg(
        mean_change_percent=('change_percent', 'mean'),
        median_change_percent=('change_percent', 'median'),
        cases=('change_percent', 'size'),
        improved_cases=('improved', 'sum'),
    ).reset_index()
    summary = per_seed.groupby(['target', 'distribution'], sort=False).agg(
        mean_change_percent=('mean_change_percent', 'mean'),
        sd_change_percent=('mean_change_percent', 'std'),
        training_seeds=('seed', 'size'),
    ).reset_index()
    assert len(cases) == 72 and len(per_seed) == 12 and len(summary) == 4
    assert (summary.training_seeds == 3).all()
    assert per_seed.cases.eq(per_seed.distribution.map({'Internal': 4, 'External': 8})).all()
    for row in summary.itertuples(index=False):
        values = per_seed[(per_seed.target == row.target) & (per_seed.distribution == row.distribution)].mean_change_percent.to_numpy()
        np.testing.assert_allclose(row.sd_change_percent, np.std(values, ddof=1), rtol=1e-14)
    cases.to_csv(output / 'case_rmse_changes.csv', index=False)
    per_seed.to_csv(output / 'per_seed_rmse_changes.csv', index=False)
    summary.to_csv(output / 'mean_sd_rmse_changes.csv', index=False)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 10,
                         'axes.labelsize': 11, 'axes.titlesize': 13, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True)
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        axis.axhline(0, color='#666666', linewidth=1.2, linestyle='--', zorder=1)
        axis.text(0.025, 1.0, 'Gain reference', transform=axis.get_yaxis_transform(), fontsize=8, color='#666666', va='bottom')
        for x, distribution in enumerate(DISTRIBUTIONS):
            stat = summary[(summary.target == target) & (summary.distribution == distribution)].iloc[0]
            mean, sd = stat.mean_change_percent, stat.sd_change_percent
            axis.errorbar(x, mean, yerr=sd, fmt='D', color='#222222', markersize=7, markerfacecolor='white',
                          markeredgewidth=1.5, linewidth=1.7, capsize=7, capthick=1.6, zorder=3)
            axis.text(x, mean + sd + 3.0, f'({mean:+.2f} ± {sd:.2f})%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            for offset, seed in zip([-0.14, 0, 0.14], SEEDS):
                value = per_seed[(per_seed.target == target) & (per_seed.distribution == distribution) & (per_seed.seed == seed)].mean_change_percent.iloc[0]
                color, marker = STYLES[seed]
                axis.scatter(x + offset, value, s=57, color=color, marker=marker, edgecolors='white', linewidths=0.7, zorder=4)
        axis.set_title(f"{'AB'[panel]}  " + (r'H$_2$S' if target == 'B35_H2S' else r'SO$_2$'), loc='left', fontweight='bold')
        axis.set_xticks([0, 1], ['Internal\n4 test cases', 'External\n8 cases'])
        axis.set_xlim(-0.5, 1.5)
        axis.set_ylim(-42, 31)
        axis.set_yticks(np.arange(-40, 31, 10))
        axis.grid(axis='y', color='#DDDDDD', linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(axis='both', length=0, pad=7)
    axes[0].set_ylabel('RMSE change relative to Gain (%)')
    handles = [Line2D([], [], color='#222222', marker='D', markerfacecolor='white', markeredgewidth=1.5, linestyle='none', markersize=7, label='Mean ± SD')]
    handles += [Line2D([], [], color=color, marker=marker, linestyle='none', markersize=7, label=f'Seed {seed}') for seed, (color, marker) in STYLES.items()]
    fig.suptitle('Internal and external prediction performance', y=0.98, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.918, 'TS rolling prediction: Difficult-GradNorm relative to Gain', ha='center', fontsize=11)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.89), ncol=4, frameon=False, columnspacing=1.8)
    fig.subplots_adjust(top=0.76, bottom=0.25, left=0.09, right=0.98, wspace=0.16)
    fig.text(0.5, 0.105, 'Each seed: equal-weight mean of case-wise RMSE changes. Negative values indicate improvement.', ha='center', fontsize=9)
    fig.text(0.5, 0.057, 'Bars: sample SD across 3 training seeds (not a confidence interval). Colored points: individual seeds.', ha='center', fontsize=9)
    stem = 'internal_external_ts_mean_sd'
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'{stem}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    for name, digest in protected.items(): assert sha(root / name) == digest, name
    validation = {'passed': True, 'seeds': SEEDS, 'reference': 'Gain within the same training seed', 'evaluation': 'TS',
                  'aggregation': 'Mean of case-wise percent RMSE changes within each seed, then mean and sample SD across seeds.',
                  'sd_ddof': 1, 'uncertainty_unit': 'training seed', 'confidence_interval': False,
                  'canonical_rmse_values_verified': verified, 'case_changes': len(cases), 'seed_means': len(per_seed),
                  'summary_groups': len(summary), 'protected_sha256': protected, 'visual_review': 'pending',
                  'outputs_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.suffix in ['.png', '.pdf', '.csv']}}
    (output / 'validation_and_provenance.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'PASS: {verified} source RMSE values; sample SD ddof=1; {len(protected)} source hashes preserved.')
    print(f'Saved to {output}')


if __name__ == '__main__':
    main()
