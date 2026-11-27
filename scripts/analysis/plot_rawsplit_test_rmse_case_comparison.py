"""Plot casewise test RMSE changes against the rawsplit No-Gain seed 42.

Every cell summarizes the three training seeds for a single case, output,
prediction horizon and model family. Separate images show each region/output.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_rawsplit'
DEST = ROOT / 'results/evaluation/rawsplit_test_rmse_vs_nogain42/figures'
FAMILIES = ['no_gain', 'gain', 'ss_only', 'gain_ss', 'difficult_gradnorm']
HORIZONS = ['H1', 'H18', 'TS']
CASES = {'internal': ['R5', 'R5-1', 'R5-2', 'R5-6'],
         'external': [f'R5-{i}' for i in range(19, 27)]}
TARGETS = {'B35_H2S': 'H₂S', 'B35_SO2': 'SO₂'}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    audit_path = SOURCE / 'independent_npz_audit.json'
    source_path = SOURCE / 'relative_rmse_components.csv'
    selected_path = SOURCE / 'selected_models.json'
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    if not audit['passed'] or audit['source_sha256']['relative_rmse_components.csv'] != sha(source_path):
        raise ValueError('Test source is not covered by the independent NPZ audit')
    if audit['source_sha256']['selected_models.json'] != sha(selected_path):
        raise ValueError('Selected models differ from the independent NPZ audit')
    selection = json.loads(selected_path.read_text(encoding='utf8'))['selected_models']
    labels = {}
    for family in FAMILIES:
        rows = [r for r in selection if r['family'] == family]
        if len(rows) != 3 or {r['seed'] for r in rows} != {42, 43, 44} or len({r['alias'] for r in rows}) != 1:
            raise ValueError(f'Unexpected model family selection: {family}')
        configs = [yaml.safe_load((ROOT / row['config']).read_text(encoding='utf8'))['training'] for row in rows]
        if family == 'no_gain': label = 'No-Gain'
        elif family == 'gain': label = f"Gain\n{float(configs[0]['gain_loss_weight']):g}"
        elif family == 'ss_only': label = f"SS-only\n{float(configs[0]['ss_loss_weight']):g}"
        elif family == 'gain_ss': label = f"Gain+SS\n{float(configs[0]['gain_loss_weight']):g}/{float(configs[0]['ss_loss_weight']):g}"
        else: label = f"DG\nρ={float(configs[0]['dg_target_ratio']):g}"
        labels[family] = label
    d = pd.read_csv(source_path)
    keys = ['partition', 'case', 'target', 'evaluation', 'family', 'seed']
    if len(d) != 1080 or d.duplicated(keys).any() or not np.isfinite(d['relative_RMSE']).all():
        raise ValueError('Incomplete or invalid casewise RMSE data')
    if not np.allclose(d['RMSE'] / d['NoGain42_RMSE'], d['relative_RMSE'], rtol=1e-11, atol=1e-12):
        raise ValueError('RMSE ratios do not match the fixed No-Gain seed 42 baseline')
    if not np.allclose(d.loc[(d.family == 'no_gain') & (d.seed == 42), 'relative_RMSE'], 1.0):
        raise ValueError('No-Gain seed 42 is not its own baseline')
    d['delta_percent'] = 100.0 * (d['relative_RMSE'] - 1.0)
    group = d.groupby(keys[:-1], as_index=False).agg(
        mean_delta_percent=('delta_percent', 'mean'),
        sd_delta_percentage_points=('delta_percent', 'std'),
        seeds=('seed', 'nunique'), mean_rmse=('RMSE', 'mean'),
        baseline_rmse=('NoGain42_RMSE', 'first'))
    if len(group) != 360 or not group.seeds.eq(3).all() or group.isna().any().any():
        raise ValueError('Incomplete three-seed grouped coverage')
    if group.mean_delta_percent.min() < -100 or group.mean_delta_percent.max() > 250:
        raise ValueError('Prespecified common figure scale does not contain all means')
    DEST.mkdir(parents=True, exist_ok=False)
    group.to_csv(DEST / 'figure_values.csv', index=False, float_format='%.17g')

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.unicode_minus': False})
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=250)
    cmap = plt.get_cmap('RdBu_r')
    for region, cases in CASES.items():
        for target, target_label in TARGETS.items():
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1 if region == 'internal' else 5.6), sharey=True)
            image = None
            for i, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
                sub = group.query('partition == @region and target == @target and evaluation == @horizon')
                mean = sub.pivot(index='case', columns='family', values='mean_delta_percent').reindex(index=cases, columns=FAMILIES)
                sd = sub.pivot(index='case', columns='family', values='sd_delta_percentage_points').reindex(index=cases, columns=FAMILIES)
                if mean.shape != (len(cases), 5) or mean.isna().any().any() or sd.isna().any().any():
                    raise ValueError(f'Incomplete figure panel: {region} {target} {horizon}')
                image = ax.imshow(mean.to_numpy(), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
                for y in range(len(cases)):
                    for x in range(5):
                        value, spread = float(mean.iat[y, x]), float(sd.iat[y, x])
                        rgba = cmap(norm(value))
                        luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                        ax.text(x, y, f'{value:+.1f}\n±{spread:.1f}', ha='center', va='center',
                                fontsize=7.8, color='white' if luminance < .49 else '#202020')
                ax.set_title(f'({chr(97+i)}) {horizon}', loc='left', fontsize=10, fontweight='bold', pad=8)
                ax.set_xticks(range(5), [labels[f] for f in FAMILIES], fontsize=8)
                ax.set_yticks(range(len(cases)), cases, fontsize=8)
                ax.tick_params(length=0)
                ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
                ax.set_yticks(np.arange(-.5, len(cases), 1), minor=True)
                ax.grid(which='minor', color='white', linestyle='-', linewidth=.7)
                ax.tick_params(which='minor', bottom=False, left=False)
                ax.set_xlabel('Model', fontsize=9)
                if i == 0: ax.set_ylabel('Case', fontsize=9)
            fig.subplots_adjust(left=.07, right=.88, bottom=.20, top=.90, wspace=.14)
            cax = fig.add_axes([.91, .23, .012, .56])
            cb = fig.colorbar(image, cax=cax, ticks=[-100, -50, 0, 50, 100, 150, 200, 250])
            cb.set_label('RMSE change versus No-Gain seed 42 (%)', fontsize=9)
            cb.ax.tick_params(labelsize=8)
            # Portable filenames use ASCII while chemical labels stay in captions.
            stem = f"rmse_change_{region}_{'h2s' if target == 'B35_H2S' else 'so2'}"
            fig.savefig(DEST / f'{stem}.png', dpi=300, facecolor='white')
            fig.savefig(DEST / f'{stem}.pdf', facecolor='white')
            plt.close(fig)
            caption = (f'{region.capitalize()} test, {target_label}. Each cell shows the arithmetic mean ± sample standard deviation '
                       'across three independently trained dynamic-model seeds (42, 43, 44) of 100 × (model RMSE / '
                       'same-case, same-output, same-horizon No-Gain seed 42 RMSE − 1). Negative values indicate a '
                       'lower RMSE. No-Gain cells also average its three seeds and need not equal zero. '
                       'H1 and H18 are the first and eighteenth rolling forecast horizons; TS is the '
                       'autoregressive time-series prediction. Colors use identical limits across figures '
                       '(−100% to +250%); blanks are not used. '
                       'The retained test cases were used previously and are not a new blind holdout.\n')
            (DEST / f'{stem}_caption.txt').write_text(caption, encoding='utf8')
    outputs = {p.name: sha(p) for p in DEST.iterdir() if p.is_file()}
    (DEST / 'provenance.json').write_text(json.dumps({'passed': True, 'source_relative_rmse_sha256': sha(source_path),
        'source_selection_sha256': sha(selected_path), 'source_audit_sha256': sha(audit_path),
        'figure_cells': len(group), 'seeds': [42, 43, 44], 'color_scale_percent': [-100, 0, 250],
        'outputs_sha256': outputs}, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print(f'Wrote 4 figures and 360 cells to {DEST}')


if __name__ == '__main__':
    main()
