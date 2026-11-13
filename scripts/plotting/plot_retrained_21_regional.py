"""Make a numbered regional KCI/RMSE figure from a completed evaluation CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'results/visualization/retrain_lhs10_23_21_models_validation_seed42'
FAMILIES = {
    'no_gain': ('No-Gain', '#303030', 'P'),
    'gain': ('Gain', '#0072B2', 'o'),
    'ss_only': ('SS-only', '#D55E00', 's'),
    'gain_ss': ('Gain+SS', '#009E73', '^'),
    'difficult_gradnorm': ('Difficult-GradNorm', '#7B4AB4', 'D'),
}


def load_data(path: Path):
    frame = pd.read_csv(path)
    required = {'model', 'family', 'partition', 'AvgRelRMSE', 'ANN_reference_KCI_percent'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(frame.columns))}')
    manifest = json.loads((ROOT/'configs/pgin/retrain_lhs10_23/config_manifest.json').read_text(encoding='utf-8'))
    jobs = [x for x in manifest if x['seed'] == 42]
    mapping = {j['alias']: j['family'] for j in jobs}
    if len(frame) != 42 or frame.duplicated(['model', 'partition']).any():
        raise ValueError('Expected exactly one coordinate per model and region (42 rows).')
    if set(frame.model) != set(mapping) or set(frame.partition) != {'internal', 'external'}:
        raise ValueError('Input is not the complete seed-42 21-model evaluation.')
    if not frame.groupby('model').partition.nunique().eq(2).all():
        raise ValueError('Each model must have both regions.')
    if not frame.family.eq(frame.model.map(mapping)).all():
        raise ValueError('Model families differ from the seed-42 manifest.')
    vals = frame[['AvgRelRMSE', 'ANN_reference_KCI_percent']].to_numpy(float)
    if not np.isfinite(vals).all() or not (vals[:, 0] > 0).all() or not ((vals[:, 1] >= 0) & (vals[:, 1] <= 100)).all():
        raise ValueError('Coordinates must have finite positive RMSE and KCI in [0,100].')
    if not np.allclose(frame.loc[frame.model.eq('no_gain'), 'AvgRelRMSE'], 1, atol=1e-12):
        raise ValueError('No-Gain must be the same-seed baseline.')
    return frame, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_DIR/'regional_plot_coordinates.csv')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--overwrite', action='store_true', help='Replace this script\'s numbered PNG and PDF.')
    args = parser.parse_args()
    frame, jobs = load_data(args.input)
    for ext in ['png', 'pdf']:
        path = args.output_dir/f'regional_kci_vs_avgrelrmse_numbered.{ext}'
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'Refusing to overwrite {path}')
    identifiers = {j['alias']: i for i, j in enumerate(jobs, 1)}
    fig = plt.figure(figsize=(17.8, 10.5))
    axes = [fig.add_axes([.065, .425, .395, .46]), fig.add_axes([.555, .425, .395, .46])]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    for ax, part, title in zip(axes, ['internal', 'external'], ['Interpolation group', 'Extrapolation group']):
        region = frame[frame.partition.eq(part)].copy()
        for family, (_, color, marker) in FAMILIES.items():
            rows = region[region.family.eq(family)]
            ax.scatter(rows.ANN_reference_KCI_percent, rows.AvgRelRMSE, marker=marker,
                       c=color, s=50, edgecolors='white', linewidths=.6, zorder=3)
        # The numeric labels occupy fixed, evenly spaced slots in a dedicated
        # strip at the right of each axes; leader lines show their exact points.
        region = region.sort_values('AvgRelRMSE', ascending=False).reset_index(drop=True)
        for i, row in region.iterrows():
            _, color, _ = FAMILIES[row.family]
            ax.annotate(str(identifiers[row.model]),
                        xy=(row.ANN_reference_KCI_percent, row.AvgRelRMSE), xycoords='data',
                        xytext=(1.025, 1-(i+.5)/len(region)), textcoords='axes fraction',
                        color=color, fontsize=8.5, va='center', ha='left',
                        arrowprops={'arrowstyle':'-', 'color':color, 'lw':.45, 'alpha':.55},
                        annotation_clip=False)
        ax.axhline(1, color='#696969', ls='--', lw=1, zorder=1)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=11)
        ax.set_xlabel('ANN-reference KCI (%)', fontsize=10)
        ax.grid(color='#d0d0d0', alpha=.5, linewidth=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.margins(x=.12, y=.12)
    axes[0].set_ylabel('AvgRelRMSE (No-Gain = 1)', fontsize=10)
    handles = [Line2D([], [], marker=mark, color=color, ls='', markersize=7, label=label)
               for label,color,mark in FAMILIES.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.94), ncol=5,
               frameon=False, fontsize=9)
    fig.suptitle('21 retrained models | seed 42 validation', fontsize=15, fontweight='bold', y=.986)
    fig.text(.065, .36, 'Model number key', fontsize=10, fontweight='bold')
    for i,j in enumerate(jobs):
        col,row = divmod(i,7)
        name = j['alias']
        color = FAMILIES[j['family']][1]
        family = j['family']
        if family == 'no_gain':
            label = 'No-Gain'
        elif family == 'difficult_gradnorm':
            cli = j['cli_args']
            label = f"Difficult-GradNorm  rho={float(cli[cli.index('--target-ratio')+1]):g}"
        else:
            cfg = yaml.safe_load((ROOT/j['config']).read_text(encoding='utf-8-sig'))
            key = 'ss_loss_weight' if family == 'ss_only' else 'gain_loss_weight'
            label = f"{FAMILIES[family][0]}  weight={float(cfg['training'][key]):g}"
        fig.text(.065+col*.31, .337-row*.022, f"{identifiers[name]:02d}  {label}",
                 color=color, fontsize=9, va='top', family='DejaVu Sans Mono')
    fig.text(.065, .095,
             'Y: geometric mean of 18 same-case RMSE / No-Gain RMSE ratios per region (H1, H18, TS; H2S, SO2).\n'
             'X: separate steady-state probes, pooled ANN-reference gain sign agreement; not an Aspen or closed-loop metric.\n'
             'Left: R5-3/4/5 and in-range probes; R5-5 partly crosses core bounds. Right: R5-27/28/29 and soft-OOD probes.\n'
             'Axis limits differ between panels. Single training seed; no uncertainty across seeds shown.', fontsize=8.5, va='top', linespacing=1.35)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = args.output_dir/f'regional_kci_vs_avgrelrmse_numbered.{ext}'
        fig.savefig(path, dpi=250, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
