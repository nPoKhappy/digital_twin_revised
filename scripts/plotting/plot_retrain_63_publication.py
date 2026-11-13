"""Plot all 63 retrained models as regional, family-separated scatter plots.

One marker is one trained seed. No coordinate is jittered or averaged.
"""
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

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from scripts.plotting.retrain_parameter_ui import model_parameters

SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation/regional_coordinates_63_models.csv'
OUTPUT = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation/publication_figure'
FAMILIES = [('gain', 'Gain'), ('ss_only', 'SS-only'),
            ('gain_ss', 'Gain + SS'), ('difficult_gradnorm', 'Difficult-GradNorm')]
SEEDS = (42, 43, 44)
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
MARKERS = {42: 'o', 43: 's', 44: '^'}
PANELS = 'abcdefgh'


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    data = pd.read_csv(SOURCE)
    meta = model_parameters(ROOT)
    models = {x['model']: x for x in meta}
    expected = {(seed, name, region) for seed in SEEDS for name in models for region in ('internal', 'external')}
    actual = list(zip(data.seed, data.model, data.region))
    if len(data) != 126 or len(set(actual)) != 126 or set(actual) != expected:
        raise ValueError('Expected 63 models and one coordinate for each region.')
    if not data.family.eq(data.model.map(lambda x: models[x]['family'])).all():
        raise ValueError('Family does not match manifest and configuration.')
    if not (np.isfinite(data[['KCI_percent', 'AvgRelRMSE']]).all().all()
            and data.KCI_percent.between(0, 100).all() and data.AvgRelRMSE.gt(0).all()):
        raise ValueError('Nonfinite or impossible coordinate.')
    if not np.allclose(data.KCI_percent, 100 * data.correct / data.valid, rtol=1e-12):
        raise ValueError('KCI does not match its count.')
    if not np.allclose(data.loc[data.model.eq('no_gain'), 'AvgRelRMSE'], 1):
        raise ValueError('No-Gain must be the within-seed reference.')
    for seed in SEEDS:
        directory = ROOT / f'results/visualization/retrain_lhs10_23_21_models_validation_seed{seed}'
        report = json.loads((directory/'evaluation_validation.json').read_text(encoding='utf-8'))
        if not report.get('passed') or report.get('seed') != seed or report.get('model_count') != 21:
            raise ValueError(f'Seed {seed} did not complete validation.')
        original = pd.read_csv(directory/'regional_plot_coordinates.csv').rename(columns={
            'partition':'region', 'ANN_reference_KCI_percent':'KCI_percent'})
        comparison = data[data.seed.eq(seed)].merge(original, on=['model','family','region'],
                                                     suffixes=('_combined', '_original'), validate='one_to_one')
        if len(comparison) != 42:
            raise ValueError(f'Seed {seed} missing coordinates.')
        for col in ('KCI_percent', 'AvgRelRMSE', 'correct', 'valid'):
            if not np.allclose(comparison[f'{col}_combined'], comparison[f'{col}_original'], rtol=1e-12, atol=1e-12):
                raise ValueError(f'Seed {seed} {col} differs from evaluation.')
    return data, meta


def row_limits(data, family):
    vals = data['AvgRelRMSE'] if family is None else data.loc[data.family.isin([family, 'no_gain']), 'AvgRelRMSE']
    lower, upper = float(vals.min()), float(vals.max())
    margin = max(.03, (upper - lower) * .15)
    return max(0, lower - margin), upper + margin


def draw(data, meta, start, stop, stem):
    rows = FAMILIES[start:stop]
    nr = len(rows)
    fig = plt.figure(figsize=(9.2, 2.35 * nr + .9))
    gs = fig.add_gridspec(nr, 3, width_ratios=(1, 1, .31),
                          left=.08, right=.99, top=.91, bottom=.075 if nr == 4 else .12,
                          wspace=.16, hspace=.46)
    ylim = row_limits(data, None)
    for r, (family, name) in enumerate(rows):
        i = start + r
        family_models = sorted([m for m in meta if m['family'] == family], key=lambda m: m['value'])
        if len(family_models) != 5:
            raise ValueError(f'Expected five parameter settings in {family}.')
        for c, region in enumerate(('internal', 'external')):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xlim(85.8, 100.2)
            ax.set_ylim(*ylim)
            ax.set_xticks([86, 90, 94, 98, 100])
            ax.set_title(f'({PANELS[i * 2 + c]})  {name}  |  {"Interpolation" if c == 0 else "Extrapolation"}',
                         loc='left', fontsize=9, fontweight='bold', pad=6)
            ax.axhline(1, color='#777777', linewidth=.7, linestyle='--', zorder=1)
            ax.grid(color='#d4d4d4', linewidth=.45, alpha=.7, zorder=0)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            ax.tick_params(labelsize=7.5, length=3, pad=2)
            if c == 0:
                ax.set_ylabel('AvgRelRMSE', fontsize=9)
            if r == nr - 1:
                ax.set_xlabel('KCI(%)', fontsize=9)
            region_data = data[data.region.eq(region)]
            for m, color in zip(family_models, COLORS):
                for seed in SEEDS:
                    point = region_data[region_data.model.eq(m['model']) & region_data.seed.eq(seed)]
                    assert len(point) == 1
                    ax.scatter(point.KCI_percent, point.AvgRelRMSE, marker=MARKERS[seed],
                               s=32, facecolor=color, edgecolor='white', linewidth=.5, zorder=3)
            for seed in SEEDS:
                point = region_data[region_data.model.eq('no_gain') & region_data.seed.eq(seed)]
                assert len(point) == 1
                ax.scatter(point.KCI_percent, point.AvgRelRMSE, marker=MARKERS[seed],
                           s=35, facecolor='#555555', edgecolor='white', linewidth=.5, zorder=4)
        lax = fig.add_subplot(gs[r, 2]); lax.axis('off')
        label = 'ρ' if family == 'difficult_gradnorm' else 'Loss weight'
        lax.text(0.03, .98, label, fontsize=8, fontweight='bold', va='top', transform=lax.transAxes)
        for j, (m, color) in enumerate(zip(family_models, COLORS)):
            yy = .79 - j * .135
            lax.plot([.14], [yy], color=color, marker='o', markersize=6, linestyle='', transform=lax.transAxes)
            lax.text(.28, yy, f"{m['value']:g}", fontsize=8, va='center', transform=lax.transAxes)
    handles = [Line2D([], [], marker=MARKERS[s], color='#555555', markersize=6, ls='',
                      markeredgecolor='white', markeredgewidth=.4, label=f'Seed {s}') for s in SEEDS]
    handles.append(Line2D([], [], marker='o', color='#555555', markersize=6, ls='',
                          markeredgecolor='white', markeredgewidth=.4, label='No-Gain reference'))
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(.50, .982), handletextpad=.4, columnspacing=1.3)
    for ext in ('png', 'pdf'):
        path = OUTPUT/f'{stem}.{ext}'
        fig.savefig(path, dpi=350, facecolor='white')
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Replace figure files generated by this script.')
    args = parser.parse_args()
    names = [f'{stem}.{ext}' for stem in ('regional_63_models_4x2', 'regional_63_models_4x2_common_y', 'regional_63_models_part1', 'regional_63_models_part2') for ext in ('png', 'pdf')]
    for name in names:
        if (OUTPUT/name).exists() and not args.overwrite:
            raise FileExistsError(f'Refusing to overwrite {OUTPUT/name}')
    data, meta = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42})
    draw(data, meta, 0, 4, 'regional_63_models_4x2')
    draw(data, meta, 0, 4, 'regional_63_models_4x2_common_y')
    draw(data, meta, 0, 2, 'regional_63_models_part1')
    draw(data, meta, 2, 4, 'regional_63_models_part2')
    provenance = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=checksum(SOURCE),
                      distinct_models=63, regions=2, unique_coordinates=126, seed_values=list(SEEDS),
                      layout='rows are model families; columns are internal and external validation',
                      markers='seed', colors='parameter, independently keyed in each row',
                      x_axes='All panels and split figures share identical x limits; label is KCI(%)',
                      y_axes='All panels and split figures share identical y limits',
                      no_gain='3 references in each region repeated for each family, not new models',
                      jitter=False, per_region_baseline='same-seed No-Gain',
                      figure_sha256={n:checksum(OUTPUT/n) for n in names})
    (OUTPUT/'figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
