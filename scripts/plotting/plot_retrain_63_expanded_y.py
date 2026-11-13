"""Plot the 63 validation models with an explicitly piecewise y scale.

All eight panels retain every input coordinate.  The scale changes at 1.35;
the interval below that value gets more screen space.  No jitter is used.
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

from scripts.plotting.plot_retrain_63_publication import (
    ROOT, SOURCE, FAMILIES, SEEDS, COLORS, MARKERS, PANELS, load_data,
)


OUTPUT = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation/publication_figure_yzoom'
Y_MIN, Y_SPLIT, Y_MAX = .35, 1.35, 2.75
X_MIN, X_MAX = 85.8, 100.2


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_figure(data, meta, start, stop, stem):
    families = FAMILIES[start:stop]
    rows = len(families)
    fig = plt.figure(figsize=(9.2, rows * 2.8 + .9))
    grid = fig.add_gridspec(rows, 3, width_ratios=(1, 1, .31),
                            left=.08, right=.99, top=.88, bottom=.055 if rows == 4 else .085,
                            wspace=.16, hspace=.40)
    # A continuous, invertible mapping avoids clipping points near the split.
    # The two numeric segments have a displayed-height ratio of 2.6 : 1.
    upper_slope = (Y_SPLIT - Y_MIN) / (2.6 * (Y_MAX - Y_SPLIT))
    def forward(values):
        values = np.asarray(values)
        return np.where(values <= Y_SPLIT, values, Y_SPLIT + (values - Y_SPLIT) * upper_slope)
    def inverse(values):
        values = np.asarray(values)
        return np.where(values <= Y_SPLIT, values, Y_SPLIT + (values - Y_SPLIT) / upper_slope)
    plotted = []
    for r, (family, name) in enumerate(families):
        idx = start + r
        choices = sorted((m for m in meta if m['family'] == family), key=lambda m: m['value'])
        assert len(choices) == 5
        for c, region in enumerate(('internal', 'external')):
            axis = fig.add_subplot(grid[r, c])
            axis.set_yscale('function', functions=(forward, inverse))
            axis.set_ylim(Y_MIN, Y_MAX)
            axis.set_yticks([.4, .6, .8, 1., 1.2, 1.35, 2., 2.5])
            axis.set_yticklabels(['0.4', '0.6', '0.8', '1.0', '1.2', '1.35', '2.0', '2.5'])
            axis.set_xlim(X_MIN, X_MAX)
            axis.set_xticks([86, 90, 94, 98, 100])
            axis.set_title(f'({PANELS[idx * 2 + c]})  {name}  |  '
                           f'{"Interpolation" if c == 0 else "Extrapolation"}',
                           loc='left', fontsize=9, fontweight='bold', pad=6)
            axis.grid(color='#d4d4d4', linewidth=.45, alpha=.7, zorder=0)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.tick_params(labelsize=7.5, length=3, pad=2)
            axis.axhline(1, color='#777777', linewidth=.7, linestyle='--', zorder=1)
            axis.axhline(Y_SPLIT, color='#999999', linewidth=.65, linestyle=':', zorder=1)
            if c == 0:
                axis.set_ylabel('AvgRelRMSE', fontsize=9)
            if r == rows - 1:
                axis.set_xlabel('KCI(%)', fontsize=9)
            region_data = data[data.region.eq(region)]
            for item, color in [(m, color) for m, color in zip(choices, COLORS)] + [(dict(model='no_gain'), '#555555')]:
                for seed in SEEDS:
                    row = region_data[region_data.model.eq(item['model']) & region_data.seed.eq(seed)]
                    assert len(row) == 1
                    x, y = float(row.KCI_percent.iloc[0]), float(row.AvgRelRMSE.iloc[0])
                    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
                        raise ValueError(f'Point outside chart: {item["model"]}, {seed}, {region}, {x}, {y}')
                    axis.scatter(x, y, marker=MARKERS[seed], s=32 if item['model'] != 'no_gain' else 35,
                                 facecolor=color, edgecolor='white', linewidth=.5, zorder=4)
                    plotted.append((family, region, item['model'], seed, x, y))
        legend = fig.add_subplot(grid[r, 2]); legend.axis('off')
        legend.text(.03, .98, 'ρ' if family == 'difficult_gradnorm' else 'Loss weight',
                    fontsize=8, fontweight='bold', va='top', transform=legend.transAxes)
        for j, (item, color) in enumerate(zip(choices, COLORS)):
            yy = .79 - j * .135
            legend.plot([.14], [yy], color=color, marker='o', markersize=6, linestyle='', transform=legend.transAxes)
            legend.text(.28, yy, f"{item['value']:g}", fontsize=8, va='center', transform=legend.transAxes)
    handles = [Line2D([], [], marker=MARKERS[s], color='#555555', markersize=6, ls='',
                      markeredgecolor='white', markeredgewidth=.4, label=f'Seed {s}') for s in SEEDS]
    handles.append(Line2D([], [], marker='o', color='#555555', markersize=6, ls='',
                          markeredgecolor='white', markeredgewidth=.4, label='No-Gain reference'))
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(.50, .985), handletextpad=.4, columnspacing=1.3)
    fig.text(.5, .94, 'Nonlinear Y-axis: expanded below 1.35; no values omitted (all panels)',
             fontsize=8, ha='center', va='center')
    if len(plotted) != rows * 2 * 18:
        raise AssertionError('Incorrect number of plotted coordinates')
    for ext in ('png', 'pdf'):
        path = OUTPUT / f'{stem}.{ext}'
        fig.savefig(path, dpi=350, facecolor='white')
        print(path)
    plt.close(fig)
    return plotted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    names = [f'{name}.{ext}' for name in ('regional_63_models_expanded_y_4x2',
                                          'regional_63_models_expanded_y_part1',
                                          'regional_63_models_expanded_y_part2')
             for ext in ('png', 'pdf')]
    for name in names:
        if (OUTPUT/name).exists() and not args.overwrite:
            raise FileExistsError(f'Refusing to overwrite {OUTPUT/name}')
    data, meta = load_data()
    original_hash = sha(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42})
    all_plotted = plot_figure(data, meta, 0, 4, 'regional_63_models_expanded_y_4x2')
    plot_figure(data, meta, 0, 2, 'regional_63_models_expanded_y_part1')
    plot_figure(data, meta, 2, 4, 'regional_63_models_expanded_y_part2')
    # Each non-baseline point is drawn once, each baseline once per family.
    unique = {(seed, model, region, x, y) for _, region, model, seed, x, y in all_plotted}
    assert len(all_plotted) == 144 and len(unique) == len(data) == 126
    assert original_hash == sha(SOURCE)
    info = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=original_hash,
                unique_coordinates=126, visual_marks=144, jitter=False,
                y_axis=dict(lower=[Y_MIN, Y_SPLIT], upper=[Y_SPLIT, Y_MAX],
                            scale='piecewise; lower segment gets 2.6 times the display height of the upper segment',
                            no_data_interval_omitted=True),
                x_axis=[X_MIN, X_MAX], source_geometry='all model coordinates are plotted without moving',
                no_gain='Same three reference points are repeated across families.',
                outputs={n: sha(OUTPUT/n) for n in names})
    (OUTPUT/'figure_provenance.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (OUTPUT/'README.md').write_text(
        '# 63-model validation: expanded Y scale\n\n'
        'These figures retain all 126 distinct regional coordinates. The y axis is piecewise: '
        '0.35–1.35 is given 2.6 times the display height of 1.35–2.75. The ratio of display '
        'distance to numeric difference therefore changes at 1.35; the dotted reference line and figure '
        'heading disclose this. No numeric range is skipped and no points are jittered. All panels '
        'share the same x and y scales. No-Gain references are repeated across model families.\n\n'
        'KCI is agreement with an ANN reference, not a direct measurement of the Aspen gain. '
        'AvgRelRMSE uses each seed’s matching No-Gain as denominator; R5-5 partly crosses the '
        'core boundary. Full constant-scale figures remain in the adjacent publication_figure folder.\n\n'
        'Rebuild: `.\\.venv\\Scripts\\python.exe -m scripts.plotting.plot_retrain_63_expanded_y --overwrite`.\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
