"""Redraw frozen validation coordinates for manuscript-sized figures.

Does not load checkpoints, run inference or modify source evaluation data.
"""
from __future__ import annotations

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
BASE = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'
OUT = BASE / 'jtice_figures'
FAMILIES = [('gain', 'Gain'), ('ss_only', 'SS-only'), ('gain_ss', 'Gain + SS'),
            ('difficult_gradnorm', 'Difficult-GradNorm')]
SEEDS = (42, 43, 44)
MARKERS = {42: 'o', 43: 's', 44: '^'}
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def draw(data, meta, families, stem, expanded, bounds):
    rows = len(families)
    # 190 mm wide: Elsevier general starting width, not a JTICE-specific claim.
    fig = plt.figure(figsize=(190 / 25.4, (210 if rows == 4 else 120) / 25.4))
    gs = fig.add_gridspec(rows, 3, width_ratios=[1, 1, .28], left=.09, right=.995,
                          bottom=.07, top=.94, hspace=.48, wspace=.19)
    lower, upper = bounds['y']
    split = 1.35
    slope = (split - lower) / (2.6 * (upper - split))
    def fwd(v):
        v = np.asarray(v)
        return np.where(v <= split, v, split + (v - split) * slope)
    def inv(v):
        v = np.asarray(v)
        return np.where(v <= split, v, split + (v - split) / slope)
    plotted = []
    for row, (family, label) in enumerate(families):
        choices = meta[meta.family.eq(family)].sort_values('value').to_dict('records')
        if len(choices) != 5:
            raise ValueError(f'Expected 5 settings for {family}')
        for col, region in enumerate(('internal', 'external')):
            ax = fig.add_subplot(gs[row, col])
            ax.set_xlim(*bounds['x'])
            ax.set_ylim(*bounds['y'])
            ax.set_xticks([86, 90, 94, 98, 100])
            if expanded:
                ax.set_yscale('function', functions=(fwd, inv))
                ax.set_ylim(*bounds['y'])
                ticks = [v for v in (.4, .6, .8, 1., 1.2, 1.35, 2.1, 2.9, 3.7)
                         if lower <= v <= upper]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f'{v:g}' for v in ticks])
                ax.axhline(split, color='.7', lw=.45, ls=':')
            ax.axhline(1., color='.52', ls='--', lw=.6, zorder=1)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7, width=.6, length=2.5, pad=2)
            # Light horizontal guides preserve readability in dense comparisons.
            ax.grid(axis='y', color='.9', lw=.4, zorder=0)
            idx = FAMILIES.index((family, label))*2 + col
            ax.set_title(f'({chr(97+idx)}) {label} | '+('Interpolation' if col == 0 else 'Extrapolation'),
                         loc='left', fontsize=7, fontweight='bold', pad=4)
            if col == 0:
                ax.set_ylabel('AvgRelRMSE\n(nonlinear)' if expanded else 'AvgRelRMSE', fontsize=7)
            if row == rows-1:
                ax.set_xlabel('KCI (%)', fontsize=7)
            subset = data[data.region.eq(region)]
            for item, color in [(m, c) for m, c in zip(choices, COLORS)] + [({'model': 'no_gain'}, '#303030')]:
                for seed in SEEDS:
                    point = subset[subset.model.eq(item['model']) & subset.seed.eq(seed)]
                    if len(point) != 1:
                        raise ValueError(f'Missing or duplicate point: {item} {region} {seed}')
                    x, y = float(point.KCI_percent.iloc[0]), float(point.AvgRelRMSE.iloc[0])
                    if not (bounds['x'][0] <= x <= bounds['x'][1] and lower <= y <= upper):
                        raise ValueError(f'Out of range: {item} {region} {seed}')
                    ax.scatter(x, y, s=23, marker=MARKERS[seed], facecolor=color,
                               edgecolor='white', linewidth=.45, zorder=4)
                    plotted.append((int(seed), item['model'], region, x, y))
        key = fig.add_subplot(gs[row, 2]); key.axis('off')
        key.text(0., .98, r'$\rho$' if family == 'difficult_gradnorm' else 'Weight',
                 fontsize=7, fontweight='bold', va='top', transform=key.transAxes)
        for i, (entry, color) in enumerate(zip(choices, COLORS)):
            y = .81 - i*.155
            key.plot([.12], [y], marker='o', color=color, ms=4.5, linestyle='', transform=key.transAxes)
            key.text(.29, y, f"{entry['value']:g}", fontsize=7, va='center', transform=key.transAxes)
    handles = [Line2D([], [], color='.25', marker=MARKERS[s], ls='', markersize=4.5,
                      label=f'Seed {s}') for s in SEEDS]
    handles += [Line2D([], [], color='.25', marker='o', ls='', markersize=4.5, label='No-Gain')]
    fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(.5, .995),
               frameon=False, fontsize=7, columnspacing=1.4, handletextpad=.4)
    for fmt in ('pdf', 'png'):
        fig.savefig(OUT/f'{stem}.{fmt}', dpi=300, facecolor='white')
    plt.close(fig)
    return plotted


def main():
    coords_path = BASE / 'regional_coordinates_63_models.csv'
    meta_path = BASE / 'model_parameter_metadata.csv'
    prov_path = BASE / 'provenance.json'
    prov = json.loads(prov_path.read_text(encoding='utf-8'))
    for path in (coords_path, meta_path):
        if sha(path) != prov['outputs_sha256'][path.name]:
            raise ValueError(f'Original provenance mismatch: {path}')
    original = {str(p.relative_to(ROOT)): sha(p) for p in (coords_path, meta_path, prov_path)}
    coords, meta = pd.read_csv(coords_path), pd.read_csv(meta_path)
    if len(coords) != 126 or coords[['model', 'seed', 'region']].duplicated().any():
        raise ValueError('Expected 126 distinct model/seed/region points')
    if set(coords.seed) != set(SEEDS) or len(meta) != 21:
        raise ValueError('Unexpected model or seed roster')
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    bounds = prov['plot_bounds']
    outputs = {}
    for expanded in (True, False):
        scale = 'expanded_y' if expanded else 'linear'
        for subset, label in ((FAMILIES, 'all_families'), (FAMILIES[:2], 'part1'), (FAMILIES[2:], 'part2')):
            marks = draw(coords, meta, subset, f'validation_{label}_{scale}', expanded, bounds)
            expected = {(int(r.seed), r.model, r.region, float(r.KCI_percent), float(r.AvgRelRMSE))
                        for r in coords.itertuples(index=False)
                        if r.family in {f[0] for f in subset} or r.model == 'no_gain'}
            if set(marks) != expected or len(marks) != len(subset)*2*18:
                raise ValueError(f'Plotted coordinates mismatch: {label}')
            outputs[f'{label}_{scale}'] = {'drawn_markers': len(marks), 'unique_model_region_coordinates': len(set(marks))}
            family_names = ', '.join(name for _, name in subset)
            scale_description = ('A shared piecewise-linear y axis changes slope at AvgRelRMSE = 1.35; '
                                 'all plotted values are retained.' if expanded else
                                 'All panels use a shared linear y axis.')
            caption = (f'Validation comparison of {family_names}. Each model was evaluated in interpolation '
                       'and extrapolation. Marker shapes indicate training seeds 42, 43 and 44; colours indicate '
                       'the weight or target ratio. The No-Gain seeds are repeated as references in '
                       'each family panel. AvgRelRMSE is normalized to No-Gain seed 42 in the corresponding '
                       'region; KCI is directional agreement with the shared ANN reference. '
                       f'{scale_description}')
            (OUT / f'validation_{label}_{scale}_caption.txt').write_text(caption + '\n', encoding='utf-8')
    for name, digest in original.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Source changed during draw: {name}')
    notes = OUT/'figure_notes.md'
    notes.write_text('''# Validation figure notes\n\nJTICE guide snapshot supplied by the user; verify current journal instructions at submission. Each figure has a separate caption text file. PDFs are vector figures with embedded fonts; PNG files are previews.\n\nAll figures show the existing 63 model runs across interpolation and extrapolation (126 distinct model-region coordinates). The three No-Gain runs are repeated in each family panel as a reference. Marker shape identifies training seed; colour identifies the loss weight or target ratio in the corresponding row. AvgRelRMSE uses No-Gain seed 42 as the common baseline in each region. KCI is ANN-reference directional agreement.\n\nThe expanded-y figures use a shared piecewise-linear y scale with a change of slope at 1.35; no values were omitted. The linear versions use a common linear y scale. All x axes include KCI = 100%.\n\nThe monochrome companion directory contains one figure per family. Each parameter is identified by its panel title rather than colour; marker shape identifies the seed and hollow markers show No-Gain. The same No-Gain points appear in every row for comparison.\n''', encoding='utf-8')
    produced = [OUT/f'validation_{part}_{scale}.{ext}' for scale in ('expanded_y', 'linear')
                for part in ('all_families', 'part1', 'part2') for ext in ('pdf', 'png')]
    produced += [OUT/f'validation_{part}_{scale}_caption.txt' for scale in ('expanded_y', 'linear')
                 for part in ('all_families', 'part1', 'part2')]
    produced.append(notes)
    manifest = {'source_sha256': original, 'script_sha256': sha(Path(__file__)),
                'model_runs': 63, 'regions': 2, 'unique_model_region_coordinates': 126,
                'outputs': outputs, 'output_sha256': {p.name:sha(p) for p in produced},
                'style': '190 mm width, PDF vector; PNG 300 dpi preview',
                'note': 'JTICE user-provided guide snapshot; current journal instructions need verification.'}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
