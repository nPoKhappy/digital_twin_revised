"""Redraw the selected-test ANN-reference KCI breakdown at publication scale.

Inputs are the already audited three-seed CSVs; no inference or resampling.
Run: python -m scripts.analysis.plot_jtice_teacher_gain_kci
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
from PIL import Image
from scripts.analysis.plot_retrained_teacher_sign_seed42 import model_label

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/gain_pair_difficulty_seed42'
OUT = SOURCE / 'jtice_figures'
FAMILIES = [
    ('no_gain', 'no_gain', 'No-Gain', '#222222', 's', '--'),
    ('gain', 'gain25', 'Gain = 25', '#0072B2', 'o', '-'),
    ('ss_only', 'ss2p5_only', 'SS-only = 2.5', '#009E73', '^', '-.'),
    ('gain_ss', 'gain25_ss25', 'Gain + SS = 25 / 25', '#D55E00', 'D', ':'),
    ('difficult_gradnorm', 'dg_ratio1p5', 'Difficult-GradNorm, $\\rho$ = 1.5', '#CC79A7', 'v', (0, (5, 1, 1, 1))),
]
PAIRS = [('B35_H2S', 'air2_SP', r'Air2 $\rightarrow$ H$_2$S'),
         ('B35_H2S', 'HEATER2_output_T_SP', r'T2 $\rightarrow$ H$_2$S'),
         ('B35_SO2', 'air2_SP', r'Air2 $\rightarrow$ SO$_2$'),
         ('B35_SO2', 'HEATER2_output_T_SP', r'T2 $\rightarrow$ SO$_2$')]
REGIONS = [('in_range', 'In-range'), ('soft_ood', 'Soft-OOD')]
Y_LIMITS = (-6, 106)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated(kind: str) -> pd.DataFrame:
    prefix = 'conditional_kci_by_teacher_sign' if kind == 'sign' else 'conditional_kci_by_gain_magnitude'
    group = 'teacher_sign_label' if kind == 'sign' else 'quartile'
    summary = pd.read_csv(SOURCE / f'{prefix}.csv')
    seeds = pd.read_csv(SOURCE / f'{prefix}_per_seed.csv')
    keys = ['family', 'alias', 'distribution', 'target', 'mv', group]
    expected = {f: a for f, a, *_ in FAMILIES}
    assert summary.groupby('family').alias.nunique().eq(1).all()
    assert summary.groupby('family').alias.first().to_dict() == expected
    assert set(seeds.seed) == {42, 43, 44}
    assert len(summary) == (80 if kind == 'sign' else 160)
    assert len(seeds) == (240 if kind == 'sign' else 480)
    assert not seeds.duplicated(keys + ['seed']).any()
    assert seeds.groupby(keys).seed.nunique().eq(3).all()
    assert seeds.groupby(keys).probe_count.nunique().eq(1).all()
    # Reconstruct each seed's score from exact comparison counts.
    valid = seeds.total > 0
    assert np.allclose(seeds.loc[valid, 'kci_percent'],
                       100 * seeds.loc[valid, 'correct'] / seeds.loc[valid, 'total'], rtol=0, atol=1e-10)
    assert seeds.loc[~valid, 'kci_percent'].isna().all()
    rebuilt = seeds.groupby(keys, as_index=False).agg(
        count=('kci_percent', 'count'), probes=('probe_count', 'first'),
        mean=('kci_percent', 'mean'), sd=('kci_percent', 'std'))
    d = summary.merge(rebuilt, on=keys, validate='one_to_one')
    assert len(d) == len(summary)
    assert (d.n_seeds == d['count']).all() and (d.probe_count == d.probes).all()
    assert np.allclose(d.mean_kci_percent, d['mean'], equal_nan=True, rtol=0, atol=1e-10)
    assert np.allclose(d.sd_kci_percentage_points, d['sd'], equal_nan=True, rtol=0, atol=1e-10)
    assert d.n_seeds.isin([0, 3]).all()
    return summary


def make_figure(kind: str, frame: pd.DataFrame) -> None:
    group = 'teacher_sign_label' if kind == 'sign' else 'quartile'
    groups = ['Negative', 'Positive'] if kind == 'sign' else ['Q1', 'Q2', 'Q3', 'Q4']
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .6})
    # Elsevier-wide double-column starting width, not a verified JTICE-specific dimension.
    fig, axes = plt.subplots(2, 4, figsize=(190/25.4, 133/25.4), sharey=True)
    fig.subplots_adjust(left=.073, right=.985, bottom=.115, top=.79, wspace=.23, hspace=.48)
    handles = [Line2D([0], [0], color=c, marker=m, linestyle=ls, lw=1, ms=3.6, label=label)
               for _, _, label, c, m, ls in FAMILIES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .96), ncol=3,
               frameon=False, fontsize=6.5, handlelength=2.0, columnspacing=1.0, handletextpad=.4)
    panel = 0
    offsets = np.linspace(-.065, .065, len(FAMILIES))
    for ri, (region, region_label) in enumerate(REGIONS):
        for ci, (target, mv, pair_label) in enumerate(PAIRS):
            ax = axes[ri, ci]
            subset = frame[(frame.distribution == region) & (frame.target == target) & (frame.mv == mv)]
            for i, (family, _, _, color, marker, ls) in enumerate(FAMILIES):
                rows = subset[subset.family == family].set_index(group).reindex(groups)
                assert len(rows) == len(groups) and not rows.n_seeds.isna().any()
                x = np.arange(len(groups)) + offsets[i]
                y = rows.mean_kci_percent.to_numpy(dtype=float)
                sd = rows.sd_kci_percentage_points.to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker=marker, linestyle=ls, linewidth=.85, markersize=3.1,
                        markeredgewidth=.35, zorder=2+i)
                mask = np.isfinite(y) & np.isfinite(sd)
                ax.errorbar(x[mask], y[mask], yerr=sd[mask], fmt='none', ecolor=color,
                            elinewidth=.7, capsize=1.3, capthick=.7, zorder=1+i)
            for j, value in enumerate(groups):
                n = subset[subset[group] == value].probe_count
                assert len(n) == len(FAMILIES) and n.nunique() == 1
                if int(n.iloc[0]) == 0:
                    ax.text(j, 8, 'n = 0', color='#555555', ha='center', va='center', fontsize=6)
            ax.set_ylim(*Y_LIMITS)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(range(len(groups)), groups)
            ax.set_xlim(-.35, len(groups)-.65)
            ax.tick_params(length=2.2, width=.6, pad=2)
            ax.grid(axis='y', color='#d8d8d8', linewidth=.35)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'({chr(97+panel)}) {pair_label}', loc='left', fontsize=7, pad=3)
            if ci == 0:
                ax.set_ylabel(f'{region_label}\nKCI (%)')
            if ri == 1:
                ax.set_xlabel('ANN gain sign' if kind == 'sign' else 'ANN |gain| quartile')
            panel += 1
    name = 'kci_teacher_gain_sign' if kind == 'sign' else 'kci_teacher_gain_magnitude'
    fig.savefig(OUT / f'{name}.pdf', facecolor='white')
    fig.savefig(OUT / f'{name}.png', dpi=600, facecolor='white')
    with Image.open(OUT / f'{name}.png') as preview:
        preview.convert('L').save(OUT / f'{name}_grayscale.png', dpi=(600, 600))
    plt.close(fig)


def write_captions() -> list[Path]:
    common = (
        'Panels (a-d) show in-range test operating points; panels (e-h) show soft-OOD test operating points. '
        'The columns show Air2 to H2S, T2 to H2S, Air2 to SO2 and T2 to SO2, respectively. '
        'The key identifies the model family and its validation-selected parameters. '
        'KCI is the percentage of dynamic-model gain directions agreeing with the shared ANN reference, '
        'over operating points with absolute ANN gain at least 1e-5, two perturbation directions '
        'and 18 prediction steps; a zero dynamic gain counts as a disagreement. '
        'Points show the mean of model-training seeds 42, 43 and 44; error bars show plus or minus one '
        'sample standard deviation across these three seeds (n = 3). All three runs share the same ANN '
        'and test operating points. The bars describe training-seed variability, not a confidence interval. '
        'Markers are displaced horizontally within groups solely to improve visibility. '
        'SS denotes steady state and rho is the Difficult-GradNorm target ratio. '
    )
    captions = {
        'kci_teacher_gain_sign_caption.txt': (
            'KCI grouped by ANN gain sign. ' + common +
            'The groups are the negative and positive signs of the ANN gain. In both regions, '
            'n = 0 for Air2 to SO2 negative gain means that there are no eligible operating points; '
            'no KCI value is shown for that group.'
        ),
        'kci_teacher_gain_magnitude_caption.txt': (
            'KCI grouped by ANN gain magnitude. ' + common +
            'Q1-Q4 are quartiles of normalized absolute ANN gain, calculated separately for each '
            'region and input-output pair using the existing test grouping. '
            'The source data and quartile-edge table provide the grouping thresholds and counts.'
        ),
    }
    paths = []
    for name, caption in captions.items():
        path = OUT / name
        path.write_text(caption + '\n', encoding='utf-8')
        paths.append(path)
    return paths


def main() -> None:
    global FAMILIES, Y_LIMITS
    source_test = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42/updated_validation_best'
    selected = json.loads((source_test / 'selected_models.json').read_text(encoding='utf-8'))['selected_models']
    aliases = {}
    for family, _, _, _, _, _ in FAMILIES:
        rows = [row for row in selected if row['family'] == family]
        if len(rows) != 3 or {int(row['seed']) for row in rows} != {42, 43, 44} or len({row['alias'] for row in rows}) != 1:
            raise ValueError(f'Invalid selected models for {family}')
        aliases[family] = rows[0]['alias']
    FAMILIES = [(family, aliases[family], model_label(family, aliases[family]), color, marker, style)
                for family, _, _, color, marker, style in FAMILIES]
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {kind: validated(kind) for kind in ('sign', 'magnitude')}
    bounds = pd.concat(list(frames.values()), ignore_index=True)
    lower = (bounds.mean_kci_percent - bounds.sd_kci_percentage_points).dropna()
    upper = (bounds.mean_kci_percent + bounds.sd_kci_percentage_points).dropna()
    Y_LIMITS = (min(-6, float(lower.min()) - 2), max(106, float(upper.max()) + 2))
    for kind, frame in frames.items():
        make_figure(kind, frame)
    files = [SOURCE / f'conditional_kci_by_{stem}{suffix}.csv'
             for stem in ('teacher_sign', 'gain_magnitude') for suffix in ('', '_per_seed')]
    captions = write_captions()
    products = sorted(OUT.glob('kci_teacher_gain_*.*'))
    assert all(p in products for p in captions)
    notes = OUT / 'figure_notes.md'
    notes.write_text(
        '# Figure preparation and provenance\n\n'
        'The colour PDFs are vector figures with embedded fonts and a 190 x 133 mm canvas. '
        'The PNG and grayscale PNG files are previews; PNG is not listed as a final format in the '
        'JTICE author-guide text supplied by the user. See the two separately supplied caption '
        'files for figure descriptions, definitions and error-bar details. The 190 mm width is an '
        'Elsevier general sizing example, not a confirmed JTICE-specific width. Confirm the current '
        'journal instructions and production dimensions before submission.\n\n'
        'These test probes were used in prior analyses and are not a fresh blind holdout. '
        'KCI is ANN-reference directional agreement and does not alone prove Aspen physical gain direction. '
        'Some historical validation files were deleted; consult source READMEs and provenance for limitations.\n\n'
        'Rebuild from the repository root with '
        '`python -m scripts.analysis.plot_jtice_teacher_gain_kci`. '
        'The script checks per-seed counts and reconstructs the reported means and sample SD; '
        'it does not retrain or re-evaluate models.\n', encoding='utf-8')
    (OUT / 'figure_provenance.json').write_text(json.dumps({
        'purpose': 'JTICE-oriented provisional figures; verify journal guide before submission',
        'source_files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'script_sha256': sha(Path(__file__)),
        'output_files_sha256': {p.name: sha(p) for p in [*products, notes]},
        'models': {family: alias for family, alias, *_ in FAMILIES},
        'statistics': 'Mean of training seeds 42/43/44 +/- 1 sample SD; all valid test probes; seeds share one ANN and operating points',
        'scope': 'ANN-reference directional agreement; test probes were previously used and are not a fresh blind test',
        'sign_empty_groups': 'Air2 to SO2, negative ANN gain: n=0 in both regions',
        'dimensions_mm': [190, 133], 'pdf_fonttype': 42, 'png_preview_dpi': 600,
    }, indent=2), encoding='utf-8')
    print(f'Wrote two PDF figures plus color and grayscale PNG previews to {OUT}')


if __name__ == '__main__':
    main()
