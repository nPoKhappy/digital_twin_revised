"""Render common-No-Gain-seed42 validation figures and an offline report.

This reads the separately calculated fixed-baseline tables.  Original evaluation
and model artifacts are never modified; the exact model/parameter definitions
and plotting functions are shared with the existing three-seed visualizations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting import plot_retrain_63_publication as linear
from scripts.plotting import plot_retrain_63_expanded_y as expanded
from scripts.plotting.render_retrain_validation_html import PAGE
from scripts.plotting.retrain_parameter_ui import model_parameters, render_parameter_page

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'
SOURCE = OUTPUT / 'regional_coordinates_63_models.csv'
ORIGINAL = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation/regional_coordinates_63_models.csv'
SEEDS = (42, 43, 44)
DIST = {'in_range': 'internal', 'soft_ood': 'external'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    report = json.loads((OUTPUT / 'analysis_validation.json').read_text(encoding='utf-8-sig'))
    provenance = json.loads((OUTPUT / 'provenance.json').read_text(encoding='utf-8-sig'))
    if (not report.get('passed') or report.get('model_count') != 63
            or report.get('regional_points') != 126 or provenance.get('baseline_seed') != 42):
        raise ValueError('Fixed-seed-42 analysis has not completed successfully.')
    for name, digest in provenance['outputs_sha256'].items():
        if sha(OUTPUT / name).lower() != digest.lower():
            raise ValueError(f'Fixed-baseline analysis output changed: {name}')
    for name, digest in provenance['sources_sha256'].items():
        if sha(ROOT / name).lower() != digest.lower():
            raise ValueError(f'Fixed-baseline analysis input changed: {name}')
    data = pd.read_csv(SOURCE)
    previous = pd.read_csv(ORIGINAL)
    meta = model_parameters(ROOT)
    by_name = {m['model']: m for m in meta}
    keys = ['seed', 'model', 'family', 'region']
    expected = {(s, m, v['family'], r) for s in SEEDS for m, v in by_name.items() for r in ('internal', 'external')}
    actual = set(map(tuple, data[keys].itertuples(index=False, name=None)))
    if len(data) != 126 or actual != expected:
        raise ValueError('Expected 126 unique fixed-baseline model/seed/region coordinates.')
    if not np.isfinite(data[['KCI_percent', 'AvgRelRMSE']].to_numpy(float)).all() or not data.AvgRelRMSE.gt(0).all():
        raise ValueError('Plot coordinates must be finite with positive relative RMSE.')
    paired = data.merge(previous, on=keys, suffixes=('_fixed', '_original'), validate='one_to_one')
    if len(paired) != 126:
        raise ValueError('New coordinates do not match the original model roster.')
    for col in ['KCI_percent', 'correct', 'valid']:
        np.testing.assert_allclose(paired[f'{col}_fixed'], paired[f'{col}_original'], rtol=0, atol=1e-12)
    if not np.allclose(data.KCI_percent, 100 * data.correct / data.valid, rtol=1e-12):
        raise ValueError('KCI counts do not reproduce KCI percentages.')
    baseline = data[(data.model == 'no_gain') & (data.seed == 42)]
    if len(baseline) != 2 or not np.allclose(baseline.AvgRelRMSE, 1, atol=1e-12):
        raise ValueError('No-Gain seed 42 must equal one in both regions.')
    details = {}
    for seed in SEEDS:
        source = ROOT / f'results/visualization/retrain_lhs10_23_21_models_validation_seed{seed}/kci_components.csv'
        counts = pd.read_csv(source)
        if len(counts) != 168 or counts.duplicated(['model', 'distribution', 'target', 'mv']).any():
            raise ValueError(f'Seed {seed}: invalid KCI component coverage.')
        records = []
        for row in counts.itertuples(index=False):
            records.append(dict(model=row.model, partition=DIST[row.distribution], mv=row.mv,
                                target=row.target, correct=int(row.tp + row.tn), valid=int(row.valid_comparisons)))
        details[str(seed)] = records
        grouped = counts.assign(correct=counts.tp + counts.tn).groupby(['model', 'distribution']).agg(
            correct=('correct', 'sum'), valid=('valid_comparisons', 'sum'))
        for row in data[data.seed == seed].itertuples(index=False):
            distribution = {v: k for k, v in DIST.items()}[row.region]
            check = grouped.loc[(row.model, distribution)]
            if int(check.correct) != int(row.correct) or int(check.valid) != int(row.valid):
                raise ValueError(f'Seed {seed}: KCI components differ: {row.model}, {row.region}.')
    return data, meta, details


def make_page(data, details):
    datasets = {}
    for seed in SEEDS:
        rows = data[data.seed == seed]
        points = [dict(model=r.model, family=r.family, partition=r.region, x=float(r.KCI_percent),
                       y=float(r.AvgRelRMSE), correct=int(r.correct), valid=int(r.valid))
                  for r in rows.itertuples(index=False)]
        datasets[str(seed)] = dict(points=points, details=details[str(seed)])
    page = render_parameter_page(PAGE, datasets, ROOT)
    substitutions = {
        '準確度與增益方向': '準確度與增益方向（共同基準）',
        '左側內插、右側外插：兩個座標分別來自不同的驗證資料集。': '所有 seed 均以 No-Gain seed 42 為基準。左側內插、右側外插：兩個座標分別來自不同的驗證資料集。',
        '相同案例及目前所選 seed 的 No-Gain RMSE': '相同案例、輸出及評估方式的 No-Gain seed 42 RMSE',
        'No-Gain = 1，較低表示相對誤差較小。': 'No-Gain seed 42 = 1；seed 43、44 仍各自以 seed 42 為分母，可能不等於 1。較低表示相對誤差較小。',
        '圖上虛線為 No-Gain 相對誤差基準。': '圖上虛線為 No-Gain seed 42 的共同相對誤差基準。',
        'ANN-reference KCI (%)': 'KCI(%)',
    }
    for old, new in substitutions.items():
        if old not in page:
            raise ValueError(f'Expected report wording not found: {old}')
        page = page.replace(old, new)
    if '目前所選 seed 的 No-Gain RMSE' in page or 'No-Gain = 1' in page:
        raise ValueError('An obsolete same-seed baseline statement remains in the report.')
    return page


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    out_html = OUTPUT / 'regional_validation_nogain42_by_seed.html'
    fig_dir = OUTPUT / 'figures'
    stems = ['nogain42_4x2_linear', 'nogain42_part1_linear', 'nogain42_part2_linear',
             'nogain42_4x2_expanded_y', 'nogain42_part1_expanded_y', 'nogain42_part2_expanded_y']
    names = [f'{stem}.{suffix}' for stem in stems for suffix in ('png', 'pdf')]
    outputs = [out_html, *[fig_dir / name for name in names], OUTPUT / 'figure_provenance.json']
    if not args.overwrite:
        existing = [str(p) for p in outputs if p.exists()]
        if existing:
            raise FileExistsError(f'Refusing to overwrite visualizations: {existing}')
    data, meta, details = load()
    input_hash = sha(SOURCE)
    page = make_page(data, details)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding='utf-8')
    linear.OUTPUT = fig_dir
    expanded.OUTPUT = fig_dir
    # Keep the dense part visibly expanded while leaving space above the largest
    # marker; both split and full figures use the exact same limits.
    expanded.Y_MIN = min(.35, math.floor(float(data.AvgRelRMSE.min()) * .9 * 20) / 20)
    expanded.Y_MAX = max(2.85, math.ceil(float(data.AvgRelRMSE.max()) * 1.05 * 20) / 20)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    old_save = matplotlib.figure.Figure.savefig
    def save_with_baseline(fig, *args, **kwargs):
        if not getattr(fig, '_nogain42_labelled', False):
            if fig.texts:
                header = fig.texts[0]
                if 'Nonlinear Y-axis:' not in header.get_text():
                    raise ValueError('Unexpected existing figure heading.')
                header.set_text('Baseline: No-Gain seed 42  |  Nonlinear Y-axis: expanded below 1.35; no values omitted')
            else:
                fig.text(.5, .945, 'Baseline: No-Gain seed 42', fontsize=8, ha='center', va='center')
            fig._nogain42_labelled = True
        return old_save(fig, *args, **kwargs)
    matplotlib.figure.Figure.savefig = save_with_baseline
    try:
        linear.draw(data, meta, 0, 4, 'nogain42_4x2_linear')
        linear.draw(data, meta, 0, 2, 'nogain42_part1_linear')
        linear.draw(data, meta, 2, 4, 'nogain42_part2_linear')
        marks = expanded.plot_figure(data, meta, 0, 4, 'nogain42_4x2_expanded_y')
        expanded.plot_figure(data, meta, 0, 2, 'nogain42_part1_expanded_y')
        expanded.plot_figure(data, meta, 2, 4, 'nogain42_part2_expanded_y')
    finally:
        matplotlib.figure.Figure.savefig = old_save
    unique = {(seed, model, region, x, y) for _, region, model, seed, x, y in marks}
    if len(marks) != 144 or len(unique) != 126:
        raise AssertionError('Expected all 126 unique coordinates and repeated No-Gain references.')
    if sha(SOURCE) != input_hash:
        raise ValueError('Fixed-baseline input changed while rendering.')
    provenance = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=input_hash,
                      unique_coordinates=126, baseline='same-condition No-Gain seed 42',
                      original_kci_unchanged=True, x_axis='KCI(%)', bottom_caption=False,
                      common_x=[expanded.X_MIN, expanded.X_MAX],
                      nonlinear_y=[expanded.Y_MIN, expanded.Y_SPLIT, expanded.Y_MAX],
                      no_omitted_values=True, jitter=False,
                      files={str(p.relative_to(OUTPUT)): sha(p) for p in outputs if p.is_file() and p.name != 'figure_provenance.json'})
    (OUTPUT / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'HTML: {out_html}')
    print(f'12 figure files: {fig_dir}')


if __name__ == '__main__':
    main()
