"""Plot five seed42 models from existing predictions and all18 KCI; no model updates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.plotting import plot_gain_vs_no_gain_comparison as rmse_tools
from scripts.plotting import plot_kci_pair_comparison as kci_tools
from scripts.analysis import analyze_gain_pair_difficulty as difficulty

OUTPUT = ROOT / 'results/visualization/five_model_comparison_seed42_ratio0p01_0p015'
OLD_DG = 'Difficult-GradNorm ratio=0.015'
NEW_DG = 'Difficult-GradNorm ratio=0.01'
MODELS = {
    **{k: v for k, v in rmse_tools.RESULT_DIRS.items() if k != 'Difficult-GradNorm'},
    OLD_DG: rmse_tools.RESULT_DIRS['Difficult-GradNorm'],
    NEW_DG: 'pgin_gain005_ratio0p01_seed42_difficult_group_gradnorm_no_ss',
}
KCI_DIRS = {
    **{k: v[0].split('/')[0] for k, v in kci_tools.RESULT_FILES.items() if k != 'Difficult-GradNorm'},
    OLD_DG: kci_tools.RESULT_FILES['Difficult-GradNorm'][0].split('/')[0],
    NEW_DG: 'difficult_gradnorm_ratio0p01_no_ss_seed42',
}
STYLES = {
    **{k: v for k, v in rmse_tools.MODEL_STYLES.items() if k != 'Difficult-GradNorm'},
    OLD_DG: rmse_tools.MODEL_STYLES['Difficult-GradNorm'],
    NEW_DG: {'color': '#6A3D9A', 'marker': 'P', 'linestyle': '--'},
}
SOURCES: dict[str, str] = {}
FIGURES = []


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def protect(path):
    SOURCES[str(Path(path).relative_to(ROOT))] = digest(path)


def save(fig, name):
    fig.canvas.draw()
    for ax in fig.axes:
        for line in ax.lines:
            x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                lo, hi = ax.get_ylim()
                assert (y[finite] >= lo).all() and (y[finite] <= hi).all(), name
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=240, bbox_inches='tight')
    FIGURES.append(name)
    plt.close(fig)


def decorate(fig, axes, title, footnote, top=.80):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=15, fontweight='bold', y=.99)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .012, footnote, ha='center', fontsize=9)
    fig.subplots_adjust(top=top, bottom=.13, hspace=.46, wspace=.36)


def plot_rmse(data, cases, label, name):
    fig, axes = plt.subplots(2, len(cases), figsize=(3.15 * len(cases), 7.6), squeeze=False)
    for q, target in enumerate(rmse_tools.TARGETS):
        for j, case in enumerate(cases):
            ax = axes[q, j]
            for model, style in STYLES.items():
                values = [rmse_tools.metric_value(data, model, case, target, e, 'RMSE') for e in rmse_tools.EVALUATIONS]
                ax.plot(range(3), values, **style, label=model, lw=1.7, ms=5.5)
            ax.set_title(f'{case} | {rmse_tools.TARGET_LABELS[target]}', fontweight='bold')
            ax.set_xticks(range(3), rmse_tools.EVALUATIONS)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
            ax.margins(y=.18)
            ax.grid(axis='y', alpha=.3)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0: ax.set_ylabel(f'{rmse_tools.TARGET_LABELS[target]} RMSE')
            if q == 1: ax.set_xlabel('Evaluation')
    decorate(fig, axes, f'Five-model original RMSE comparison ({label}; seed42)',
             'Original RMSE; H1, H18 and rolling time series. No case averaging; panel y-axis ranges vary. No error bars.')
    save(fig, name)


def plot_kci(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for j, distribution in enumerate(kci_tools.DISTRIBUTIONS):
        ax = axes[j]
        for model, style in STYLES.items():
            values = [kci_tools.kci_value(data[data.model.eq(model)], distribution, pair) for pair in kci_tools.PAIR_KEYS]
            ax.plot(range(4), values, **style, label=model, lw=1.8, ms=6)
        ax.set_title(f'{"AB"[j]}  {kci_tools.DISTRIBUTION_LABELS[distribution]}', loc='left', fontweight='bold')
        ax.set_xticks(range(4), kci_tools.PAIR_LABELS, rotation=16, ha='right')
        ax.set_ylim(65, 102)
        ax.set_xlabel('Gain pair')
        ax.grid(axis='y', alpha=.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('All18 KCI (%)')
    decorate(fig, axes, 'Five-model all18 KCI comparison (seed42)',
             'KCI is ANN-teacher sign agreement among valid comparisons; it is not a prediction RMSE or control metric.', top=.74)
    fig.subplots_adjust(bottom=.24)
    save(fig, 'five_model_all18_kci')


def plot_conditional(data, column, groups, title, name):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharey=True)
    for i, distribution in enumerate(difficulty.DISTRIBUTIONS):
        for j, pair in enumerate(difficulty.PAIR_ORDER):
            ax = axes[i, j]
            for model, style in STYLES.items():
                rows = data[(data.model == model) & (data.distribution == distribution) & (data.pair == pair)].set_index(column).loc[groups]
                ax.plot(range(len(groups)), rows.kci_percent.to_numpy(), **style, label=model, lw=1.5, ms=5)
            ax.set_title(difficulty.PLOT_PAIR_LABELS[pair], fontweight='bold')
            ax.set_xticks(range(len(groups)), groups)
            ax.set_xlim(-.08, len(groups) - 1 + .08)
            ax.set_ylim(0, 103)
            ax.grid(axis='y', alpha=.3)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0: ax.set_ylabel(f'{difficulty.DISTRIBUTION_LABELS[distribution]}\nAll18 KCI (%)')
            if i == 1: ax.set_xlabel('Teacher gain magnitude quartile' if column == 'magnitude_bin' else 'Teacher gain sign')
            if column == 'teacher_sign_label' and pair == difficulty.PAIR_ORDER[2]:
                ax.text(0, 8, 'N/A', ha='center', fontsize=9, color='#666666')
    decorate(fig, axes, f'Five-model all18 KCI by teacher gain {title} (seed42)',
             'Pooled valid comparisons; empty teacher-sign groups are N/A. Lines show point estimates, without error bars.')
    save(fig, name)


def conditional_summary(probe, model, column):
    groups = difficulty.QUARTILES if column == 'magnitude_bin' else difficulty.SIGN_ORDER
    rows = []
    for distribution in difficulty.DISTRIBUTIONS:
        for pair in difficulty.PAIR_ORDER:
            for value in groups:
                p = probe[(probe.distribution == distribution) & (probe.pair == pair) & (probe[column] == value)]
                total, correct = int(p.total.sum()), int(p.correct.sum())
                rows.append(dict(model=model, distribution=distribution, pair=pair, **{column: value},
                                 probe_points=len(p), comparisons=total, correct=correct,
                                 kci_percent=100 * correct / total if total else np.nan))
    return pd.DataFrame(rows)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(OUTPUT.relative_to(ROOT)), help='A fresh directory under results/visualization.')
    args = parser.parse_args()
    OUTPUT = (ROOT / args.output_dir).resolve()
    if ROOT / 'results/visualization' not in OUTPUT.parents:
        raise ValueError('Output must be inside results/visualization.')
    if OUTPUT.exists():
        raise FileExistsError(f'Choose a fresh output directory; preserving {OUTPUT}')
    OUTPUT.mkdir(parents=True)
    for path in (ROOT / 'results/visualization').rglob('*'):
        if path.is_file() and OUTPUT not in path.parents: protect(path)
    rmse_tools.configure_style()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11})
    frames = []
    case_checks = 0
    for model, directory in MODELS.items():
        base = ROOT / 'results' / directory
        for f in ['grouped_horizon_parity/grouped_parity_metrics.csv', 'grouped_time_series/grouped_time_series_metrics.csv']: protect(base / f)
        frame = rmse_tools.load_model_metrics(ROOT, model, directory)
        assert len(frame) == 72
        for case, group in frame.groupby('case', sort=False):
            case_dir = base / (f'{case}_test_split' if case in rmse_tools.INTERNAL_CASES else case)
            hpath, tpath = case_dir / 'horizon_metrics_t1_to_t18.csv', case_dir / 'evaluation_metrics.csv'
            protect(hpath); protect(tpath)
            horizon = pd.read_csv(hpath).set_index(['Variable', 'Horizon'])
            ts = pd.read_csv(tpath).set_index('Variable')
            for row in group.itertuples():
                saved = ts.loc[row.target] if row.evaluation == 'TS' else horizon.loc[(row.target, int(row.evaluation[1:]))]
                np.testing.assert_allclose([row.RMSE, row.R2], [saved.RMSE, saved.R2], rtol=1e-9, atol=1e-12)
                case_checks += 1
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    assert len(values) == case_checks == 360 and not values.duplicated(['model', 'case', 'target', 'evaluation']).any()
    old_values = pd.read_csv(ROOT / 'results/visualization/four_model_rmse_visualization_seed42/four_model_rmse_values.csv')
    old_values['model'] = old_values.model.replace({'Difficult-GradNorm': OLD_DG})
    matching = values[values.model != NEW_DG].merge(old_values, on=['model', 'case', 'target', 'evaluation'], suffixes=('_new', '_old'), validate='one_to_one')
    assert len(matching) == 288
    np.testing.assert_allclose(matching[['RMSE_new', 'R2_new']], matching[['RMSE_old', 'R2_old']], rtol=1e-9, atol=1e-12)
    values.to_csv(OUTPUT / 'five_model_rmse_values.csv', index=False)
    plot_rmse(values, rmse_tools.INTERNAL_CASES, 'internal test', 'five_model_raw_rmse_internal_test')
    plot_rmse(values, rmse_tools.EXTERNAL_CASES, 'external all-data', 'five_model_raw_rmse_external_all_data')
    print('[Done] 360 RMSE/R2 rows verified and plotted.', flush=True)

    scale_path = ROOT / 'results/Tabular_MLP_New/zscore_std.csv'
    protect(scale_path)
    scales = pd.read_csv(scale_path, index_col=0).squeeze('columns')
    teacher_path = ROOT / 'results/gain_eval_all18' / KCI_DIRS['No-Gain'] / 'gain_probe_diagnostics.csv'
    teacher = difficulty.load_teacher_landscape(teacher_path, scales, 200000)
    edges = difficulty.build_quartile_edges(teacher)
    summary = difficulty.summarize_teacher(teacher)
    edges.to_csv(OUTPUT / 'gain_magnitude_quartile_edges.csv', index=False)
    summary.to_csv(OUTPUT / 'teacher_gain_summary.csv', index=False)
    kciframes, magnitudes, signs = [], [], []
    for model, directory in KCI_DIRS.items():
        print('[Read] Full probe diagnostics:', model, flush=True)
        base = ROOT / 'results/gain_eval_all18' / directory
        protect(base / 'gain_probe_diagnostics.csv'); protect(base / 'gain_evaluation_detail.csv')
        official = pd.read_csv(base / 'gain_evaluation_detail.csv')
        assert len(official) == 8 and official.horizon.eq('all18').all() and official.seed.eq(42).all()
        rebuilt = 100 * (official.tp + official.tn) / official.valid_comparisons
        np.testing.assert_allclose(rebuilt, official.kci_percent, rtol=1e-12)
        official.insert(0, 'model', model)
        kciframes.append(official)
        probe = difficulty.load_probe_accuracy(base / 'gain_probe_diagnostics.csv', scales, edges, 200000)
        difficulty.validate_against_official(ROOT / 'results/gain_eval_all18', model, directory, probe)
        for column, store in [('magnitude_bin', magnitudes), ('teacher_sign_label', signs)]:
            grouped = conditional_summary(probe, model, column)
            for r in official.itertuples():
                subset = grouped[(grouped.distribution == r.distribution) & (grouped.pair == difficulty.PAIR_LABELS[(r.target, r.mv)])]
                assert subset.comparisons.sum() == r.valid_comparisons
                assert subset.correct.sum() == r.tp + r.tn
            store.append(grouped)
    kci_data, magnitude_data, sign_data = [pd.concat(items, ignore_index=True) for items in [kciframes, magnitudes, signs]]
    assert len(kci_data) == 40 and len(magnitude_data) == 160 and len(sign_data) == 80
    kci_data.to_csv(OUTPUT / 'five_model_all18_kci_values.csv', index=False)
    magnitude_data.to_csv(OUTPUT / 'conditional_kci_by_gain_magnitude.csv', index=False)
    sign_data.to_csv(OUTPUT / 'conditional_kci_by_teacher_sign.csv', index=False)
    plot_kci(kci_data)
    plot_conditional(magnitude_data, 'magnitude_bin', difficulty.QUARTILES, 'magnitude', 'conditional_kci_by_gain_magnitude')
    plot_conditional(sign_data, 'teacher_sign_label', difficulty.SIGN_ORDER, 'sign', 'conditional_kci_by_teacher_sign')
    difficulty.plot_teacher_landscape(teacher, summary, OUTPUT)
    FIGURES.append('teacher_gain_landscape')
    assert all(digest(ROOT / path) == value for path, value in SOURCES.items())
    validation = {'passed': True, 'training_seed': 42, 'model_count': 5, 'RMSE_R2_rows_rechecked': case_checks,
                  'original_four_model_rows_unchanged': len(matching), 'KCI_pair_rows_rechecked': 40,
                  'conditional_magnitude_rows': 160, 'conditional_sign_rows': 80, 'conditional_counts_reconciled_to_official': True,
                  'optimizer_updates': 0, 'error_bars': False, 'original_sources_unchanged': True, 'visual_review': 'pending',
                  'figures': FIGURES, 'protected_source_sha256': SOURCES}
    (OUTPUT / 'figure_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    provenance = {'dynamic_result_directories': MODELS, 'KCI_directories': KCI_DIRS, 'script': str(Path(__file__).relative_to(ROOT)),
                  'script_sha256': digest(__file__), 'comparison': 'five models; includes both DG ratios; not a two-model-only comparison',
                  'KCI': 'all18 strict sign agreement among valid ANN-teacher comparisons',
                  'conditional_KCI': 'pooled comparison counts; common teacher magnitude quartiles; no uncertainty intervals plotted'}
    (OUTPUT / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('[PASS] Six PNG/PDF figures, source data and verification saved to', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
