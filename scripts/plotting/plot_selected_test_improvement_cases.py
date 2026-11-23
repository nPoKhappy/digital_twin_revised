"""Plot exploratory, performance-selected test cases from saved TS predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42/updated_validation_best'
OUT = SOURCE / 'selected_test_ts_examples'
SEEDS = (42, 43, 44)
TARGETS = ('B35_H2S', 'B35_SO2')
FAMILIES = {
    'no_gain': ('No-Gain', '#777777', '--'),
    'gain25': ('Gain = 25', '#0072B2', '-'),
    'ss2p5_only': ('SS-only = 2.5', '#009E73', '-.'),
    'gain25_ss25': ('Gain + SS = 25 / 25', '#D55E00', (0, (5, 2, 1, 2))),
    'dg_ratio1p5': (r'Difficult-GradNorm, $\rho$ = 1.5', '#CC79A7', ':'),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    global FAMILIES
    selected_models = json.loads((SOURCE / 'selected_models.json').read_text(encoding='utf-8'))['selected_models']
    aliases = {}
    for family in ('no_gain', 'gain', 'ss_only', 'gain_ss', 'difficult_gradnorm'):
        rows = [r for r in selected_models if r['family'] == family]
        if len(rows) != 3 or {r['seed'] for r in rows} != set(SEEDS) or len({r['alias'] for r in rows}) != 1:
            raise ValueError(f'Invalid test selection for {family}')
        aliases[family] = rows[0]['alias']
    FAMILIES = {
        aliases['no_gain']: ('No-Gain', '#777777', '--'),
        aliases['gain']: ('Gain = ' + aliases['gain'].removeprefix('gain').replace('p', '.'), '#0072B2', '-'),
        aliases['ss_only']: ('SS-only = ' + aliases['ss_only'].removeprefix('ss').removesuffix('_only').replace('p', '.'), '#009E73', '-.'),
        aliases['gain_ss']: ('Gain + SS = ' + aliases['gain_ss'].removeprefix('gain').replace('_ss', ' / ').replace('p', '.'), '#D55E00', (0, (5, 2, 1, 2))),
        aliases['difficult_gradnorm']: (r'Difficult-GradNorm, $\rho$ = ' + aliases['difficult_gradnorm'].removeprefix('dg_ratio').replace('p', '.'), '#CC79A7', ':'),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = SOURCE / 'test_metrics.csv'
    metric = pd.read_csv(metrics_path)
    metric = metric[metric['evaluation'].eq('TS')].copy()
    frames: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    case_regions = metric[['case', 'partition']].drop_duplicates()
    if case_regions['case'].duplicated().any() or len(case_regions) != 12:
        raise ValueError('Unexpected case/partition mapping')
    rows = []
    selection_path = SOURCE / 'selected_models.json'
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (metrics_path, selection_path)}
    for case, partition in case_regions.itertuples(index=False, name=None):
        common_time = common_truth = None
        for family in FAMILIES:
            for seed in SEEDS:
                model = f'{family}_seed{seed}'
                base = SOURCE / 'model_data' / model
                npz = base / f'{case}_prediction.npz'
                time_file = base / f'{case}_time_series.csv'
                with np.load(npz, allow_pickle=False) as data:
                    truth, pred = data['ts_true'].copy(), data['ts_pred'].copy()
                table = pd.read_csv(time_file)
                times = table['time_min'].to_numpy(dtype=float)
                if truth.shape != pred.shape or truth.shape != (len(times), 2):
                    raise ValueError(f'Unexpected prediction dimensions: {model}, {case}')
                np.testing.assert_allclose(truth, table[[f'{t}_true' for t in TARGETS]], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(pred, table[[f'{t}_pred' for t in TARGETS]], rtol=1e-12, atol=1e-12)
                if common_time is None:
                    common_time, common_truth = times, truth
                else:
                    np.testing.assert_array_equal(times, common_time)
                    np.testing.assert_allclose(truth, common_truth, rtol=1e-12, atol=1e-12)
                frames[(case, family, seed)] = (times, truth, pred)
                for j, target in enumerate(TARGETS):
                    computed = float(np.sqrt(np.mean((pred[:, j] - truth[:, j]) ** 2)))
                    ref = metric[(metric['case'] == case) & (metric['model'] == model) & (metric['target'] == target)]
                    if len(ref) != 1 or int(ref.iloc[0]['n']) != len(times):
                        raise ValueError(f'Missing or duplicate TS metric for {case}, {model}, {target}')
                    np.testing.assert_allclose(computed, ref.iloc[0]['RMSE'], rtol=1e-10, atol=1e-12)
                    rows.append({'partition': partition, 'case': case, 'target': target,
                                 'family': family, 'seed': seed, 'n': len(times), 'rmse': computed})
                input_hashes[str(npz.relative_to(ROOT))] = sha256(npz)
                input_hashes[str(time_file.relative_to(ROOT))] = sha256(time_file)
    per_seed = pd.DataFrame(rows)
    family_metrics = per_seed.groupby(['partition', 'case', 'target', 'family'], as_index=False).agg(
        mean_seed_rmse=('rmse', 'mean'), sd_seed_rmse=('rmse', 'std'), n_seeds=('seed', 'nunique'))
    if not family_metrics['n_seeds'].eq(3).all():
        raise ValueError('Every family must have three seeds')
    baseline = family_metrics[family_metrics.family.eq('no_gain')][['partition', 'case', 'target', 'mean_seed_rmse']]
    baseline = baseline.rename(columns={'mean_seed_rmse': 'baseline_mean_seed_rmse'})
    family_metrics = family_metrics.merge(baseline, on=['partition', 'case', 'target'], validate='many_to_one')
    family_metrics['improvement_percent'] = 100 * (1 - family_metrics.mean_seed_rmse / family_metrics.baseline_mean_seed_rmse)
    # Select the case's strongest family by average percent improvement across both outputs.
    scores = (family_metrics[~family_metrics.family.eq('no_gain')]
              .groupby(['partition', 'case', 'family'], as_index=False)
              .agg(mean_output_improvement_percent=('improvement_percent', 'mean'), outputs=('target', 'nunique')))
    if not scores['outputs'].eq(2).all():
        raise ValueError('Both output variables are required')
    ranking = (scores.sort_values(['partition', 'case', 'mean_output_improvement_percent', 'family'],
                                  ascending=[True, True, False, True])
               .drop_duplicates(['partition', 'case'])
               .sort_values(['partition', 'mean_output_improvement_percent', 'case'], ascending=[True, False, True]))
    ranking['rank_in_region'] = ranking.groupby('partition').cumcount() + 1
    ranking['selected'] = ranking['rank_in_region'] <= 2
    # Keep the illustrative cases the user previously chose, including R5-6.
    selected = {'internal': ['R5-2', 'R5-1', 'R5-6'], 'external': ['R5-23', 'R5-20']}
    worst = (scores[scores.partition.eq('internal')]
             .sort_values(['mean_output_improvement_percent', 'case', 'family']).iloc[0])
    r56 = (scores[(scores.partition == 'internal') & (scores.case == 'R5-6')]
           .sort_values(['mean_output_improvement_percent', 'family']).iloc[0])
    ranking['selected'] = ranking.apply(lambda r: r['case'] in selected[r['partition']], axis=1)
    ranking['selection_reason'] = np.where(ranking['selected'], 'previously_selected_example', '')
    ranking['selection_driver_family'] = ranking['family']
    ranking['selection_driver_score_percent'] = ranking['mean_output_improvement_percent']
    ranking.loc[(ranking.partition == 'internal') & (ranking.case == 'R5-6'), 'selection_reason'] = 'previously_selected_deterioration_example'
    ranking.loc[(ranking.partition == 'internal') & (ranking.case == 'R5-6'), 'selection_driver_family'] = r56['family']
    ranking.loc[(ranking.partition == 'internal') & (ranking.case == 'R5-6'), 'selection_driver_score_percent'] = r56['mean_output_improvement_percent']
    ranking['selected'] = ranking.selection_reason.ne('')
    per_seed.to_csv(OUT / 'case_metrics_per_seed.csv', index=False)
    family_metrics.to_csv(OUT / 'case_metrics_three_seed_mean.csv', index=False)
    ranking.to_csv(OUT / 'case_selection_ranking.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 10, 'legend.fontsize': 8.2, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    legend = [Line2D([0], [0], color='black', lw=2.0, label='Aspen reference')]
    legend += [Line2D([0], [0], color=color, ls=style, lw=1.5, label=label)
               for label, color, style in FAMILIES.values()]
    for partition, cases in selected.items():
        if len(cases) != (3 if partition == 'internal' else 2):
            raise ValueError(f'Unexpected number of cases for {partition}')
        fig, axes = plt.subplots(len(cases), 2, figsize=(7.2, 6.8 if len(cases) == 3 else 5.1), sharex=False)
        for row, case in enumerate(cases):
            for col, target in enumerate(TARGETS):
                ax = axes[row, col]
                time, truth, _ = frames[(case, 'no_gain', 42)]
                ax.plot(time, truth[:, col] * 1000, color='black', lw=2.0, zorder=6)
                for family, (_, color, style) in FAMILIES.items():
                    preds = np.stack([frames[(case, family, seed)][2][:, col] for seed in SEEDS])
                    ax.plot(time, preds.mean(axis=0) * 1000, color=color, ls=style, lw=1.5, zorder=5)
                label = r'H$_2$S' if col == 0 else r'SO$_2$'
                ax.set_title(f'{case}  ·  {label}', loc='left')
                ax.set_xlabel('Time (min)')
                ax.set_ylabel(r'Mole fraction ($10^{-3}$)')
                ax.grid(axis='y', color='0.88', lw=0.5)
                ax.tick_params(direction='out', length=3)
        fig.legend(handles=legend, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.0), frameon=False)
        fig.subplots_adjust(left=0.11, right=0.985, top=0.95 if len(cases) == 3 else 0.93,
                            bottom=0.14 if len(cases) == 3 else 0.18, hspace=0.48, wspace=0.36)
        fig.savefig(OUT / f'{partition}_selected_test_ts.png', dpi=600, facecolor='white')
        fig.savefig(OUT / f'{partition}_selected_test_ts.pdf', facecolor='white')
        plt.close(fig)
    protocol = {
        'purpose': 'Exploratory test examples deliberately selected by observed test performance; not representative or independent model-selection evidence.',
        'selection': 'The five illustrated cases are retained from the previous requested figure, including the earlier R5-6 deterioration example. Current per-case results and rankings are recomputed; selection is based on previously examined test outcomes and is not representative.',
        'worst_internal_case': {'case': worst['case'], 'family': worst['family'], 'mean_output_improvement_percent': float(worst['mean_output_improvement_percent'])},
        'selected_cases': selected,
        'seeds': list(SEEDS), 'models': list(FAMILIES),
        'prediction_curve': 'Arithmetic mean of the three seed TS prediction values at each time; no seed uncertainty band. RMSE of this mean curve differs from the mean of individual-seed RMSEs used for case selection.',
        'time_series': '18 Aspen reference history steps initialize the encoder; subsequent output predictions are fed back at each sliding-window step. Decoder inputs are recorded future inputs. The curves are TS, not H1/H18, and are conditional on these recorded inputs.',
        'reference': 'Aspen-derived recorded time series, 10-row median downsampled. Internal test uses held-out segments of R5 cases; external uses R5-19 to R5-26. These test data have been viewed previously.',
        'input_sha256': input_hashes,
    }
    (OUT / 'provenance.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (OUT / 'README.md').write_text(
        '# Exploratory test time-series examples\n\n'
        'Cases are selected by test performance, so these examples are illustrative, not a representative performance estimate. '
        'The ranking and all case/seed RMSE values are provided in the accompanying CSV files.\n\n'
        f'Selected: interpolation {selected["internal"]}; extrapolation {selected["external"]}. '
        'These are the five cases illustrated in the previous requested figure, including R5-6. '
        'Current ranks use the average of H2S and SO2 percentage improvements in mean individual-seed TS RMSE '
        f'relative to the mean No-Gain seed RMSE. The largest current internal deterioration is {worst["case"]} '
        f'with {worst["family"]} (mean improvement {worst["mean_output_improvement_percent"]:.1f}%). '
        'Three seeds: 42, 43, 44.\n\n'
        'The y-axis is mole fraction in units of 10^-3 kmol/kmol. '
        'Curves show the arithmetic mean of seed-wise predictions, without error bars. '
        'The RMSE of a mean prediction curve is not the same as the mean of individual seed RMSEs. '
        'TS predictions use 18 Aspen reference history steps followed by recursive model-output feedback, '
        'with the recorded decoder inputs supplied at each step; these are not H1 or H18 results. '
        'The test cases were previously examined. See provenance.json for source hashes.\n\n'
        'Recreate from repository root: `python -m scripts.plotting.plot_selected_test_improvement_cases`\n', encoding='utf-8')
    print(ranking.to_string(index=False))
    print('Output:', OUT)


if __name__ == '__main__':
    main()
