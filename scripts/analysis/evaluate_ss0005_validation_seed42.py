"""Compare four SS weights including SS=0.005 on the existing five user-designated validation cases."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiment_paths import experiment_results_dir
from src import utils
from src.dataset import MultiStepS2SDataset
from src.models import get_model
from src.variable_selection import variable_selection
from scripts.prediction import predict as native

OUTPUT = ROOT / 'results/visualization/ss_weight_validation_with_ss0005_seed42'
CASES = {'internal': ['R5-3', 'R5-4', 'R5-5'], 'external': ['R5-13', 'R5-14']}
TARGETS = ['B35_H2S', 'B35_SO2']
MODELS = {'ss0001': 0.001, 'ss0005': 0.005, 'ss001': 0.01, 'ss005': 0.05}
STYLES = [dict(color='#D55E00', ls='-', marker='o'),
          dict(color='#0072B2', ls=':', marker='s'),
          dict(color='#009E73', ls='--', marker='^'),
          dict(color='#CC79A7', ls='-.', marker='D')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inverse(values, columns, mean, std):
    shape = values.shape
    frame = pd.DataFrame(values.reshape(-1, len(columns)), columns=columns)
    frame = utils.inverse_zscore(frame, mean[columns], std[columns])
    return utils.inverse_log_transform(frame, TARGETS).to_numpy().reshape(shape)


def scores(truth, pred):
    assert np.isfinite(truth).all() and np.isfinite(pred).all()
    return utils.calculate_metrics(truth, pred)


def save_figure(fig, name):
    fig.canvas.draw()
    for ax in fig.axes:
        low, high = ax.get_ylim()
        for line in ax.lines:
            values = np.asarray(line.get_ydata(), dtype=float)
            assert np.isfinite(values).all()
            assert ((values >= low) & (values <= high)).all(), name
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def draw(values, horizons, overlay):
    for partition, cases in CASES.items():
        for kind in ['rmse', 'horizon', 'TS']:
            fig, axes = plt.subplots(2, len(cases), figsize=(4.5 * len(cases), 7.7), squeeze=False)
            for i, target in enumerate(TARGETS):
                for j, case in enumerate(cases):
                    ax = axes[i, j]
                    if kind == 'TS':
                        truth = overlay[('true', case)]
                        x = (np.arange(len(truth)) + 18) * 10
                        ax.plot(x, truth[:, i], color='black', lw=2, label='True')
                    for (alias, weight), style in zip(MODELS.items(), STYLES):
                        label = f'SS = {weight:g}'
                        if kind == 'TS':
                            ax.plot(x, overlay[(alias, case)][:, i], label=label, lw=1.5,
                                    **{k: v for k, v in style.items() if k != 'marker'})
                        elif kind == 'horizon':
                            rows = horizons[(horizons.model == alias) & (horizons.case == case) & (horizons.target == target)]
                            ax.plot(rows.horizon, rows.RMSE, label=label, lw=1.7, markevery=3, ms=5, **style)
                            ax.set_xticks([1, 6, 12, 18])
                        else:
                            rows = values[(values.model == alias) & (values.case == case) & (values.target == target)].set_index('evaluation')
                            ax.plot(range(3), rows.loc[['H1', 'H18', 'TS'], 'RMSE'], label=label, lw=1.7, ms=6, **style)
                            ax.set_xticks(range(3), ['H1', 'H18', 'TS'])
                    species = r'H$_2$S' if i == 0 else r'SO$_2$'
                    ax.set_title(f'{case}{"*" if case == "R5-5" else ""} | {species}', fontweight='bold')
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.grid(axis='y', alpha=.22)
                    ax.margins(y=.13)
                    if j == 0:
                        ax.set_ylabel(('Concentration' if kind == 'TS' else 'RMSE') + ' (kmol/kmol)')
                    if i == 1:
                        ax.set_xlabel({'TS': 'Elapsed time from case start (min)', 'horizon': 'Prediction horizon (10 min/step)', 'rmse': 'Evaluation'}[kind])
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=4, frameon=False)
            title = {'rmse': 'H1 / H18 / TS RMSE', 'horizon': 'Horizon-wise RMSE', 'TS': 'Recursive time-series predictions'}[kind]
            fig.suptitle(f'{title}: {partition} validation | Gain = 0.05 | seed42', fontsize=14, fontweight='bold', y=.99)
            note = 'Single seed; individual cases, no case averaging or uncertainty bars. Panel y-axis ranges vary.'
            if partition == 'internal':
                note += '\n* R5-5 retains its previous internal label; some operating points lie outside the air2/T2 core.'
            fig.text(.5, .018, note, ha='center', fontsize=8.5)
            fig.subplots_adjust(top=.81, bottom=.15, hspace=.38, wspace=.35)
            save_figure(fig, f'ss_weight_{kind}_{partition}_validation')


def main():
    assert not OUTPUT.exists(), f'Preserve existing output: {OUTPUT}'
    OUTPUT.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 10, 'pdf.fonttype': 42})
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protected = {}
    def protect(path):
        protected[str(path.relative_to(ROOT))] = sha(path)
    for path in (ROOT / 'src').rglob('*.py'):
        protect(path)
    protect(ROOT / 'scripts/prediction/predict.py')
    protect(Path(__file__).resolve())
    raw, clean = {}, {}
    audits = []
    for partition, cases in CASES.items():
        for case in cases:
            suffix = '.xlsx' if partition == 'internal' else '.csv'
            path = ROOT / 'data/Claus_dynamic' / f'Test_dataform_change_air2_R=5-{case.split("-")[1]}{suffix}'
            protect(path)
            raw[case] = pd.read_excel(path) if suffix == '.xlsx' else pd.read_csv(path)
            clean[case] = raw[case].rolling(10, min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
            inside = raw[case].air2_SP.between(140, 300) & raw[case].HEATER2_output_T_SP.between(140, 240)
            audits.append(dict(partition=partition, case=case, source=str(path.relative_to(ROOT)), raw_rows=len(raw[case]), processed_rows=len(clean[case]), TS_points=len(clean[case])-18, horizon_windows=len(clean[case])-35, within_air2_T2_core_percent=100*inside.mean()))
    records, horizon_records, overlay, checks = [], [], {}, []
    prior_path = ROOT / 'results/visualization/ss_weight_validation_seed42/validation_metrics.csv'
    protect(prior_path)
    prior = pd.read_csv(prior_path)
    for alias, weight in MODELS.items():
        config_path = ROOT / f'configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_{alias}_seed42.yaml'
        protect(config_path)
        config = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
        exp = config['exp_name']
        assert config['training']['ss_loss_weight'] == weight
        assert config['training']['gain_loss_weight'] == .05 and config['training']['seed'] == 42
        assert config['window']['prediction_length'] == 18 and config['window']['train_window_mins'] == 180
        assert config['window']['sampling_interval_min'] == 10 and config['window']['use_median_downsampling']
        de, ys, _, en = variable_selection(config['data']['variables_num'])
        assert en == de + ys
        config['data'].update(num_en_input=len(en), num_de_input=len(de), num_output=len(ys))
        paths = [ROOT / 'saved_models' / f'{exp}.pth'] + [experiment_results_dir(exp) / name for name in ['zscore_mean.csv', 'zscore_std.csv']]
        for path in paths:
            protect(path)
        mean, std = [pd.read_csv(path, index_col=0).squeeze('columns') for path in paths[1:]]
        model = get_model(config).to(device)
        model.load_state_dict(torch.load(paths[0], map_location=device, weights_only=True))
        model.eval()
        indices = [ys.index(t) for t in TARGETS]
        for partition, cases in CASES.items():
            for case in cases:
                print(f'[Evaluate] SS={weight} {case} ({device})', flush=True)
                z = utils.apply_zscore(utils.apply_log_transform(clean[case], TARGETS), mean, std)
                assert np.isfinite(z[en].to_numpy()).all()
                assert (clean[case][TARGETS] > 0).all().all()
                outdir = OUTPUT / 'predictions' / alias / case
                outdir.mkdir(parents=True, exist_ok=True)
                cache = outdir / 'predictions.npz'
                if cache.exists():
                    cached = np.load(cache)
                    assert str(cached['checkpoint_sha256']) == sha(paths[0])
                    tp, tt, hp, ht = [cached[k] for k in ['TS_pred', 'TS_true', 'horizon_pred', 'horizon_true']]
                else:
                    history = torch.tensor(z[en].iloc[:18].to_numpy(), dtype=torch.float32).unsqueeze(0)
                    future = torch.tensor(z[de].iloc[18:].to_numpy(), dtype=torch.float32).unsqueeze(0)
                    pred = native.predict_sliding_window(model, history, future, device, len(ys)).cpu().numpy()[0]
                    tp = inverse(pred, ys, mean, std)[:, indices]
                    tt = inverse(z[ys].iloc[18:].to_numpy(), ys, mean, std)[:, indices]
                    ds = MultiStepS2SDataset(z, en, de, ys, 18, 18)
                    preds, truths = [], []
                    with torch.no_grad():
                        for history, future, truth in torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False):
                            history, future = history.to(device), future.to(device)
                            steps = []
                            for step in range(18):
                                control = future[:, step:step+1]
                                result = model(history, control)
                                steps.append(result)
                                history = torch.cat([history[:, 1:], torch.cat([control, result], dim=2)], dim=1)
                            preds.append(torch.cat(steps, dim=1).cpu().numpy()[..., indices])
                            truths.append(truth.numpy()[..., indices])
                    hp = inverse(np.concatenate(preds), TARGETS, mean, std)
                    ht = inverse(np.concatenate(truths), TARGETS, mean, std)
                    np.savez_compressed(cache, TS_pred=tp, TS_true=tt, horizon_pred=hp, horizon_true=ht, checkpoint_sha256=sha(paths[0]))
                assert tp.shape == tt.shape == (len(clean[case])-18, 2)
                assert hp.shape == ht.shape == (len(clean[case])-35, 18, 2)
                np.testing.assert_allclose(tt, clean[case][TARGETS].iloc[18:], rtol=1e-12, atol=1e-14)
                if ('true', case) in overlay:
                    np.testing.assert_allclose(tt, overlay[('true', case)], rtol=1e-12, atol=1e-14)
                overlay[('true', case)], overlay[(alias, case)] = tt, tp
                pd.DataFrame({'elapsed_min': (np.arange(len(tp))+18)*10, 'H2S_true': tt[:, 0], 'H2S_pred': tp[:, 0], 'SO2_true': tt[:, 1], 'SO2_pred': tp[:, 1]}).to_csv(outdir / 'time_series.csv', index=False)
                for i, target in enumerate(TARGETS):
                    base = dict(partition=partition, case=case, model=alias, ss_weight=weight, target=target)
                    records.append(dict(base, evaluation='TS', n=len(tp), **scores(tt[:, i], tp[:, i])))
                    for h in range(18):
                        metrics = scores(ht[:, h, i], hp[:, h, i])
                        horizon_records.append(dict(base, horizon=h+1, n=len(hp), **metrics))
                        if h in [0, 17]:
                            records.append(dict(base, evaluation=f'H{h+1}', n=len(hp), **metrics))
                if alias == 'ss0005' and case == 'R5-3':
                    native.analyze_horizon_performance(model, z, config, str(outdir), mean, std, TARGETS, en, de, ys, 18, device)
                    reference = pd.read_csv(outdir / 'horizon_metrics_t1_to_t18.csv')
                    for _, row in reference.iterrows():
                        current = next(r for r in horizon_records if r['model'] == alias and r['case'] == case and r['target'] == row.Variable and r['horizon'] == row.Horizon)
                        np.testing.assert_allclose(current['RMSE'], row.RMSE, rtol=1e-9, atol=1e-12)
                        checks.append(dict(check='native_horizon', case=case, target=row.Variable, evaluation=f'H{row.Horizon}', current_RMSE=current['RMSE'], reference_RMSE=row.RMSE))
    values, horizons = pd.DataFrame(records), pd.DataFrame(horizon_records)
    assert len(values) == 120 and len(horizons) == 720
    for _, row in values[values.model.ne('ss0005')].iterrows():
        reference = prior[(prior.model == row.model) & (prior.case == row.case) & (prior.target == row.target) & (prior.evaluation == row.evaluation)]
        assert len(reference) == 1
        expected = float(reference.iloc[0].RMSE)
        np.testing.assert_allclose(row.RMSE, expected, rtol=1e-5, atol=1e-10)
        checks.append(dict(check='previous_three_SS_validation', case=row.case, target=row.target, evaluation=row.evaluation, current_RMSE=row.RMSE, reference_RMSE=expected))
    values.to_csv(OUTPUT / 'validation_metrics.csv', index=False)
    horizons.to_csv(OUTPUT / 'horizon_metrics.csv', index=False)
    summary = values.groupby(['partition', 'model', 'ss_weight', 'target', 'evaluation'], sort=False).agg(cases=('case', 'size'), mean_case_RMSE=('RMSE', 'mean')).reset_index()
    summary.to_csv(OUTPUT / 'validation_summary.csv', index=False)
    grid = summary.pivot(index=['partition', 'target', 'evaluation'], columns='ss_weight', values='mean_case_RMSE')
    assert grid.shape == (12, 4) and (grid > 0).all().all()
    normalized = grid.div(grid.min(axis=1), axis=0)
    normalized.to_csv(OUTPUT / 'normalized_12_criteria.csv')
    overall = pd.DataFrame({'mean_normalized_RMSE': normalized.mean(), 'geometric_normalized_RMSE': np.exp(np.log(normalized).mean()), 'best_criteria': (grid.rank(axis=1, method='min') == 1).sum(), 'worst_normalized_RMSE': normalized.max()}).sort_values('mean_normalized_RMSE')
    overall.to_csv(OUTPUT / 'overall_comparison.csv')
    pd.DataFrame(audits).to_csv(OUTPUT / 'data_audit.csv', index=False)
    pd.DataFrame(checks).to_csv(OUTPUT / 'metric_recheck.csv', index=False)
    draw(values, horizons, overlay)
    assert all(sha(ROOT / path) == digest for path, digest in protected.items())
    report = dict(passed=True, date='2026-11-03', cases=CASES, ss_weights=list(MODELS.values()), gain_weight=.05, seed=42, device=device, optimizer_updates=0, selected_metric_rows=len(values), horizon_metric_rows=len(horizons), reference_metric_checks=len(checks), protected_inputs_unchanged=True, protected_sha256=protected, plot_data_bounds_checked=True, visual_review='pending', preprocessing='10-row non-overlapping median; saved per-model training scaling; log H2S/SO2', TS='True history only for first 18 steps, then recursively feed predictions until case end; known future decoder inputs', horizon='True 18-step history reinitialized at each window; recursively predict 18 steps; 109 windows per case', summary='Equal-case arithmetic mean RMSE, not pooled RMSE', data_roles='Same previously designated validation trajectories; R5-5 partly outside core; no claim of new unseen test data')
    (OUTPUT / 'evaluation_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    protocol = dict(cases=CASES, ss_weights=list(MODELS.values()), gain_weight=.05, seed=42, optimizer_updates=0, added_model='ss0005', comparison_scope='Existing five designated validation trajectories only; original three-model output preserved', aggregate='Equal-case mean RMSE within each partition', overall_score='12 equal-weight criteria = 2 partitions x 2 targets x H1/H18/TS; normalize each mean-case RMSE by the best of these four SS candidates, then average; descriptive only')
    (OUTPUT / 'evaluation_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(overall.to_string(), flush=True)
    print(summary[summary.evaluation.eq('TS')].to_string(index=False), flush=True)
    print(f'PASS: 120 selected metrics, 720 horizon metrics, {len(checks)} reference checks; {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
