"""Evaluate five fixed seed42 models on user-designated validation trajectories."""
from __future__ import annotations

import contextlib
import copy
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
from src import utils
from src.models import get_model
from src.variable_selection import variable_selection
from src.dataset import MultiStepS2SDataset
from scripts.prediction import predict as native
from scripts.plotting.plot_five_models_gradnorm_ratios_seed42 import MODELS, STYLES, OLD_DG, NEW_DG

OUTPUT = ROOT / 'results/validation_r5_3_4_5_r5_13_14_seed42_20261022'
FIGURES = ROOT / 'results/visualization/validation_r5_3_4_5_r5_13_14_seed42'
INTERNAL = ['R5-3', 'R5-4', 'R5-5']
EXTERNAL = ['R5-13', 'R5-14']
TARGETS = ['B35_H2S', 'B35_SO2']
CONFIGS = [
    'configs/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42.yaml',
    'configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42.yaml',
    'configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss001_seed42.yaml',
    'configs/pgin/difficult_gradnorm/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42_difficult_group_gradnorm_no_ss.yaml',
    'configs/pgin/difficult_gradnorm/difficult_gradnorm_ratio0p01_no_ss_seed42.yaml',
]
ALIASES = ['no_gain', 'gain', 'gain_ss001', 'dg_ratio0p015', 'dg_ratio0p01']
PROTECTED = {}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def protect(path):
    PROTECTED[str(Path(path).relative_to(ROOT))] = sha(path)


def prepared(raw, config):
    interval = config['data'].get('sampling_interval_min', config['window']['sampling_interval_min'])
    assert interval == 10 and config['window'].get('use_median_downsampling', True)
    clean = raw.rolling(interval, min_periods=interval).median(numeric_only=True).iloc[interval-1::interval].reset_index(drop=True)
    return clean.dropna()


def physical(values, columns, mu, sigma):
    frame = pd.DataFrame(values, columns=columns)
    return utils.inverse_log_transform(utils.inverse_zscore(frame, mu[columns], sigma[columns]), [t for t in TARGETS if t in columns]).to_numpy()


def save_figure(fig, name):
    fig.canvas.draw()
    for ax in fig.axes:
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), float)
            assert np.isfinite(y).all()
            lo, hi = ax.get_ylim()
            assert ((y >= lo) & (y <= hi)).all(), name
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES / f'{name}.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def draw_metrics(values, cases, partition):
    fig, axes = plt.subplots(2, len(cases), figsize=(4.2 * len(cases), 8), squeeze=False)
    for i, target in enumerate(TARGETS):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            for model, style in STYLES.items():
                rows = values[(values.model == model) & (values.case == case) & (values.target == target)].set_index('evaluation')
                ax.plot(range(3), rows.loc[['H1', 'H18', 'TS'], 'RMSE'], **style, label=model, lw=1.8, ms=6)
            ax.set_title(f'{case}{"*" if case == "R5-5" else ""} | {"H$_2$S" if i == 0 else "SO$_2$"}', fontweight='bold')
            ax.set_xticks(range(3), ['H1', 'H18', 'TS'])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
            ax.margins(y=.18)
            ax.grid(axis='y', alpha=.3)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0: ax.set_ylabel('Original-unit RMSE')
            if i == 1: ax.set_xlabel('Evaluation')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f'Five-model RMSE: {partition} validation (seed42)', fontsize=15, fontweight='bold', y=.99)
    note = 'User-designated partitions; original RMSE, no case averaging or error bars; panel y-axis ranges vary.'
    if partition == 'internal': note += '\n* R5-5 includes points outside the specified air2/T2 core bounds.'
    fig.text(.5, .015, note, ha='center', fontsize=8.5)
    fig.subplots_adjust(top=.79, bottom=.14, hspace=.43, wspace=.35)
    save_figure(fig, f'five_model_rmse_{partition}_validation')


def main():
    assert not (OUTPUT / 'evaluation_validation.json').exists(), 'Preserving completed evaluation.'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    assert not list(FIGURES.iterdir()), 'Preserving existing figures.'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 10, 'pdf.fonttype': 42})
    protocol = {
        'date': '2026-10-22', 'internal_validation': INTERNAL, 'external_validation': EXTERNAL,
        'models': MODELS, 'training_seed': 42, 'optimizer_updates': 0,
        'primary_interest': 'SO2 TS RMSE, with H2S tradeoffs shown separately; no automatic model replacement',
        'evaluations': ['H1', 'H18', 'TS'], 'case_aggregation': 'equal-case mean RMSE within each user-defined partition',
        'preprocessing': 'existing 10-row median downsampling then H2S/SO2 log and saved training statistics',
        'data_roles': 'User-designated validation; no claim that previously evaluated R5-13/14 are newly unseen',
        'reuse': 'Reuse existing R5-13/14 predictions; generate R5-3/4/5 predictions in this isolated output directory',
    }
    protocol_file = OUTPUT / 'evaluation_protocol.json'
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text(encoding='utf-8')) == protocol, 'Cannot resume a different protocol.'
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    for path in (ROOT / 'src').rglob('*.py'):
        protect(path)
    protect(ROOT / 'scripts/prediction/predict.py')
    source_paths = {case: ROOT / 'data/Claus_dynamic' / f'Test_dataform_change_air2_R=5-{case.split("-")[1]}.{ "xlsx" if case in INTERNAL else "csv"}' for case in INTERNAL + EXTERNAL}
    raw_data = {}
    for case, path in source_paths.items():
        protect(path)
        raw_data[case] = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path, float_precision='round_trip')
    audits, records, checks, horizon_data, overlay = [], [], [], {}, {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for (label, exp), config_path, alias in zip(MODELS.items(), CONFIGS, ALIASES):
        protect(ROOT / config_path)
        config = yaml.safe_load((ROOT / config_path).read_text(encoding='utf-8-sig'))
        assert config['exp_name'] == exp
        de, ys, _, en = variable_selection(config['data']['variables_num'])
        config['data'].update(num_en_input=len(en), num_de_input=len(de), num_output=len(ys), path=str(ROOT / 'data/Claus_dynamic'))
        config.setdefault('output', {})['results_dir'] = str(OUTPUT / 'predictions')
        config['data']['inference_files'] = [{'name': case, 'filename': source_paths[case].name, 'inference_strategy': 'sliding_window'} for case in INTERNAL]
        W = config['window']['train_window_mins'] // config['window']['sampling_interval_min']
        model_path = ROOT / 'saved_models' / f'{exp}.pth'
        mu_path, sigma_path = [ROOT / 'results' / exp / name for name in ['zscore_mean.csv', 'zscore_std.csv']]
        for path in [model_path, mu_path, sigma_path]: protect(path)
        mu = pd.read_csv(mu_path, index_col=0).squeeze('columns')
        sigma = pd.read_csv(sigma_path, index_col=0).squeeze('columns')
        model = get_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        config['exp_name'] = alias
        config['evaluation_source_experiment'] = exp
        config['evaluation_source_checkpoint'] = str(model_path.relative_to(ROOT))
        training_hashes = set()
        for file in config['data']['training_files']:
            path = ROOT / 'data/Claus_dynamic' / file
            protect(path)
            training = pd.read_csv(path)
            training_hashes.update(pd.util.hash_pandas_object(training[en].round(8), index=False).tolist())
        snapshot_dir = OUTPUT / 'run_configs'
        snapshot_dir.mkdir(exist_ok=True)
        (snapshot_dir / f'{alias}.yaml').write_text('# Evaluation snapshot only; source checkpoint is recorded below.\n' + yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        for case in INTERNAL + EXTERNAL:
            print('[Evaluate]', label, case, flush=True)
            clean = prepared(raw_data[case], config)
            assert np.isfinite(clean[en].to_numpy()).all() and (clean[TARGETS] > 0).all().all()
            z = utils.apply_zscore(utils.apply_log_transform(clean, TARGETS), mu, sigma)
            if case in INTERNAL:
                path = OUTPUT / 'predictions' / alias / case
                complete = all((path / name).exists() for name in ['prediction_results.csv', 'evaluation_metrics.csv', 'horizon_metrics_t1_to_t18.csv'])
                if not complete:
                    with (OUTPUT / 'prediction_evaluation.log').open('a', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        native.run_prediction(config, {'name': case, 'filename': source_paths[case].name, 'inference_strategy': 'sliding_window'}, model, device, mu, sigma, en, de, ys, W)
            else:
                path = ROOT / 'results' / exp / case
                for filename in ['prediction_results.csv', 'evaluation_metrics.csv', 'horizon_metrics_t1_to_t18.csv']: protect(path / filename)
            pred = pd.read_csv(path / 'prediction_results.csv', index_col=0, float_precision='round_trip')
            ts_metrics = pd.read_csv(path / 'evaluation_metrics.csv', float_precision='round_trip').set_index('Variable')
            h_metrics = pd.read_csv(path / 'horizon_metrics_t1_to_t18.csv', float_precision='round_trip').set_index(['Variable', 'Horizon'])
            true = physical(z[ys].iloc[W:].to_numpy(), ys, mu, sigma)
            assert len(pred) == len(true) and np.isfinite(pred.to_numpy()).all()
            for i, target in enumerate(ys):
                score = float(np.sqrt(np.mean((pred[target].to_numpy() - true[:, i])**2)))
                np.testing.assert_allclose(score, ts_metrics.loc[target, 'RMSE'], rtol=1e-9, atol=1e-11)
                checks.append({'model': label, 'case': case, 'target': target, 'evaluation': 'TS', 'saved_RMSE': ts_metrics.loc[target, 'RMSE'], 'recomputed_RMSE': score})
            quality_indices = [ys.index(t) for t in TARGETS]
            overlay[(label, case)] = pred[TARGETS].to_numpy()
            overlay[('True', case)] = true[:, quality_indices]
            ds = MultiStepS2SDataset(z, en, de, ys, W, config['window']['prediction_length'])
            pp, yy = [], []
            with torch.no_grad():
                for history, future, truth in torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False):
                    history, future = history.to(device), future.to(device)
                    steps = []
                    for step in range(future.shape[1]):
                        _, memory = model.encoder(history)
                        out = model.decoder(future[:, step:step+1], memory)
                        steps.append(out)
                        history = torch.cat([history[:, 1:], torch.cat([future[:, step:step+1], out], dim=2)], dim=1)
                    pp.append(torch.cat(steps, dim=1).cpu().numpy()[..., quality_indices])
                    yy.append(truth.numpy()[..., quality_indices])
            pz, yz = np.concatenate(pp), np.concatenate(yy)
            hp = np.exp(pz.astype(np.float64) * sigma[TARGETS].to_numpy() + mu[TARGETS].to_numpy())
            ht = np.exp(yz.astype(np.float64) * sigma[TARGETS].to_numpy() + mu[TARGETS].to_numpy())
            horizon_data[f'{exp}_{case}_pred'] = hp
            horizon_data[f'{exp}_{case}_true'] = ht
            for step in [1, 18]:
                for j, target in enumerate(TARGETS):
                    score = float(np.sqrt(np.mean((hp[:, step-1, j] - ht[:, step-1, j])**2)))
                    np.testing.assert_allclose(score, h_metrics.loc[(target, step), 'RMSE'], rtol=1e-5, atol=1e-10)
                    checks.append({'model': label, 'case': case, 'target': target, 'evaluation': f'H{step}', 'saved_RMSE': h_metrics.loc[(target, step), 'RMSE'], 'recomputed_RMSE': score})
            for target in TARGETS:
                for evaluation in ['H1', 'H18', 'TS']:
                    r = ts_metrics.loc[target] if evaluation == 'TS' else h_metrics.loc[(target, int(evaluation[1:]))]
                    records.append({'partition': 'internal_validation' if case in INTERNAL else 'external_validation', 'model': label, 'case': case, 'target': target, 'evaluation': evaluation, 'RMSE': float(r.RMSE), 'R2': float(r.R2)})
            if label == next(iter(MODELS)):
                raw = raw_data[case]
                inside = raw.air2_SP.between(140, 300) & raw.HEATER2_output_T_SP.between(140, 240)
                overlap = pd.util.hash_pandas_object(raw[en].round(8), index=False).isin(training_hashes)
                audits.append({'case': case, 'source': str(source_paths[case].relative_to(ROOT)), 'raw_rows': len(raw), 'processed_rows': len(clean), 'TS_points': len(pred), 'H1_H18_windows': len(ds), 'raw_rows_matching_training_63_features_round8': int(overlap.sum()), 'within_air2_T2_core_percent': float(inside.mean()*100), 'air2_min': float(raw.air2_SP.min()), 'air2_max': float(raw.air2_SP.max()), 'T2_min': float(raw.HEATER2_output_T_SP.min()), 'T2_max': float(raw.HEATER2_output_T_SP.max())})
    values = pd.DataFrame(records)
    assert len(values) == 150 and len(checks) == 1300
    values.to_csv(OUTPUT / 'five_model_validation_metrics.csv', index=False)
    values.to_csv(FIGURES / 'five_model_validation_metrics.csv', index=False)
    pd.DataFrame(checks).to_csv(OUTPUT / 'metric_recheck.csv', index=False)
    pd.DataFrame(audits).to_csv(OUTPUT / 'data_audit.csv', index=False)
    np.savez_compressed(OUTPUT / 'horizon_predictions_for_recheck.npz', **horizon_data)
    summary = values.groupby(['partition', 'model', 'target', 'evaluation'], sort=False).agg(cases=('case', 'size'), mean_case_RMSE=('RMSE', 'mean')).reset_index()
    summary.to_csv(OUTPUT / 'validation_summary.csv', index=False)
    keys = ['partition', 'case', 'target', 'evaluation']
    ratio = values[values.model == NEW_DG].merge(values[values.model == OLD_DG], on=keys, suffixes=('_ratio0p01', '_ratio0p015'), validate='one_to_one')
    ratio['change_percent'] = 100 * (ratio.RMSE_ratio0p01 / ratio.RMSE_ratio0p015 - 1)
    ratio.to_csv(OUTPUT / 'ratio0p01_vs_ratio0p015.csv', index=False)
    change = ratio.groupby(['partition', 'target', 'evaluation'], sort=False).agg(cases=('case', 'size'), mean_case_percent_change=('change_percent', 'mean'), cases_improved=('change_percent', lambda x: int((x < 0).sum()))).reset_index()
    change.to_csv(OUTPUT / 'ratio_change_summary.csv', index=False)
    draw_metrics(values, INTERNAL, 'internal')
    draw_metrics(values, EXTERNAL, 'external')
    for partition, cases in [('internal', INTERNAL), ('external', EXTERNAL)]:
        fig, axes = plt.subplots(2, len(cases), figsize=(4.2*len(cases), 8), squeeze=False)
        for i, target in enumerate(TARGETS):
            for j, case in enumerate(cases):
                ax = axes[i, j]
                x = np.arange(len(overlay[('True', case)]))*10 + 180
                ax.plot(x, overlay[('True', case)][:, i], color='black', lw=2, label='True')
                for model, style in STYLES.items():
                    ax.plot(x, overlay[(model, case)][:, i], color=style['color'], ls=style['linestyle'], lw=1.4, label=model)
                ax.set_title(f'{case} | {"H$_2$S" if i == 0 else "SO$_2$"}', fontweight='bold')
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                ax.grid(alpha=.2)
                ax.margins(y=.08)
                if j == 0: ax.set_ylabel('Original units')
                if i == 1: ax.set_xlabel('Elapsed time from case start (min)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False, fontsize=8.5)
        fig.suptitle(f'Five-model rolling predictions: {partition} validation (seed42)', fontsize=15, fontweight='bold', y=.99)
        fig.text(.5, .015, 'User-designated validation partitions; full y-axis includes truth and every prediction.', ha='center', fontsize=9)
        fig.subplots_adjust(top=.80, bottom=.13, hspace=.35, wspace=.35)
        save_figure(fig, f'five_model_TS_{partition}_validation')
    assert all(sha(ROOT / path) == value for path, value in PROTECTED.items())
    validation = {'passed': True, 'model_count': 5, 'cases': INTERNAL + EXTERNAL, 'metric_rows': 150, 'TS_metrics_recomputed': 1200, 'H1_H18_metrics_recomputed': 100, 'optimizer_updates': 0, 'protected_inputs_unchanged': True, 'protected_sha256': PROTECTED, 'visual_review': 'pending', 'plot_data_bounds_checked': True}
    (OUTPUT / 'evaluation_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    (FIGURES / 'figure_provenance.json').write_text(json.dumps({'metrics': str((OUTPUT/'five_model_validation_metrics.csv').relative_to(ROOT)), 'protocol': protocol, 'script_sha256': sha(__file__)}, indent=2), encoding='utf-8')
    print(change.to_string(index=False))
    print(summary[summary.evaluation.eq('TS')].to_string(index=False))
    print('PASS: predictions, 1300 independent metric checks, 4 PNG/PDF comparisons and input hashes.', flush=True)


if __name__ == '__main__':
    main()
