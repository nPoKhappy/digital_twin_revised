"""Recompute test results for four validation-selected settings and all seeds."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.experiment_paths import experiment_results_dir
from src.models import get_model
from scripts.evaluation import evaluate_gain_in_out_distribution as native
from scripts.prediction.validate_hyperparameters import (
    KEYS, TARGETS, evaluate_case, independent_rmse_rows, metric_rows, sha, write_json,
)
from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_staging'
VALIDATION = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
INTERNAL = ['R5', 'R5-1', 'R5-2', 'R5-6']
EXTERNAL = [f'R5-{i}' for i in range(19, 27)]
CASES = INTERNAL + EXTERNAL
PROBES = {d: ROOT / f'data/Claus_steady_state/lhs_test_{d}.xlsx' for d in ('in_range', 'soft_ood')}
FAMILIES = ('gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')


def case_path(case: str) -> Path:
    stem = case.replace('R5', '5')
    return ROOT / f'data/Claus_dynamic/Test_dataform_change_air2_R={stem}{"_test_split" if case in INTERNAL else ""}.csv'


def geometric(values):
    v = np.asarray(values, dtype=float)
    if not (np.isfinite(v).all() and (v > 0).all()):
        raise ValueError('Geometric-mean input is not finite and positive')
    return float(np.exp(np.log(v).mean()))


def selection_before_test(plan_path):
    """Freeze the validation-only plan and all its corresponding training runs."""
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    aliases = plan['aliases']
    if set(aliases) != set(FAMILIES):
        raise ValueError('The selection plan must specify exactly four model families')
    valid = json.loads((VALIDATION / 'analysis_validation.json').read_text(encoding='utf-8'))
    if not valid.get('passed') or valid.get('model_count') != 63:
        raise ValueError('Validation analysis has not passed')
    means = pd.read_csv(VALIDATION / 'scores_by_family_model.csv')
    jobs = json.loads(MANIFEST.read_text(encoding='utf-8'))
    selected = []
    for family in FAMILIES:
        ranked = means[means.family.eq(family)]
        matching_winners = ranked[ranked.model.eq(aliases[family])]
        if len(ranked) != 5 or len(matching_winners) != 1 or int(matching_winners.iloc[0].n_seeds) != 3:
            raise ValueError(f'Unexpected validation family coverage: {family}')
        winner = matching_winners.iloc[0]
        for seed in (42, 43, 44):
            selected.append(dict(family=family, alias=winner.model, seed=seed,
                                 role='validation_selected', validation_mean=float(winner['mean'])))
    selected.insert(0, dict(family='no_gain', alias='no_gain', seed=42, role='fixed_reference', validation_mean=None))
    for seed in (43, 44):
        selected.append(dict(family='no_gain', alias='no_gain', seed=seed,
                             role='additional_reference_seed', validation_mean=None))
    for row in selected:
        matching = [j for j in jobs if j['seed'] == row['seed'] and j['family'] == row['family'] and j['alias'] == row['alias']]
        if len(matching) != 1:
            raise ValueError(f'No unique training config for {row}')
        row.update({k: matching[0][k] for k in ('config', 'config_sha256', 'exp_name', 'checkpoint', 'results', 'cli_args') if k in matching[0]})
        row['model'] = f"{row['alias']}_seed{row['seed']}"
    if len(selected) != 15 or len({r['model'] for r in selected}) != 15:
        raise ValueError('Expected 15 unique models')
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUT / 'selected_models.json'
    record = dict(selection_rule=plan['selection_rule'], selection_plan=plan,
                  selection_plan_sha256=sha(plan_path),
                  validation_baseline='No-Gain seed 42 matched by case, target, evaluation',
                  selected_models=selected,
                  validation_source_sha256={p.name: sha(p) for p in [VALIDATION/'analysis_validation.json', VALIDATION/'scores_by_family_model.csv', VALIDATION/'scores_by_model_seed.csv', MANIFEST]})
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding='utf-8')) != record:
            raise ValueError('Validation selection differs from previous run')
    else:
        write_json(selection_path, record)
    return selected


def preflight(selected, plan_path):
    configs = {}
    sources = [Path(__file__), MANIFEST,
               VALIDATION/'analysis_validation.json', VALIDATION/'scores_by_family_model.csv', VALIDATION/'scores_by_model_seed.csv',
               ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
               ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv', ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv',
               ROOT/'scripts/evaluation/evaluate_gain_in_out_distribution.py', ROOT/'scripts/prediction/validate_hyperparameters.py',
               ROOT/'scripts/prediction/validate_gain_ss_hyperparameters.py', ROOT/'scripts/prediction/predict.py',
               ROOT/'scripts/training/train_pgin_stepwise_gain.py', *list((ROOT/'src').rglob('*.py')),
               *[case_path(case) for case in CASES], *PROBES.values()]
    for row in selected:
        config_path = ROOT / row['config']
        cfg = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
        if sha(config_path).lower() != row['config_sha256'].lower() or cfg['exp_name'] != row['exp_name'] or cfg['training']['seed'] != row['seed']:
            raise ValueError(f'Unexpected config {row["model"]}')
        native.model_contract(cfg)
        batch_path = ROOT / f"results/pgin/retrain_lhs10_23/seed{row['seed']}/_batch_run.json"
        batch = json.loads(batch_path.read_text(encoding='utf-8-sig'))
        matches = [x for x in batch['completed'] if x['exp_name'] == row['exp_name']]
        if batch['status'] != 'complete' or len(matches) != 1:
            raise ValueError(f'No completed training record: {row["model"]}')
        for path, key in [(ROOT/row['checkpoint'], 'checkpoint_sha256'),
                          (experiment_results_dir(cfg)/'zscore_mean.csv', 'zscore_mean_sha256'),
                          (experiment_results_dir(cfg)/'zscore_std.csv', 'zscore_std_sha256')]:
            if sha(path).lower() != matches[0][key].lower():
                raise ValueError(f'Training output hash differs: {path}')
            sources.append(path)
        for path in [ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
                     ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv',
                     ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv']:
            expected = batch['teacher_and_source_sha256'].get(str(path))
            if not expected or sha(path).lower() != expected.lower():
                raise ValueError(f'Teacher differs from training batch: {path}')
        sources += [config_path, batch_path]
        configs[row['model']] = cfg
    base = configs['no_gain_seed42']
    contract = native.model_contract(base)
    for name, cfg in configs.items():
        native.assert_same_contract(contract, native.model_contract(cfg), name)
    native.assert_same_scaling([(r['config'], configs[r['model']]) for r in selected], contract[0]+contract[1]+contract[2])
    protocol = dict(selection_sha256=sha(OUT/'selected_models.json'),
                    sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    selected_models=selected, internal_test=INTERNAL, external_test=EXTERNAL,
                    steady_test={k: str(v.relative_to(ROOT)) for k, v in PROBES.items()},
                    preprocessing='10-row median; saved model scaler; 18 historical steps',
                    baseline='No-Gain seed42 test RMSE, same partition/case/target/evaluation',
                    KCI='ANN-reference micro pooled all18; teacher |gain| >= 1e-5; + and - 0.5 dynamic standard deviations; zero dynamic gain incorrect',
                    interpretation='Previously used test trajectories and probes, not a newly blind holdout; future decoder inputs are recorded.',
                    seed_scope='Shared data and ANN; three seeds measure dynamic training variability only.', optimizer_updates=0)
    p = OUT/'run_protocol.json'
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Inputs changed since previous test run')
    else:
        write_json(p, protocol)
    return configs, contract, protocol


def run(plan_path):
    selected = selection_before_test(plan_path)
    configs, contract, protocol = preflight(selected, plan_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Frozen validation selection: {len(selected)} runs; device={device}', flush=True)
    frames = {}
    audits = []
    for case in CASES:
        raw = pd.read_csv(case_path(case))
        frame = raw.rolling(10, min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        if len(frame) < 36 or not np.isfinite(frame.to_numpy()).all() or not frame[TARGETS].gt(0).all().all():
            raise ValueError(f'Invalid test case: {case}')
        frames[case] = frame
        inside = raw.air2_SP.between(140, 300) & raw.HEATER2_output_T_SP.between(140, 240)
        audits.append(dict(case=case, partition='internal' if case in INTERNAL else 'external',
                           source=str(case_path(case).relative_to(ROOT)), raw_rows=len(raw), processed_rows=len(frame),
                           core_percent=float(100*inside.mean())))
    pd.DataFrame(audits).to_csv(OUT/'data_audit.csv', index=False)
    steady = {}
    for distribution, expected in [('in_range', 4385), ('soft_ood', 2692)]:
        frame = native.load_steady_state_source(str(PROBES[distribution]), contract[-1], log_target_cols=native.FULL_TARGET_COLS)
        frame = native.validate_distribution_file(frame, distribution).reset_index(drop=True)
        if len(frame) != expected: raise ValueError(f'Unexpected {distribution} test probe count: {len(frame)}')
        steady[distribution] = frame
    teacher = native.load_teacher(device)
    de, ys, en, targets, mvs, _ = contract
    teacher_gain = {}
    for name, frame in steady.items():
        teacher_gain[name] = np.concatenate([native.teacher_gain_matrix(frame.iloc[i:i+128], *teacher, targets, mvs, device).cpu().numpy() for i in range(0, len(frame), 128)])
        if not np.isfinite(teacher_gain[name]).all(): raise ValueError('Nonfinite teacher gradient')
    for number, row in enumerate(selected, 1):
        name = row['model']; cfg = configs[name]; dest = OUT/'model_data'/name
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest/'complete.json'
        if marker.exists():
            previous = json.loads(marker.read_text(encoding='utf-8'))
            if previous['protocol_sha256'] != sha(OUT/'run_protocol.json') or any(sha(dest/n) != h for n,h in previous['outputs'].items()):
                raise ValueError(f'Invalid cached output for {name}')
            print(f'[{number}/15] {name}: cached and verified', flush=True)
            continue
        print(f'[{number}/15] {name}: dynamic test', flush=True)
        mean, std = native.load_model_stats(cfg['exp_name'], cfg)
        model = get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/row['checkpoint'], map_location=device, weights_only=True)); model.eval()
        metrics, horizons, rechecks = [], [], []
        for case, frame in frames.items():
            arrays = evaluate_case(model, frame, cfg, mean, std, device)
            np.savez_compressed(dest/f'{case}_prediction.npz', **arrays)
            selected_rows, horizon_rows = metric_rows(arrays, name, case, 'internal' if case in INTERNAL else 'external')
            metrics.extend(selected_rows); horizons.extend(horizon_rows)
            rechecks.extend(independent_rmse_rows(arrays, name, case, 'internal' if case in INTERNAL else 'external'))
            pd.DataFrame({'time_min': np.arange(18, len(frame))*10,
                          **{f'{target}_true': arrays['ts_true'][:, j] for j,target in enumerate(TARGETS)},
                          **{f'{target}_pred': arrays['ts_pred'][:, j] for j,target in enumerate(TARGETS)}}).to_csv(dest/f'{case}_time_series.csv', index=False)
        metric = pd.DataFrame(metrics); check = pd.DataFrame(rechecks)
        pair = check.merge(metric, on=['model']+KEYS, suffixes=('_direct','_metric'), validate='one_to_one')
        if len(metric) != 72 or len(horizons) != 432 or len(check) != 456 or len(pair) != 72:
            raise AssertionError(f'Unexpected dynamic metric coverage: {name}')
        np.testing.assert_allclose(pair.RMSE_direct, pair.RMSE_metric, rtol=1e-12)
        metric.to_csv(dest/'test_metrics.csv', index=False)
        pd.DataFrame(horizons).to_csv(dest/'horizon_metrics.csv', index=False)
        check.to_csv(dest/'RMSE_recheck.csv', index=False)
        details = []
        for distribution, frame in steady.items():
            counts = native.empty_counts(targets, mvs); values = []
            for i in range(0, len(frame), 128):
                batch = frame.iloc[i:i+128].reset_index(drop=True)
                reference = torch.from_numpy(teacher_gain[distribution][i:i+len(batch)]).to(device)
                gain, first = native.dynamic_gain_matrix(model, batch, mean, std, de, ys, en, targets, mvs, cfg, device)
                if first != 1 or gain.shape[:2] != (2,18) or not torch.isfinite(gain).all():
                    raise ValueError('Invalid dynamic gain')
                for step in range(18): native.update_counts(counts, reference, gain[:, step], targets, mvs, 1e-5)
                values.append(gain.cpu().numpy())
                if i == 0 or i+128 >= len(frame) or i//128 % 10 == 0:
                    print(f'[{number}/15] {name}: {distribution} {min(i+128, len(frame))}/{len(frame)}', flush=True)
            arr = np.concatenate(values, axis=2)
            np.savez_compressed(dest/f'gain_{distribution}.npz', teacher_gain=teacher_gain[distribution], dynamic_gain=arr)
            independent = numpy_counts(teacher_gain[distribution], arr, 1e-5, targets, mvs)
            for target in targets:
                for mv in mvs:
                    result = native.count_row(cfg['exp_name'], distribution, target, mv, counts[(target,mv)], len(frame))
                    result.update(model=name, family=row['family'], seed=row['seed'], role=row['role'])
                    other = independent.query('target == @target and mv == @mv').iloc[0]
                    for col in ('possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain'):
                        if result[col] != other[col]: raise AssertionError(f'KCI recount differs: {name}, {col}')
                    details.append(result)
        pd.DataFrame(details).to_csv(dest/'kci_components.csv', index=False)
        outputs = {p.name: sha(p) for p in dest.iterdir() if p.is_file() and p.name != 'complete.json'}
        write_json(marker, dict(protocol_sha256=sha(OUT/'run_protocol.json'), outputs=outputs))
        del model
        gc.collect()
    summarize(selected, protocol)


def summarize(selected, protocol):
    metrics = pd.concat([pd.read_csv(OUT/'model_data'/r['model']/'test_metrics.csv') for r in selected], ignore_index=True)
    horizons = pd.concat([pd.read_csv(OUT/'model_data'/r['model']/'horizon_metrics.csv') for r in selected], ignore_index=True)
    kci = pd.concat([pd.read_csv(OUT/'model_data'/r['model']/'kci_components.csv') for r in selected], ignore_index=True)
    if (len(metrics), len(horizons), len(kci)) != (1080, 6480, 120): raise AssertionError('Aggregate coverage mismatch')
    base = metrics[metrics.model.eq('no_gain_seed42')][KEYS + ['RMSE','n']].rename(columns={'RMSE':'NoGain42_RMSE','n':'NoGain42_n'})
    pairs = metrics.merge(base, on=KEYS, validate='many_to_one')
    if len(pairs) != 1080 or not pairs.n.eq(pairs.NoGain42_n).all() or not pairs.NoGain42_RMSE.gt(0).all():
        raise AssertionError('No-Gain 42 pairing mismatch')
    pairs['relative_RMSE'] = pairs.RMSE / pairs.NoGain42_RMSE
    regional = pairs.groupby(['model','partition'], as_index=False).agg(AvgRelRMSE=('relative_RMSE',geometric), components=('relative_RMSE','size'))
    if not (regional[regional.partition.eq('internal')].components.eq(24).all() and regional[regional.partition.eq('external')].components.eq(48).all()):
        raise AssertionError('Regional relative RMSE coverage mismatch')
    counts = kci.assign(correct=kci.tp+kci.tn).groupby(['model','distribution'], as_index=False).agg(correct=('correct','sum'), valid=('valid_comparisons','sum'))
    counts['KCI_percent'] = 100*counts.correct/counts.valid
    counts['partition'] = counts.distribution.map({'in_range':'internal','soft_ood':'external'})
    coords = regional.merge(counts, on=['model','partition'], validate='one_to_one')
    info = {r['model']:r for r in selected}
    for field in ('family','alias','seed','role'):
        coords[field] = coords.model.map({name:r[field] for name,r in info.items()})
        pairs[field] = pairs.model.map({name:r[field] for name,r in info.items()})
    wide = coords.pivot(index=['model','family','alias','seed','role'], columns='partition', values='AvgRelRMSE').reset_index()
    wide['overall_50_50'] = np.sqrt(wide.internal*wide.external)
    stats = wide[wide.role.eq('validation_selected')].groupby(['family','alias'],as_index=False).agg(mean=('overall_50_50','mean'), sd=('overall_50_50','std'), seeds=('seed','nunique'))
    if not stats.seeds.eq(3).all(): raise AssertionError('Each selected method must have three seeds')
    raw = metrics.groupby(['model','partition','target','evaluation'],as_index=False).agg(mean_RMSE=('RMSE','mean'),cases=('case','nunique'))
    raw_stats = raw.merge(pd.DataFrame(selected)[['model','family','alias','seed','role']],on='model',validate='many_to_one')
    raw_seed_stats = raw_stats[raw_stats.role.eq('validation_selected')].groupby(['family','alias','partition','target','evaluation'],as_index=False).agg(mean_RMSE=('mean_RMSE','mean'),sd_RMSE=('mean_RMSE','std'),seeds=('seed','nunique'))
    kci_stats = coords[coords.role.eq('validation_selected')].groupby(['family','alias','partition'],as_index=False).agg(KCI_mean=('KCI_percent','mean'),KCI_sd=('KCI_percent','std'),seeds=('seed','nunique'))
    for frame,name in [(metrics,'test_metrics.csv'),(horizons,'horizon_metrics.csv'),(kci,'kci_components.csv'),
                       (pairs,'relative_rmse_components.csv'),(coords,'regional_coordinates.csv'),(wide,'scores_by_model_seed.csv'),
                       (stats,'scores_by_family.csv'),(raw_stats,'raw_rmse_by_model.csv'),(raw_seed_stats,'raw_rmse_by_family.csv'),
                       (kci_stats,'kci_by_family.csv')]:
        frame.to_csv(OUT/name,index=False,float_format='%.17g')
    for path,digest in protocol['sources_sha256'].items():
        if sha(ROOT/path) != digest: raise ValueError(f'Protected input changed: {path}')
    write_json(OUT/'evaluation_validation.json', dict(passed=True, model_count=15, cases=12, dynamic_metrics=len(metrics),
        horizon_metrics=len(horizons), kci_components=len(kci), test_probes=[4385,2692],
        independent_rmse_kci_recounts=True, protected_sources_unchanged=True, optimizer_updates=0))
    print('COMPLETE:', OUT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan.resolve())
