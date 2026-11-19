"""Evaluate No-Gain seeds 43/44 against the completed seed-42 test baseline."""
from __future__ import annotations

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
from scripts.prediction.compare_retrained_selected_test import (
    ROOT, INTERNAL, EXTERNAL, CASES, PROBES, case_path, geometric,
)
from scripts.prediction.validate_hyperparameters import (
    KEYS, TARGETS, evaluate_case, independent_rmse_rows, metric_rows, sha, write_json,
)
from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts

OUT = ROOT / 'results/visualization/retrain_lhs10_23_nogain_seeds43_44_test'
MAIN = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42'
BASE = MAIN / 'model_data/no_gain_seed42'
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
SEEDS = (43, 44)


def preflight():
    main_protocol_path = MAIN / 'run_protocol.json'
    main_protocol = json.loads(main_protocol_path.read_text(encoding='utf-8'))
    baseline_marker = json.loads((BASE / 'complete.json').read_text(encoding='utf-8'))
    if baseline_marker['protocol_sha256'] != sha(main_protocol_path):
        raise ValueError('Seed-42 baseline protocol hash differs')
    baseline_files = ['test_metrics.csv', 'gain_in_range.npz', 'gain_soft_ood.npz']
    baseline_files += [f'{case}_prediction.npz' for case in CASES]
    sources = [Path(__file__), Path(native.__file__), MANIFEST, main_protocol_path, BASE/'complete.json',
               ROOT/'scripts/prediction/compare_retrained_selected_test.py',
               ROOT/'scripts/prediction/validate_hyperparameters.py', ROOT/'scripts/prediction/predict.py',
               ROOT/'scripts/prediction/validate_gain_ss_hyperparameters.py',
               *list((ROOT/'src').rglob('*.py')),
               ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
               ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv',
               ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv',
               *[case_path(case) for case in CASES], *PROBES.values()]
    for name in baseline_files:
        path = BASE / name
        if sha(path) != baseline_marker['outputs'].get(name):
            raise ValueError(f'Seed-42 baseline output hash differs: {name}')
        sources.append(path)
    for path in [*[case_path(case) for case in CASES], *PROBES.values()]:
        key = str(path.relative_to(ROOT))
        if sha(path) != main_protocol['sources_sha256'].get(key):
            raise ValueError(f'Test input differs from seed-42 run: {path}')
    jobs = json.loads(MANIFEST.read_text(encoding='utf-8'))
    configs = {}
    for seed in SEEDS:
        match = [j for j in jobs if j['seed'] == seed and j['alias'] == 'no_gain' and j['family'] == 'no_gain']
        if len(match) != 1:
            raise ValueError(f'No unique No-Gain config for {seed}')
        job = match[0]
        config_path = ROOT/job['config']
        cfg = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
        if sha(config_path).lower() != job['config_sha256'].lower() or cfg['training']['seed'] != seed or cfg['exp_name'] != job['exp_name']:
            raise ValueError(f'Unexpected config for {seed}')
        native.model_contract(cfg)
        batch_path = ROOT/f'results/pgin/retrain_lhs10_23/seed{seed}/_batch_run.json'
        batch = json.loads(batch_path.read_text(encoding='utf-8-sig'))
        completed = [x for x in batch['completed'] if x['exp_name'] == job['exp_name']]
        if batch['status'] != 'complete' or len(completed) != 1:
            raise ValueError(f'No completed training record for {seed}')
        for path, key in [(ROOT/job['checkpoint'], 'checkpoint_sha256'),
                          (experiment_results_dir(cfg)/'zscore_mean.csv', 'zscore_mean_sha256'),
                          (experiment_results_dir(cfg)/'zscore_std.csv', 'zscore_std_sha256')]:
            if sha(path).lower() != completed[0][key].lower():
                raise ValueError(f'Training artifact hash differs: {path}')
            sources.append(path)
        for path in [ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
                     ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv',
                     ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv']:
            if sha(path).lower() != batch['teacher_and_source_sha256'].get(str(path), '').lower():
                raise ValueError(f'Teacher artifact differs from seed {seed} training: {path}')
        sources += [config_path, batch_path]
        configs[seed] = (job, cfg)
    contract = native.model_contract(configs[43][1])
    for seed in SEEDS:
        native.assert_same_contract(contract, native.model_contract(configs[seed][1]), str(seed))
    protocol = dict(seeds=list(SEEDS), baseline='No-Gain seed 42, matched by case/target/evaluation',
                    internal_test=INTERNAL, external_test=EXTERNAL,
                    steady_test={k:str(v.relative_to(ROOT)) for k,v in PROBES.items()},
                    selection='Additional requested baselines; no test-driven model selection.',
                    note='Previously used test cases; not a new blind holdout. Same ANN and recorded future decoder inputs.',
                    optimizer_updates=0, sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'run_protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Protected inputs changed since previous run')
    else:
        write_json(path, protocol)
    return configs, contract, protocol


def main():
    configs, contract, protocol = preflight()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'No-Gain 43/44 test preflight passed; device={device}', flush=True)
    frames = {}
    for case in CASES:
        raw = pd.read_csv(case_path(case))
        frame = raw.rolling(10, min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        if len(frame) < 36 or not np.isfinite(frame.to_numpy()).all() or not frame[TARGETS].gt(0).all().all():
            raise ValueError(f'Invalid test frame: {case}')
        frames[case] = frame
    data = {}
    for name, expected in [('in_range', 4385), ('soft_ood', 2692)]:
        df = native.load_steady_state_source(str(PROBES[name]), contract[-1], log_target_cols=native.FULL_TARGET_COLS)
        df = native.validate_distribution_file(df, name).reset_index(drop=True)
        if len(df) != expected:
            raise ValueError(f'Unexpected test probes: {name}')
        data[name] = df
    teacher = native.load_teacher(device)
    de, ys, en, targets, mvs, _ = contract
    teacher_gain = {}
    for name, frame in data.items():
        teacher_gain[name] = np.concatenate([native.teacher_gain_matrix(frame.iloc[i:i+128], *teacher, targets, mvs, device).cpu().numpy() for i in range(0,len(frame),128)])
        with np.load(BASE/f'gain_{name}.npz') as baseline:
            np.testing.assert_array_equal(teacher_gain[name], baseline['teacher_gain'])
    for seed in SEEDS:
        job, cfg = configs[seed]
        name = f'no_gain_seed{seed}'
        dest = OUT/'model_data'/name
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest/'complete.json'
        if marker.exists():
            cache = json.loads(marker.read_text(encoding='utf-8'))
            if cache['protocol_sha256'] != sha(OUT/'run_protocol.json') or any(sha(dest/n) != h for n,h in cache['outputs'].items()):
                raise ValueError(f'Invalid cached results for {name}')
            print(f'{name}: cache verified', flush=True)
            continue
        mean, std = native.load_model_stats(cfg['exp_name'], cfg)
        model = get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/job['checkpoint'], map_location=device, weights_only=True))
        model.eval()
        metrics, horizons, rechecks = [], [], []
        print(f'{name}: dynamic test', flush=True)
        for case, frame in frames.items():
            arrays = evaluate_case(model, frame, cfg, mean, std, device)
            with np.load(BASE/f'{case}_prediction.npz') as baseline:
                for key in ('ts_true', 'horizon_true'):
                    np.testing.assert_allclose(arrays[key], baseline[key], rtol=1e-12, atol=1e-14)
            np.savez_compressed(dest/f'{case}_prediction.npz', **arrays)
            partition = 'internal' if case in INTERNAL else 'external'
            selected, horizon = metric_rows(arrays, name, case, partition)
            metrics.extend(selected); horizons.extend(horizon)
            rechecks.extend(independent_rmse_rows(arrays, name, case, partition))
            pd.DataFrame({'time_min':np.arange(18,len(frame))*10,
                          **{f'{t}_true':arrays['ts_true'][:,j] for j,t in enumerate(TARGETS)},
                          **{f'{t}_pred':arrays['ts_pred'][:,j] for j,t in enumerate(TARGETS)}}).to_csv(dest/f'{case}_time_series.csv',index=False)
        metric, check = pd.DataFrame(metrics), pd.DataFrame(rechecks)
        merged = check.merge(metric, on=['model']+KEYS, suffixes=('_direct','_metric'), validate='one_to_one')
        if (len(metric),len(horizons),len(check),len(merged)) != (72,432,456,72):
            raise AssertionError('Unexpected dynamic coverage')
        np.testing.assert_allclose(merged.RMSE_direct, merged.RMSE_metric, rtol=1e-12)
        metric.to_csv(dest/'test_metrics.csv',index=False)
        pd.DataFrame(horizons).to_csv(dest/'horizon_metrics.csv',index=False)
        check.to_csv(dest/'RMSE_recheck.csv',index=False)
        details = []
        for distribution, frame in data.items():
            counts = native.empty_counts(targets,mvs); values = []
            for i in range(0,len(frame),128):
                batch = frame.iloc[i:i+128].reset_index(drop=True)
                reference = torch.from_numpy(teacher_gain[distribution][i:i+len(batch)]).to(device)
                gain, first = native.dynamic_gain_matrix(model,batch,mean,std,de,ys,en,targets,mvs,cfg,device)
                if first != 1 or gain.shape[:2] != (2,18) or not torch.isfinite(gain).all():
                    raise ValueError('Invalid dynamic gain')
                for h in range(18):
                    native.update_counts(counts,reference,gain[:,h],targets,mvs,1e-5)
                values.append(gain.cpu().numpy())
                if i == 0 or i+128 >= len(frame) or i//128 % 10 == 0:
                    print(f'{name}: {distribution} {min(i+128,len(frame))}/{len(frame)}',flush=True)
            arr = np.concatenate(values,axis=2)
            np.savez_compressed(dest/f'gain_{distribution}.npz',teacher_gain=teacher_gain[distribution],dynamic_gain=arr)
            independent = numpy_counts(teacher_gain[distribution],arr,1e-5,targets,mvs)
            for target in targets:
                for mv in mvs:
                    result = native.count_row(cfg['exp_name'],distribution,target,mv,counts[(target,mv)],len(frame))
                    result.update(model=name,family='no_gain',seed=seed)
                    other = independent.query('target == @target and mv == @mv').iloc[0]
                    for col in ('possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain'):
                        if result[col] != other[col]:
                            raise AssertionError(f'KCI recount mismatch: {name} {distribution} {col}')
                    details.append(result)
        pd.DataFrame(details).to_csv(dest/'kci_components.csv',index=False)
        outputs = {p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name != 'complete.json'}
        write_json(marker,dict(protocol_sha256=sha(OUT/'run_protocol.json'),outputs=outputs))
        del model
        gc.collect()
    summarize(protocol)


def summarize(protocol):
    metrics = pd.concat([pd.read_csv(OUT/'model_data'/f'no_gain_seed{s}'/'test_metrics.csv') for s in SEEDS],ignore_index=True)
    horizons = pd.concat([pd.read_csv(OUT/'model_data'/f'no_gain_seed{s}'/'horizon_metrics.csv') for s in SEEDS],ignore_index=True)
    kci = pd.concat([pd.read_csv(OUT/'model_data'/f'no_gain_seed{s}'/'kci_components.csv') for s in SEEDS],ignore_index=True)
    base = pd.read_csv(BASE/'test_metrics.csv')[KEYS+['RMSE','n']].rename(columns={'RMSE':'NoGain42_RMSE','n':'NoGain42_n'})
    pairs = metrics.merge(base,on=KEYS,validate='many_to_one')
    if (len(metrics),len(horizons),len(kci),len(pairs)) != (144,864,16,144) or not pairs.n.eq(pairs.NoGain42_n).all() or not pairs.NoGain42_RMSE.gt(0).all():
        raise AssertionError('No-Gain 42 pairing or coverage mismatch')
    pairs['relative_RMSE'] = pairs.RMSE/pairs.NoGain42_RMSE
    regional = pairs.groupby(['model','partition'],as_index=False).agg(AvgRelRMSE=('relative_RMSE',geometric),components=('relative_RMSE','size'))
    if not (regional[regional.partition.eq('internal')].components.eq(24).all() and regional[regional.partition.eq('external')].components.eq(48).all()):
        raise AssertionError('Unexpected region component count')
    counts = kci.assign(correct=kci.tp+kci.tn).groupby(['model','distribution'],as_index=False).agg(correct=('correct','sum'),valid=('valid_comparisons','sum'))
    counts['KCI_percent'] = 100*counts.correct/counts.valid
    counts['partition'] = counts.distribution.map({'in_range':'internal','soft_ood':'external'})
    coords = regional.merge(counts,on=['model','partition'],validate='one_to_one')
    coords['seed'] = coords.model.str.extract(r'(\d+)$').astype(int)
    wide = coords.pivot(index=['model','seed'],columns='partition',values='AvgRelRMSE').reset_index()
    wide['overall_50_50'] = np.sqrt(wide.internal*wide.external)
    for frame,name in [(metrics,'test_metrics.csv'),(horizons,'horizon_metrics.csv'),(kci,'kci_components.csv'),
                       (pairs,'relative_rmse_components.csv'),(coords,'regional_coordinates.csv'),(wide,'scores_by_model_seed.csv')]:
        frame.to_csv(OUT/name,index=False,float_format='%.17g')
    for path,digest in protocol['sources_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Protected input changed: {path}')
    validation = dict(passed=True, model_count=2, seeds=list(SEEDS), cases=12, dynamic_metrics=len(metrics),
                      horizon_metrics=len(horizons), kci_components=len(kci), relative_components=len(pairs),
                      regional_points=len(coords), truth_matches_nogain42=True, independent_rmse_kci_recounts=True,
                      protected_sources_unchanged=True, optimizer_updates=0)
    write_json(OUT/'evaluation_validation.json',validation)
    (OUT/'README.md').write_text(
        '# No-Gain seed 43 / 44 test comparison\n\n'
        'Both models use their own trained weights and scalers. Every RMSE ratio uses the matching '
        'No-Gain seed 42 test RMSE. Four internal cases contribute 24 components; eight external '
        'cases contribute 48; regional scores are geometric means and the overall score is '
        '`sqrt(internal*external)`. KCI uses test steady-state probes and the same frozen ANN. '
        'The cases and probes were used by previous experiments, so this is not a new blind holdout. '
        'All seeds share the same ANN and data; seed variation reflects dynamic training randomness.\n\n'
        '`test_metrics.csv`, `relative_rmse_components.csv`, `regional_coordinates.csv`, '
        '`scores_by_model_seed.csv`, `kci_components.csv`, and per-case NPZ predictions preserve '
        'the detailed results. No model was trained in this run.\n\n'
        'Re-run: `.\\.venv\\Scripts\\python.exe -m scripts.prediction.evaluate_nogain43_44_test`. '
        'Completed model caches and protected source hashes are checked before reuse.\n',encoding='utf-8')
    print('COMPLETE:', OUT, flush=True)
    print(wide.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
