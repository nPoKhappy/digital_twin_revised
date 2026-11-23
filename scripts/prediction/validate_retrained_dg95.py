"""Evaluate the fifteen 95%-cap Difficult-GradNorm retrains on validation.

The existing reference predictions and baseline are not modified.  This uses the
same preprocessing, teacher, KCI and metric routines as the original 63 runs.
"""
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
    INTERNAL, EXTERNAL, TARGETS, KEYS, evaluate_case, metric_rows,
    independent_rmse_rows, sha, write_json,
)
from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/visualization/retrain_lhs10_23_dg95_validation'
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
ORIGINAL_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
OLD = ROOT / 'results/visualization'
SEEDS = (42, 43, 44)
ALIASES = ('dg_ratio0p015', 'dg_ratio0p15', 'dg_ratio1p5', 'dg_ratio15', 'dg_ratio150')
REGIONS = {'internal': 'in_range', 'external': 'soft_ood'}
CASES = tuple(INTERNAL + EXTERNAL)
CASE_PATHS = {c: ROOT / f'data/Claus_dynamic/Test_dataform_change_air2_R=5-{c.split("-")[1]}.{ "xlsx" if c in INTERNAL else "csv"}' for c in CASES}
PROBES = {name: ROOT / f'data/Claus_steady_state/lhs_valid_{name}.xlsx' for name in REGIONS.values()}


def historic(seed: int) -> Path:
    return OLD / f'retrain_lhs10_23_21_models_validation_seed{seed}'


def checked_source(path: Path, source_hashes: dict, historical: dict | None = None) -> None:
    digest = sha(path)
    name = str(path.relative_to(ROOT))
    if historical is not None and historical.get(name, '').lower() != digest.lower():
        raise ValueError(f'Historical validation source differs: {name}')
    source_hashes[name] = digest


def preflight() -> tuple[list, dict, tuple, dict]:
    jobs = json.loads(MANIFEST.read_text(encoding='utf8'))
    oldjobs = json.loads(ORIGINAL_MANIFEST.read_text(encoding='utf8'))
    base = [j.copy() for j in oldjobs if j['alias'] == 'dg_ratio0p015']
    for job in base:
        job['target_ratio'] = '0.015'
        job['max_scale'] = '1000'
    jobs = base + jobs
    if len(jobs) != 15 or {(j['seed'], j['alias']) for j in jobs} != {(s, a) for s in SEEDS for a in ALIASES}:
        raise ValueError('Manifest is not the expected 15 jobs')
    expected_ratio = dict(zip(ALIASES, ('0.015', '0.15', '1.5', '15', '150')))
    expected_cap = dict(zip(ALIASES, ('1000', '20000', '500000', '10000000', '500000000')))
    sources = {}
    for p in [Path(__file__), MANIFEST, ORIGINAL_MANIFEST]:
        checked_source(p, sources)
    common = [ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
              ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv',
              ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv',
              ROOT/'scripts/evaluation/evaluate_gain_in_out_distribution.py',
              ROOT/'scripts/prediction/validate_hyperparameters.py',
              ROOT/'scripts/prediction/validate_gain_ss_hyperparameters.py',
              ROOT/'scripts/prediction/predict.py', *list((ROOT/'src').rglob('*.py')),
              *CASE_PATHS.values(), *PROBES.values()]
    for seed in SEEDS:
        d = historic(seed)
        prot = d/'run_protocol.json'
        ev = d/'evaluation_validation.json'
        checked_source(prot, sources)
        checked_source(ev, sources)
        historic_protocol = json.loads(prot.read_text(encoding='utf8'))
        validation = json.loads(ev.read_text(encoding='utf8'))
        if not validation.get('passed') or validation.get('seed') != seed or not validation.get('source_hashes_unchanged'):
            raise ValueError(f'Original validation incomplete for seed {seed}')
        for path in common:
            checked_source(path, sources, historic_protocol['sources_sha256'])
    configs = {}
    baseline_cfg = yaml.safe_load((ROOT/'configs/pgin/retrain_lhs10_23/seed42/no_gain.yaml').read_text(encoding='utf8'))
    contract = native.model_contract(baseline_cfg)
    for job in jobs:
        seed, alias, ratio = job['seed'], job['alias'], job['target_ratio']
        if ratio != expected_ratio[alias] or job['max_scale'] != expected_cap[alias] or job['cli_args'] != ['--no-ss','--target-ratio',ratio,'--max-scale',expected_cap[alias],'--use-config-steady-state-source']:
            raise ValueError(f'Wrong DG settings: {seed}, {alias}')
        checkpoint = ROOT/job['checkpoint']
        result = ROOT/job['results']
        journals = list((result.parent/'_dg_direct_retrain').glob(f'*/{alias}.json'))
        journals = [p for p in journals if json.loads(p.read_text(encoding='utf8')).get('status') == 'complete']
        if len(journals) != 1:
            raise ValueError(f'Expected one complete training journal: seed {seed} {alias}, found {len(journals)}')
        journal_path = journals[0]
        journal = json.loads(journal_path.read_text(encoding='utf8'))
        j = journal['job']
        if (journal.get('seed') != seed or journal.get('training_mode') != 'from_scratch_direct_overwrite'
                or j.get('alias') != alias or j.get('target_ratio') != ratio
                or j.get('max_scale') != expected_cap[alias] or j.get('cli_args') != job['cli_args']
                or j.get('config_sha256') != job['config_sha256']
                or journal.get('new_checkpoint_sha256') != sha(checkpoint)):
            raise ValueError(f'Training journal mismatch: {journal_path}')
        for relative, digest in journal['new_results_sha256'].items():
            path = result/relative
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f'Training output changed: {path}')
        for name, digest in journal['source_sha256'].items():
            path = Path(name)
            if not path.is_file() or sha(path).lower() != digest.lower():
                raise ValueError(f'Training source changed: {name}')
        checked_source(journal_path, sources)
        cfgpath = ROOT/job['config']
        cfg = yaml.safe_load(cfgpath.read_text(encoding='utf8'))
        if (sha(cfgpath) != job['config_sha256'] or cfg['exp_name'] != job['exp_name']
            or cfg['training']['seed'] != seed or cfg['training']['scaler_fit_scope'] != 'train_only'
            or experiment_results_dir(cfg) != result):
            raise ValueError(f'Config mismatch: {cfgpath}')
        native.assert_same_contract(contract, native.model_contract(cfg), f'{seed}/{alias}')
        for p in (cfgpath, checkpoint, result/'zscore_mean.csv', result/'zscore_std.csv'):
            checked_source(p, sources)
        configs[(seed, alias)] = cfg
    native.assert_same_scaling([(j['config'], configs[(j['seed'],j['alias'])]) for j in jobs], contract[0]+contract[1]+contract[2])
    for case in CASES:
        for name in (f'{case}_prediction.npz',):
            path = historic(42)/'model_data/no_gain'/name
            marker = json.loads((path.parent/'complete.json').read_text(encoding='utf8'))
            if marker['outputs'].get(name) != sha(path):
                raise ValueError(f'Baseline prediction hash mismatch: {path}')
            checked_source(path, sources)
        checked_source(historic(42)/'model_data/no_gain/complete.json', sources)
    protocol = dict(jobs=jobs, sources_sha256=sources, internal_cases=list(INTERNAL), external_cases=list(EXTERNAL),
                    probes={k:str(v.relative_to(ROOT)) for k,v in PROBES.items()},
                    metrics='H1/H18/TS, 10-row median preprocessing, same-case No-Gain seed42 baseline',
                    kci='ANN reference; both directions, all18; abs teacher gain >= 1e-5; dynamic zero incorrect',
                    probes_used={'in_range':4404,'soft_ood':2585}, optimizer_updates=0,
                    train_note='95th-percentile-based per-ratio max-scale; direct retrain journals verified')
    return jobs, configs, contract, protocol


def run(check_only: bool = False) -> None:
    jobs, configs, contract, protocol = preflight()
    print(f'Preflight passed: {len(jobs)} completed DG95 runs, historical validation sources verified.', flush=True)
    if check_only:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    proto = OUT/'run_protocol.json'
    if proto.exists():
        if json.loads(proto.read_text(encoding='utf8')) != protocol:
            raise ValueError('Inputs changed since incremental validation started')
    else:
        write_json(proto, protocol)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Validating on {device}', flush=True)
    frames = {}
    for case,path in CASE_PATHS.items():
        raw = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
        frame = raw.rolling(10,min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        if len(frame) < 36 or not np.isfinite(frame.to_numpy()).all() or not frame[TARGETS].gt(0).all().all():
            raise ValueError(f'Invalid case {case}')
        frames[case] = frame
    steady = {}
    for name,path in PROBES.items():
        frame = native.load_steady_state_source(str(path), contract[-1], log_target_cols=native.FULL_TARGET_COLS)
        frame = native.validate_distribution_file(frame,name).reset_index(drop=True)
        if len(frame) != protocol['probes_used'][name]:
            raise ValueError(f'Unexpected {name} probe count')
        steady[name] = frame
    teacher = native.load_teacher(device)
    de,ys,en,targets,mvs,_ = contract
    teacher_gain = {}
    for name,frame in steady.items():
        teacher_gain[name] = np.concatenate([native.teacher_gain_matrix(frame.iloc[i:i+128],*teacher,targets,mvs,device).cpu().numpy() for i in range(0,len(frame),128)])
        if not np.isfinite(teacher_gain[name]).all():
            raise ValueError('Nonfinite ANN teacher gradient')
        oldpath = historic(42)/'model_data/no_gain'/f'gain_{name}.npz'
        with np.load(oldpath) as old:
            np.testing.assert_allclose(teacher_gain[name], old['teacher_gain'], rtol=1e-5, atol=1e-10)
    for number,job in enumerate(jobs,1):
        seed,alias=job['seed'],job['alias']
        cfg=configs[(seed,alias)]
        dest=OUT/'model_data'/f'seed{seed}'/alias
        dest.mkdir(parents=True,exist_ok=True)
        marker=dest/'complete.json'
        if marker.exists():
            cached=json.loads(marker.read_text(encoding='utf8'))
            if cached['run_protocol_sha256'] != sha(proto) or any(sha(dest/n)!=h for n,h in cached['outputs'].items()):
                raise ValueError(f'Invalid cached output: {seed}/{alias}')
            print(f'[{number}/{len(jobs)}] seed {seed} {alias}: verified cache',flush=True)
            continue
        print(f'[{number}/{len(jobs)}] seed {seed} {alias}: dynamic validation',flush=True)
        mean,std=native.load_model_stats(cfg['exp_name'],cfg)
        model=get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/job['checkpoint'],map_location=device,weights_only=True));model.eval()
        rows=[]; checks=[]
        for case,frame in frames.items():
            arrays=evaluate_case(model,frame,cfg,mean,std,device)
            with np.load(historic(42)/'model_data/no_gain'/f'{case}_prediction.npz') as ref:
                for key in ('ts_true','horizon_true'):
                    np.testing.assert_allclose(arrays[key],ref[key],rtol=1e-12,atol=1e-14)
            np.savez_compressed(dest/f'{case}_prediction.npz',**arrays)
            partition='internal' if case in INTERNAL else 'external'
            selected,_=metric_rows(arrays,alias,case,partition); rows.extend(selected)
            checks.extend(independent_rmse_rows(arrays,alias,case,partition))
        metrics=pd.DataFrame(rows);check=pd.DataFrame(checks)
        check=check.merge(metrics,on=['model']+KEYS,suffixes=('_direct','_reported'),validate='one_to_one')
        if len(metrics)!=36 or len(check)!=36:
            raise ValueError('Wrong dynamic metric count')
        np.testing.assert_allclose(check.RMSE_direct,check.RMSE_reported,rtol=1e-12)
        metrics.to_csv(dest/'validation_metrics.csv',index=False)
        details=[]
        for name,frame in steady.items():
            counts=native.empty_counts(targets,mvs); values=[]
            for i in range(0,len(frame),128):
                batch=frame.iloc[i:i+128].reset_index(drop=True)
                reference=torch.from_numpy(teacher_gain[name][i:i+len(batch)]).to(device)
                gain,first=native.dynamic_gain_matrix(model,batch,mean,std,de,ys,en,targets,mvs,cfg,device)
                if first!=1 or gain.shape[:2]!=(2,18) or not torch.isfinite(gain).all():
                    raise ValueError('Invalid dynamic gain')
                for step in range(18):native.update_counts(counts,reference,gain[:,step],targets,mvs,1e-5)
                values.append(gain.cpu().numpy())
                if i==0 or i+128>=len(frame) or i//128%10==0:
                    print(f'[{number}/{len(jobs)}] {alias} seed{seed}: {name} {min(i+128,len(frame))}/{len(frame)}',flush=True)
            arr=np.concatenate(values,axis=2)
            np.savez_compressed(dest/f'gain_{name}.npz',teacher_gain=teacher_gain[name],dynamic_gain=arr)
            independent=numpy_counts(teacher_gain[name],arr,1e-5,targets,mvs)
            for target in targets:
                for mv in mvs:
                    result=native.count_row(cfg['exp_name'],name,target,mv,counts[(target,mv)],len(frame))
                    result.update(model=alias,family=job['family'])
                    other=independent.query('target == @target and mv == @mv').iloc[0]
                    for col in ('possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain'):
                        if result[col] != other[col]:
                            raise ValueError(f'KCI count mismatch {seed}/{alias}/{name}/{col}')
                    details.append(result)
        pd.DataFrame(details).to_csv(dest/'kci_components.csv',index=False)
        files=[p for p in dest.iterdir() if p.is_file() and p.name!='complete.json']
        write_json(marker,dict(run_protocol_sha256=sha(proto),outputs={p.name:sha(p) for p in files}))
        del model; gc.collect()
    metrics=[]; kci=[]
    for j in jobs:
        d=OUT/'model_data'/f"seed{j['seed']}"/j['alias']
        a=pd.read_csv(d/'validation_metrics.csv'); a['seed']=j['seed']; metrics.append(a)
        b=pd.read_csv(d/'kci_components.csv'); b['seed']=j['seed']; kci.append(b)
    metrics=pd.concat(metrics,ignore_index=True); kci=pd.concat(kci,ignore_index=True)
    if (len(metrics),len(kci))!=(540,120):
        raise ValueError('Wrong combined metric coverage')
    metrics.to_csv(OUT/'validation_metrics.csv',index=False,float_format='%.17g')
    kci.to_csv(OUT/'kci_components.csv',index=False,float_format='%.17g')
    for name,digest in protocol['sources_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'Protected source changed: {name}')
    write_json(OUT/'evaluation_validation.json',dict(passed=True,model_count=15,seeds=list(SEEDS),
        dynamic_metrics=len(metrics),kci_components=len(kci),source_hashes_unchanged=True,
        independent_rmse_and_numpy_kci_recounts=True,optimizer_updates=0))
    print(f'COMPLETE: {OUT}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    run(parser.parse_args().check_only)
