"""Evaluate all full-trajectory rawsplit models on frozen validation data."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.models import get_model
from src.experiment_paths import experiment_results_dir
from scripts.evaluation import evaluate_gain_in_out_distribution as native
from scripts.prediction.validate_hyperparameters import (
    INTERNAL, EXTERNAL, TARGETS, KEYS, evaluate_case, metric_rows,
    independent_rmse_rows, sha, write_json,
)
from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline_rawsplit'
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23_full_80_10_10/config_manifest.json'
REGIONS = {'internal': 'in_range', 'external': 'soft_ood'}
CASES = tuple(INTERNAL + EXTERNAL)
CASE_PATHS = {c: ROOT / f'data/Claus_dynamic/Test_dataform_change_air2_R=5-{c.split("-")[1]}.{ "xlsx" if c in INTERNAL else "csv"}' for c in CASES}
PROBES = {name: ROOT / f'data/Claus_steady_state/lhs_valid_{name}.xlsx' for name in REGIONS.values()}


def preflight():
    jobs = json.loads(MANIFEST.read_text(encoding='utf8'))
    if len(jobs) != 63 or len({(j['seed'], j['alias']) for j in jobs}) != 63:
        raise ValueError('Expected 63 distinct model/seed jobs')
    jobs = sorted(jobs, key=lambda j: (j['seed'] != 42 or j['alias'] != 'no_gain', j['seed'], j['alias']))
    source_paths = [MANIFEST, Path(__file__), *CASE_PATHS.values(), *PROBES.values(),
                    ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
                    ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv',
                    ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv',
                    ROOT/'scripts/evaluation/evaluate_gain_in_out_distribution.py',
                    ROOT/'scripts/prediction/validate_hyperparameters.py',
                    ROOT/'scripts/prediction/validate_gain_ss_hyperparameters.py',
                    ROOT/'scripts/prediction/predict.py', *sorted((ROOT/'src').rglob('*.py'))]
    batches = {}
    for seed in (42,43,44):
        batch = ROOT/f'results/pgin/retrain_lhs10_23_full_80_10_10/seed{seed}/_batch_run.json'
        record = json.loads(batch.read_text(encoding='utf8'))
        if record['status'] != 'complete' or len(record['completed']) != 21:
            raise ValueError(f'Incomplete training batch {seed}')
        for name, digest in record['sources_sha256'].items():
            if not Path(name).is_file() or sha(name).lower() != digest.lower():
                raise ValueError(f'Training source changed: {name}')
        batches[seed] = {entry['job']['alias']: entry for entry in record['completed']}
        source_paths.append(batch)
    configs = {}
    for j in jobs:
        entry = batches[j['seed']].get(j['alias'])
        if not entry or entry['job'] != j:
            raise ValueError(f'Training roster mismatch {j["alias"]}/{j["seed"]}')
        for name, digest in entry['outputs_sha256'].items():
            if sha(ROOT/name).lower() != digest.lower():
                raise ValueError(f'Training artifact changed: {name}')
            source_paths.append(ROOT/name)
        p=ROOT/j['config']
        if sha(p) != j['config_sha256']:
            raise ValueError(f'Configuration changed: {p}')
        cfg=yaml.safe_load(p.read_text(encoding='utf8'))
        if cfg['exp_name'] != j['exp_name'] or experiment_results_dir(cfg) != ROOT/j['results']:
            raise ValueError('Config/output mismatch')
        configs[(j['seed'],j['alias'])]=cfg
        source_paths.append(p)
    contract=native.model_contract(configs[(42,'no_gain')])
    for key,cfg in configs.items(): native.assert_same_contract(contract,native.model_contract(cfg),str(key))
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    protocol={'jobs':jobs,'sources_sha256':sources,'internal_cases':INTERNAL,'external_cases':EXTERNAL,
              'probes':{n:str(p.relative_to(ROOT)) for n,p in PROBES.items()},
              'probes_used':{'in_range':4404,'soft_ood':2585},'baseline':'new rawsplit No-Gain seed42',
              'metrics':'H1/H18/TS; 10-row median; same-case No-Gain seed42 baseline',
              'kci':'ANN reference; both directions, all18; abs teacher gain >= 1e-5; dynamic zero incorrect',
              'optimizer_updates':0}
    return jobs,configs,contract,protocol


def run():
    jobs,configs,contract,protocol=preflight()
    OUT.mkdir(parents=True,exist_ok=True)
    proto=OUT/'run_protocol.json'
    if proto.exists():
        if json.loads(proto.read_text(encoding='utf8')) != protocol: raise ValueError('Validation inputs changed')
    else: write_json(proto,protocol)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Validated 63 training records; device={device}',flush=True)
    frames={}
    for case,path in CASE_PATHS.items():
        raw=pd.read_excel(path) if path.suffix=='.xlsx' else pd.read_csv(path)
        frame=raw.rolling(10,min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        if len(frame)<36 or not np.isfinite(frame.to_numpy()).all() or not frame[TARGETS].gt(0).all().all(): raise ValueError(f'Invalid case {case}')
        frames[case]=frame
    steady={}
    for name,path in PROBES.items():
        frame=native.load_steady_state_source(str(path),contract[-1],log_target_cols=native.FULL_TARGET_COLS)
        frame=native.validate_distribution_file(frame,name).reset_index(drop=True)
        if len(frame)!=protocol['probes_used'][name]: raise ValueError(f'Wrong probe count: {name} {len(frame)}')
        steady[name]=frame
    teacher=native.load_teacher(device)
    de,ys,en,targets,mvs,_=contract
    teacher_gain={name:np.concatenate([native.teacher_gain_matrix(frame.iloc[i:i+128],*teacher,targets,mvs,device).cpu().numpy() for i in range(0,len(frame),128)]) for name,frame in steady.items()}
    for number,j in enumerate(jobs,1):
        seed,alias=j['seed'],j['alias']; cfg=configs[(seed,alias)]
        dest=OUT/'model_data'/f'seed{seed}'/alias; dest.mkdir(parents=True,exist_ok=True)
        marker=dest/'complete.json'
        if marker.exists():
            cached=json.loads(marker.read_text(encoding='utf8'))
            if cached['run_protocol_sha256']!=sha(proto) or any(sha(dest/n)!=h for n,h in cached['outputs'].items()): raise ValueError('Invalid cached output')
            print(f'[{number}/63] seed={seed} {alias}: cache',flush=True); continue
        print(f'[{number}/63] seed={seed} {alias}: dynamic validation',flush=True)
        mean,std=native.load_model_stats(cfg['exp_name'],cfg)
        model=get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/j['checkpoint'],map_location=device,weights_only=True));model.eval()
        rows=[]; checks=[]
        for case,frame in frames.items():
            arrays=evaluate_case(model,frame,cfg,mean,std,device)
            np.savez_compressed(dest/f'{case}_prediction.npz',**arrays)
            partition='internal' if case in INTERNAL else 'external'
            selected,_=metric_rows(arrays,alias,case,partition); rows.extend(selected)
            checks.extend(independent_rmse_rows(arrays,alias,case,partition))
        metrics=pd.DataFrame(rows); check=pd.DataFrame(checks).merge(metrics,on=['model']+KEYS,suffixes=('_direct','_reported'),validate='one_to_one')
        if len(metrics)!=36 or len(check)!=36: raise ValueError('Wrong dynamic metric count')
        np.testing.assert_allclose(check.RMSE_direct,check.RMSE_reported,rtol=1e-12)
        metrics.to_csv(dest/'validation_metrics.csv',index=False)
        details=[]
        for name,frame in steady.items():
            counts=native.empty_counts(targets,mvs); values=[]
            for i in range(0,len(frame),128):
                batch=frame.iloc[i:i+128].reset_index(drop=True)
                reference=torch.from_numpy(teacher_gain[name][i:i+len(batch)]).to(device)
                gain,first=native.dynamic_gain_matrix(model,batch,mean,std,de,ys,en,targets,mvs,cfg,device)
                if first!=1 or gain.shape[:2]!=(2,18) or not torch.isfinite(gain).all(): raise ValueError('Invalid dynamic gain')
                for step in range(18): native.update_counts(counts,reference,gain[:,step],targets,mvs,1e-5)
                values.append(gain.cpu().numpy())
            arr=np.concatenate(values,axis=2)
            np.savez_compressed(dest/f'gain_{name}.npz',teacher_gain=teacher_gain[name],dynamic_gain=arr)
            independent=numpy_counts(teacher_gain[name],arr,1e-5,targets,mvs)
            for target in targets:
                for mv in mvs:
                    result=native.count_row(cfg['exp_name'],name,target,mv,counts[(target,mv)],len(frame))
                    result.update(model=alias,family=j['family'])
                    other=independent.query('target == @target and mv == @mv').iloc[0]
                    for col in ('possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain'):
                        if result[col]!=other[col]: raise ValueError(f'KCI recount mismatch {seed}/{alias}/{name}/{col}')
                    details.append(result)
        pd.DataFrame(details).to_csv(dest/'kci_components.csv',index=False)
        files=[p for p in dest.iterdir() if p.is_file() and p.name!='complete.json']
        write_json(marker,dict(run_protocol_sha256=sha(proto),outputs={p.name:sha(p) for p in files}))
        print(f'[{number}/63] seed={seed} {alias}: complete',flush=True)
        del model; gc.collect()
    metrics=[]; kci=[]
    for j in jobs:
        d=OUT/'model_data'/f"seed{j['seed']}"/j['alias']
        a=pd.read_csv(d/'validation_metrics.csv'); a['seed']=j['seed']; metrics.append(a)
        b=pd.read_csv(d/'kci_components.csv'); b['seed']=j['seed']; kci.append(b)
    metrics=pd.concat(metrics,ignore_index=True); kci=pd.concat(kci,ignore_index=True)
    if (len(metrics),len(kci)) != (2268,504): raise ValueError('Wrong validation coverage')
    metrics.to_csv(OUT/'validation_metrics.csv',index=False,float_format='%.17g')
    kci.to_csv(OUT/'kci_components.csv',index=False,float_format='%.17g')
    for name,digest in protocol['sources_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Source changed: {name}')
    write_json(OUT/'evaluation_validation.json',dict(passed=True,model_count=63,seeds=[42,43,44],dynamic_metrics=len(metrics),kci_components=len(kci),independent_rmse_and_numpy_kci_recounts=True,source_hashes_unchanged=True,optimizer_updates=0))
    print(f'COMPLETE: {OUT}',flush=True)


if __name__=='__main__': run()
