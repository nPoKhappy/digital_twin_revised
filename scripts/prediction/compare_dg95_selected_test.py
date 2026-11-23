"""Freeze 95-percentile-cap validation winners and evaluate selected test models.

Unchanged non-DG test predictions are hash checked and reused. The three
validation-selected DG models are evaluated from the new checkpoints. Output
is staged so the completed previous report remains intact until verification.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.evaluation import evaluate_gain_in_out_distribution as native
from scripts.prediction import compare_retrained_mean_tradeoff_test as base
from scripts.prediction.validate_hyperparameters import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
VIS = ROOT / 'results/visualization'
VALID = VIS / 'retrain_lhs10_23_63_models_validation_nogain42_baseline'
DG_VALID = VIS / 'retrain_lhs10_23_dg95_validation'
OLD = VIS / 'retrain_lhs10_23_selected_test_nogain42/updated_validation_best'
OUT = VIS / 'retrain_lhs10_23_selected_test_nogain42/updated_dg95_staging'
MANIFESTS = (ROOT/'configs/pgin/retrain_lhs10_23/config_manifest.json',
             ROOT/'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json')
FAMILIES = ('gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
SEEDS = (42, 43, 44)


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def select_validation():
    report = load(VALID/'analysis_validation.json')
    if not report.get('passed') or report.get('model_count') != 63:
        raise ValueError('Updated 63-model validation has not passed')
    provenance = load(VALID/'provenance.json')
    for name, digest in provenance['outputs_sha256'].items():
        if sha(VALID/name) != digest:
            raise ValueError(f'Validation output changed: {name}')
    d = pd.read_csv(VALID/'regional_coordinates_63_models.csv')
    if len(d) != 126 or d.duplicated(['model','seed','region']).any():
        raise ValueError('Invalid validation coverage')
    if not np.isfinite(d[['AvgRelRMSE','KCI_percent']]).all().all():
        raise ValueError('Nonfinite validation metrics')
    p = d.pivot(index=['model','family','seed'], columns='region', values=['AvgRelRMSE','KCI_percent'])
    p.columns = ['_'.join(x) for x in p.columns]
    p = p.reset_index()
    p['error_50_50'] = np.sqrt(p.AvgRelRMSE_internal*p.AvgRelRMSE_external)
    p['kci_50_50'] = (p.KCI_percent_internal+p.KCI_percent_external)/2
    candidates=[]
    for family in FAMILIES:
        sub=p[p.family.eq(family)].groupby('model',as_index=False).agg(
            error_mean=('error_50_50','mean'), error_sd=('error_50_50','std'),
            kci_mean_percent=('kci_50_50','mean'), kci_sd_percentage_points=('kci_50_50','std'), n_seeds=('seed','nunique'))
        if len(sub)!=5 or not sub.n_seeds.eq(3).all():
            raise ValueError(f'Unexpected family coverage: {family}')
        emin,emax=sub.error_mean.min(),sub.error_mean.max()
        kmin,kmax=sub.kci_mean_percent.min(),sub.kci_mean_percent.max()
        if not (emax>emin and kmax>kmin):
            raise ValueError(f'Cannot normalize candidates: {family}')
        sub['error_normalized']=(sub.error_mean-emin)/(emax-emin)
        sub['kci_shortfall_normalized']=(kmax-sub.kci_mean_percent)/(kmax-kmin)
        sub['distance']=np.hypot(sub.error_normalized,sub.kci_shortfall_normalized)
        sub=sub.sort_values(['distance','error_mean','model'],kind='mergesort')
        for rank,row in enumerate(sub.to_dict('records'),1):
            candidates.append({'family':family,'alias':row.pop('model'),'rank':rank,**row})
    aliases={family:next(x['alias'] for x in candidates if x['family']==family and x['rank']==1) for family in FAMILIES}
    sources=[VALID/n for n in ('analysis_validation.json','provenance.json','regional_coordinates_63_models.csv',
            'relative_rmse_components.csv','kci_components.csv','scores_by_model_seed.csv')]
    rule=('Validation only: for each seed use sqrt(internal AvgRelRMSE * external AvgRelRMSE) '
          'against No-Gain seed 42 and average internal/external KCI; average these over seeds 42/43/44. '
          'Within each family min-max normalize the five error and KCI-shortfall values and choose '
          'minimum equal-weight Euclidean distance; ties by lower error then alias.')
    return {'aliases':aliases,'selection_rule':rule,'selection_details':{'baseline':'No-Gain seed 42',
            'seeds':list(SEEDS),'validation_only':True,'family_candidate_count':5},
            'validation_sources_sha256':{str(x.relative_to(ROOT)):sha(x) for x in sources},
            'scores':candidates},candidates


def select_models(plan_path):
    plan=load(plan_path)
    if set(plan['aliases']) != set(FAMILIES):
        raise ValueError('Invalid selection plan')
    jobs0,jobs1=(load(p) for p in MANIFESTS)
    jobs=[j for j in jobs0 if j['family'] != 'difficult_gradnorm' or j['alias']=='dg_ratio0p015']+jobs1
    selected=[]
    for family in ('no_gain',*FAMILIES):
        alias='no_gain' if family=='no_gain' else plan['aliases'][family]
        for seed in SEEDS:
            match=[j for j in jobs if j['family']==family and j['alias']==alias and j['seed']==seed]
            if len(match)!=1: raise ValueError(f'No unique selected run: {family} {seed}')
            j=match[0]
            selected.append({'family':family,'alias':alias,'seed':seed,
                'role':('fixed_reference' if seed==42 else 'additional_reference_seed') if family=='no_gain' else 'validation_selected',
                'validation_mean':None,'model':f'{alias}_seed{seed}',
                **{k:j[k] for k in ('config','config_sha256','exp_name','checkpoint','results','cli_args') if k in j}})
    record={'selection_rule':plan['selection_rule'],'selection_plan':plan,
        'selection_plan_sha256':sha(plan_path),'validation_baseline':'No-Gain seed 42, same partition/case/target/evaluation',
        'selected_models':selected}
    path=OUT/'selected_models.json'
    if path.exists() and load(path)!=record: raise ValueError('Selection changed during run')
    if not path.exists(): write_json(path,record)
    return selected


def preflight(selected,plan_path):
    oldproto_raw=(OLD/'run_protocol.json').read_text(encoding='utf8')
    oldproto=json.loads(oldproto_raw)
    oldproto_sha=sha(OLD/'run_protocol.json')
    old_selection={r['model']:r for r in load(OLD/'selected_models.json')['selected_models']}
    validation_plan=load(plan_path)
    for rel,digest in validation_plan['validation_sources_sha256'].items():
        if sha(ROOT/rel)!=digest: raise ValueError(f'Validation source changed: {rel}')
    # Prior predictions may be reused only for models whose data, model,
    # scaler, evaluation code and marker are still byte-identical.
    source_paths=[ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
        ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv',ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv',
        ROOT/'scripts/evaluation/evaluate_gain_in_out_distribution.py',ROOT/'scripts/prediction/validate_hyperparameters.py',
        ROOT/'scripts/prediction/validate_gain_ss_hyperparameters.py',ROOT/'scripts/prediction/predict.py',
        *list((ROOT/'src').rglob('*.py')),*[base.case_path(c) for c in base.CASES],*base.PROBES.values()]
    for path in source_paths:
        rel=str(path.relative_to(ROOT))
        if oldproto['sources_sha256'].get(rel)!=sha(path):
            raise ValueError(f'Prior test source differs; cannot reuse caches: {rel}')
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    sources.update(validation_plan['validation_sources_sha256'])
    configs={}; reused=[]
    for row in selected:
        config=ROOT/row['config']; cfg=yaml.safe_load(config.read_text(encoding='utf8'))
        if sha(config)!=row['config_sha256'] or cfg['exp_name']!=row['exp_name'] or cfg['training']['seed']!=row['seed']:
            raise ValueError(f'Config mismatch: {row["model"]}')
        result=ROOT/row['results']; checkpoint=ROOT/row['checkpoint']
        for path in (config,checkpoint,result/'zscore_mean.csv',result/'zscore_std.csv'):
            sources[str(path.relative_to(ROOT))]=sha(path)
        configs[row['model']]=cfg
        if row['family']=='difficult_gradnorm':
            journals=list((result.parent/'_dg_direct_retrain').glob(f'*/{row["alias"]}.json'))
            matches=[p for p in journals if (lambda j: j.get('status')=='complete' and
                     j.get('new_checkpoint_sha256')==sha(checkpoint) and j.get('job',{}).get('config_sha256')==sha(config))(load(p))]
            if len(matches)!=1: raise ValueError(f'No unique completed new DG journal: {row["model"]}')
            journal=load(matches[0])
            for name in ('zscore_mean.csv','zscore_std.csv'):
                if journal['new_results_sha256'].get(name)!=sha(result/name):
                    raise ValueError(f'DG scaler mismatch: {row["model"]}/{name}')
            for absname,digest in journal['source_sha256'].items():
                if sha(Path(absname)).lower()!=digest.lower(): raise ValueError(f'DG source changed: {absname}')
            sources[str(matches[0].relative_to(ROOT))]=sha(matches[0])
        else:
            if row['model'] not in old_selection: raise ValueError(f'No prior cached non-DG test model: {row["model"]}')
            for path in (config,checkpoint,result/'zscore_mean.csv',result/'zscore_std.csv'):
                if oldproto['sources_sha256'].get(str(path.relative_to(ROOT)))!=sha(path):
                    raise ValueError(f'Non-DG training artifact differs from test cache: {path}')
            folder=OLD/'model_data'/row['model']; marker=load(folder/'complete.json')
            if marker['protocol_sha256']!=oldproto_sha: raise ValueError(f'Prior marker protocol mismatch: {row["model"]}')
            for name,digest in marker['outputs'].items():
                path=folder/name
                if sha(path)!=digest: raise ValueError(f'Prior test cache differs: {path}')
                sources[str(path.relative_to(ROOT))]=digest
            reused.append((row,folder,marker))
    if len(reused)!=12: raise ValueError('Expected 12 unchanged and 3 newly evaluated models')
    contract=native.model_contract(configs['no_gain_seed42'])
    for name,cfg in configs.items(): native.assert_same_contract(contract,native.model_contract(cfg),name)
    native.assert_same_scaling([(r['config'],configs[r['model']]) for r in selected],contract[0]+contract[1]+contract[2])
    protocol={'selection_sha256':sha(OUT/'selected_models.json'),'selection_plan_sha256':sha(plan_path),
        'sources_sha256':sources,'selected_models':selected,'internal_test':base.INTERNAL,'external_test':base.EXTERNAL,
        'steady_test':{k:str(v.relative_to(ROOT)) for k,v in base.PROBES.items()},
        'preprocessing':'10-row median; saved model scaler; 18 historical steps',
        'baseline':'No-Gain seed42 test RMSE, same partition/case/target/evaluation',
        'KCI':'ANN-reference micro pooled all18; teacher |gain| >= 1e-5; +/-0.5 dynamic std; zero dynamic gain incorrect',
        'interpretation':'Previously used test trajectories and probes, not a new blind holdout; recorded future decoder inputs',
        'seed_scope':'Shared test data and ANN; dynamic training variability only',
        'prior_test_protocol_sha256':oldproto_sha,'prior_test_protocol_snapshot':oldproto_raw,
        'reused_models':12,'newly_evaluated_models':3,
        'source_limit':'Prior test protocol and validation histories belong to previous DG checkpoints; non-DG cache hashes and sources checked independently.',
        'optimizer_updates':0}
    proto=OUT/'run_protocol.json'
    if proto.exists() and load(proto)!=protocol: raise ValueError('Sources changed during test run')
    if not proto.exists(): write_json(proto,protocol)
    for row,folder,marker in reused:
        dest=OUT/'model_data'/row['model']; dest.mkdir(parents=True,exist_ok=True)
        for name,digest in marker['outputs'].items():
            target=dest/name
            if not target.exists(): shutil.copy2(folder/name,target)
            if sha(target)!=digest: raise ValueError(f'Reused test cache mismatch: {target}')
        new={'protocol_sha256':sha(proto),'outputs':marker['outputs'],
             'reused_from':str(folder.relative_to(ROOT)),'prior_protocol_sha256':oldproto_sha}
        if (dest/'complete.json').exists() and load(dest/'complete.json')!=new: raise ValueError('Reuse marker changed')
        if not (dest/'complete.json').exists(): write_json(dest/'complete.json',new)
    return configs,contract,protocol


def main():
    plan,candidates=select_validation()
    OUT.mkdir(parents=True,exist_ok=True)
    plan_path=OUT/'validation_selection_plan.json'
    if plan_path.exists() and load(plan_path)!=plan: raise ValueError('Frozen plan differs')
    if not plan_path.exists(): write_json(plan_path,plan)
    with (OUT/'validation_selection_candidates.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidates[0]));writer.writeheader();writer.writerows(candidates)
    print('Validation selection:',plan['aliases'],flush=True)
    base.OUT=OUT
    base.selection_before_test=select_models
    base.preflight=preflight
    base.run(plan_path)


if __name__=='__main__': main()
