"""Freeze the expanded validation-only three-seed selection before test use."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VIS = ROOT / 'results/visualization'
BASE = VIS / 'retrain_lhs10_23_63_models_validation_nogain42_baseline'
INCREMENTAL = VIS / 'retrain_lhs10_23_expanded_dg_validation'
OUT = VIS / 'retrain_lhs10_23_selected_test_nogain42/updated_validation_best_staging'
OLD_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
NEW_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
FAMILIES = ('gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
SEEDS = (42, 43, 44)
CASES = ('R5-3', 'R5-4', 'R5-5', 'R5-27', 'R5-28', 'R5-29')
KEYS = ['partition', 'case', 'model', 'target', 'evaluation']


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main() -> None:
    report, provenance = load(BASE/'analysis_validation.json'), load(BASE/'provenance.json')
    if not (report.get('passed') and report.get('model_count') == 63 and report.get('regional_points') == 126
            and report.get('components') == 2268 and report.get('seeds') == list(SEEDS)
            and provenance.get('baseline_seed') == 42):
        raise ValueError('Expanded validation is not complete')
    for name, digest in provenance['outputs_sha256'].items():
        if sha(BASE/name).lower() != digest.lower():
            raise ValueError(f'Validation output changed: {name}')
    missing_historical = []
    for relative, digest in provenance['sources_sha256'].items():
        path = ROOT / relative
        if not path.is_file():
            if 'retrain_lhs10_23_63_models_validation_nogain42_baseline_previous_before_dg_expansion' not in relative:
                raise FileNotFoundError(path)
            missing_historical.append({'path': relative, 'expected_sha256': digest})
        elif sha(path).lower() != digest.lower():
            raise ValueError(f'Validation source changed: {relative}')
    if len(missing_historical) not in (0, 5):
        raise ValueError('Historical snapshot is unexpectedly partial')
    increment_report, increment_protocol = load(INCREMENTAL/'evaluation_validation.json'), load(INCREMENTAL/'run_protocol.json')
    if not (increment_report.get('passed') and increment_report.get('model_count') == 12
            and increment_report.get('source_hashes_unchanged') and increment_report.get('independent_rmse_and_numpy_kci_recounts')):
        raise ValueError('New DG validation is not complete')
    for relative, digest in increment_protocol['sources_sha256'].items():
        if sha(ROOT/relative).lower() != digest.lower():
            raise ValueError(f'Expanded validation input changed: {relative}')

    oldjobs, newjobs = load(OLD_MANIFEST), load(NEW_MANIFEST)
    jobs = [j for j in oldjobs if j['family'] != 'difficult_gradnorm' or j['alias'] == 'dg_ratio0p015'] + newjobs
    roster = {(j['seed'], j['alias']): j for j in jobs}
    if len(jobs) != 63 or len(roster) != 63:
        raise ValueError('Expanded roster must have 63 unique runs')
    for seed in SEEDS:
        if sum(s == seed for s, _ in roster) != 21:
            raise ValueError(f'Wrong job coverage for seed {seed}')
    model_family = {j['alias']: j['family'] for j in jobs if j['seed'] == 42}
    if len(model_family) != 21 or any(sum(v == f for v in model_family.values()) != 5 for f in FAMILIES):
        raise ValueError('Expected five candidates in each family')

    coords = pd.read_csv(BASE/'regional_coordinates_63_models.csv')
    components = pd.read_csv(BASE/'relative_rmse_components.csv')
    kci = pd.read_csv(BASE/'kci_components.csv')
    expected = {(s, model, region) for s, model in roster for region in ('internal', 'external')}
    if len(coords) != 126 or len(set(zip(coords.seed, coords.model, coords.region))) != 126 or set(zip(coords.seed, coords.model, coords.region)) != expected:
        raise ValueError('Validation coordinate coverage differs from roster')
    if len(components) != 2268 or len(kci) != 504:
        raise ValueError('Validation component coverage differs from roster')
    if not (np.isfinite(coords[['AvgRelRMSE','KCI_percent']]).all().all() and coords.AvgRelRMSE.gt(0).all()
            and coords.KCI_percent.between(0,100).all() and np.allclose(coords.KCI_percent,100*coords.correct/coords.valid)):
        raise ValueError('Invalid validation metrics')
    if not np.allclose(components.relative_RMSE, components.RMSE/components.NoGain_seed42_RMSE, rtol=1e-12, atol=1e-12):
        raise ValueError('Relative RMSE differs from fixed reference')

    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), OLD_MANIFEST, NEW_MANIFEST,
        BASE/'analysis_validation.json', BASE/'provenance.json', BASE/'regional_coordinates_63_models.csv',
        BASE/'relative_rmse_components.csv', BASE/'kci_components.csv', BASE/'scores_by_model_seed.csv',
        INCREMENTAL/'evaluation_validation.json', INCREMENTAL/'run_protocol.json']}
    reused, new = 0, 0
    for (seed, model), job in roster.items():
        is_new = job in newjobs
        d = (INCREMENTAL/'model_data'/f'seed{seed}'/model if is_new else
             VIS/f'retrain_lhs10_23_21_models_validation_seed{seed}'/'model_data'/model)
        protocol = INCREMENTAL/'run_protocol.json' if is_new else d.parents[1]/'run_protocol.json'
        complete = load(d/'complete.json')
        if complete['run_protocol_sha256'].lower() != sha(protocol).lower():
            raise ValueError(f'Validation protocol mismatch: {seed} {model}')
        for name in ['validation_metrics.csv','kci_components.csv','gain_in_range.npz','gain_soft_ood.npz',
                     *[f'{case}_prediction.npz' for case in CASES]]:
            path = d/name
            digest = sha(path)
            if digest.lower() != complete['outputs'][name].lower():
                raise ValueError(f'Validation model output changed: {path}')
            source_hashes[str(path.relative_to(ROOT))] = digest
        source_hashes[str((d/'complete.json').relative_to(ROOT))] = sha(d/'complete.json')
        source_hashes[str(protocol.relative_to(ROOT))] = sha(protocol)
        model_metrics = pd.read_csv(d/'validation_metrics.csv')
        left = components[(components.seed == seed)&(components.model == model)]
        merged = left.merge(model_metrics, on=KEYS, suffixes=('_aggregate','_cached'), validate='one_to_one')
        if len(merged) != 36 or not np.array_equal(merged.n_aggregate,merged.n_cached) or not np.allclose(merged.RMSE_aggregate,merged.RMSE_cached,rtol=1e-12,atol=1e-15):
            raise ValueError(f'Cached RMSE differs: {seed} {model}')
        model_kci = pd.read_csv(d/'kci_components.csv')
        left = kci[(kci.seed == seed)&(kci.model == model)]
        merged = left.merge(model_kci, on=['model','distribution','target','mv'],suffixes=('_aggregate','_cached'),validate='one_to_one')
        if len(merged) != 8 or any(not np.array_equal(merged[f'{col}_aggregate'],merged[f'{col}_cached']) for col in ('valid_comparisons','tp','tn','fp','fn','zero_gain')):
            raise ValueError(f'Cached KCI differs: {seed} {model}')
        reused += not is_new
        new += is_new
    if (reused,new) != (51,12):
        raise ValueError('Expected 51 historic and 12 new runs')

    p = coords.pivot(index=['model','family','seed'],columns='region',values=['AvgRelRMSE','KCI_percent'])
    p.columns = ['_'.join(t) for t in p.columns]
    p = p.reset_index()
    p['error_50_50'] = np.sqrt(p.AvgRelRMSE_internal*p.AvgRelRMSE_external)
    p['kci_50_50'] = (p.KCI_percent_internal+p.KCI_percent_external)/2
    old = pd.read_csv(BASE/'scores_by_model_seed.csv')
    c = p.merge(old,on=['model','family','seed'],validate='one_to_one')
    if len(c) != 63 or not np.allclose(c.error_50_50,c.overall_50_50,rtol=1e-12):
        raise ValueError('50/50 score differs from validation summary')
    candidates=[]
    for family in FAMILIES:
        sub = p[p.family.eq(family)].groupby('model',as_index=False).agg(
            error_mean=('error_50_50','mean'),error_sd=('error_50_50','std'),
            kci_mean=('kci_50_50','mean'),kci_sd=('kci_50_50','std'),n=('seed','nunique'))
        if len(sub)!=5 or not sub.n.eq(3).all(): raise ValueError('Candidate coverage mismatch')
        emin,emax=float(sub.error_mean.min()),float(sub.error_mean.max())
        kmin,kmax=float(sub.kci_mean.min()),float(sub.kci_mean.max())
        if emin==emax or kmin==kmax: raise ValueError('Undefined min-max normalization')
        sub['error_normalized']=(sub.error_mean-emin)/(emax-emin)
        sub['kci_shortfall_normalized']=(kmax-sub.kci_mean)/(kmax-kmin)
        sub['distance']=np.hypot(sub.error_normalized,sub.kci_shortfall_normalized)
        sub=sub.sort_values(['distance','error_mean','model'],kind='mergesort')
        for rank, row in enumerate(sub.to_dict('records'),1):
            candidates.append({'family':family,'alias':row['model'],'rank':rank,
                'error_mean':float(row['error_mean']),'error_sd':float(row['error_sd']),
                'kci_mean_percent':float(row['kci_mean']),'kci_sd_percentage_points':float(row['kci_sd']),
                'error_normalized':float(row['error_normalized']),'kci_shortfall_normalized':float(row['kci_shortfall_normalized']),
                'distance':float(row['distance']),'n_seeds':int(row['n']),
                'error_family_min':emin,'error_family_max':emax,'kci_family_min_percent':kmin,'kci_family_max_percent':kmax})
    aliases={family:next(c['alias'] for c in candidates if c['family']==family and c['rank']==1) for family in FAMILIES}
    rule=('Validation only; for each seed error is sqrt(internal AvgRelRMSE * external AvgRelRMSE) against fixed No-Gain seed 42, '
          'and KCI is the arithmetic mean of internal and external percentages; average each over seeds 42,43,44. '
          'Within each family min-max normalize the five errors and KCI shortfalls, then choose the shortest equal-weight '
          'Euclidean distance to the minimum-error maximum-KCI ideal; ties: error then alias.')
    result={'aliases':aliases,'selection_rule':rule,'selection_details':{'baseline':'No-Gain seed 42','validation_only':True,
        'seeds':list(SEEDS),'family_candidate_count':5,'normalization_note':'Distances depend on the five candidates and are not comparable across families.'},
        'provenance':{'sources_sha256':source_hashes,'upstream_output_sha256_checked':provenance['outputs_sha256'],
                      'expanded_validation_sources_checked':len(increment_protocol['sources_sha256']),
                      'retained_cached_models_checked':reused,'new_cached_models_checked':new,
                      'missing_historical_snapshot_sources':missing_historical,
                      'missing_historical_note':'Five previous baseline snapshot files are unavailable; current 63 cached validation outputs and new DG sources were independently checked directly. Prior historical snapshot chain is not fully verifiable.'},
        'scores':candidates}
    for relative,digest in source_hashes.items():
        if sha(ROOT/relative).lower()!=digest.lower(): raise ValueError(f'Input changed: {relative}')
    OUT.mkdir(parents=True,exist_ok=True)
    plan=OUT/'validation_selection_plan.json'; table=OUT/'validation_selection_candidates.csv'
    if plan.exists() or table.exists(): raise FileExistsError('Existing selection is frozen; refusing overwrite')
    with table.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
    with plan.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'aliases':aliases,'plan':str(plan),'missing_historical_sources':len(missing_historical)},indent=2))


if __name__ == '__main__':
    main()
