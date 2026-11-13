"""Evaluate the five frozen, equal-weight Gain+SS models on validation only."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiment_paths import experiment_results_dir, teacher_results_dir
from src.models import get_model
from scripts.evaluation import evaluate_gain_in_out_distribution as kci_native
from scripts.prediction.validate_hyperparameters import (
    INTERNAL, EXTERNAL, TARGETS, KEYS, WEIGHTS, FOLDERS,
    evaluate_case, metric_rows, independent_rmse_rows, aggregate, sha, write_json,
)
from scripts.prediction.validation_comparison_report import build_report

OUTPUT = ROOT/'results/visualization/gain_ss_all_weights_casewise_region50_validation_seed42'
KCI_OUTPUT = ROOT/'results/evaluation/gain_eval_all18/validation/gain_ss_weights_seed42_20261105'
CODES = ['005', '2p5', '25', '250', '2500']
CONFIGS = {'no_gain': ROOT/'configs/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42.yaml'}
CONFIGS.update({f'gain{code}_ss{code}': ROOT/f'configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain{code}_ss{code}_seed42.yaml' for code in CODES})
MODEL_WEIGHTS = dict(zip(list(CONFIGS)[1:], WEIGHTS))


def load_config(path):
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    kci_native.model_contract(cfg)  # Populate the canonical encoder/decoder dimensions.
    return cfg


def preflight():
    configs = {alias: load_config(p) for alias, p in CONFIGS.items()}
    base = configs['no_gain']
    for alias, cfg in configs.items():
        assert cfg['model'] == base['model'] and cfg['window'] == base['window']
        assert cfg['data']['training_files'] == base['data']['training_files']
        assert cfg['training']['seed'] == 42
        if alias != 'no_gain':
            assert cfg['training']['gain_loss_weight'] == cfg['training']['ss_loss_weight'] == MODEL_WEIGHTS[alias]
            assert cfg['training']['ss_target_scope'] == 'all_48_outputs'
    pairs = [(str(CONFIGS[a]), c) for a, c in configs.items()]
    contract = kci_native.model_contract(base)
    for path, cfg in pairs:
        kci_native.assert_same_contract(contract, kci_native.model_contract(cfg), path)
    kci_native.assert_same_scaling(pairs, contract[0]+contract[1]+contract[2])
    sources = [Path(__file__), ROOT/'scripts/prediction/validate_hyperparameters.py',
               ROOT/'scripts/prediction/validation_comparison_report.py', ROOT/'scripts/prediction/predict.py',
               ROOT/'scripts/evaluation/evaluate_gain_in_out_distribution.py',
               ROOT/'scripts/training/train_pgin_stepwise_gain.py', ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
               *(ROOT/'src').rglob('*.py'), *(teacher_results_dir()/f'zscore_{v}.csv' for v in ['mean','std'])]
    for alias, cfg in configs.items():
        sources += [CONFIGS[alias], ROOT/'saved_models'/f"{cfg['exp_name']}.pth",
                    *(experiment_results_dir(cfg)/f'zscore_{v}.csv' for v in ['mean','std'])]
    for case in INTERNAL+EXTERNAL:
        sources.append(ROOT/f'data/Claus_dynamic/Test_dataform_change_air2_R=5-{case.split("-")[1]}.{"xlsx" if case in INTERNAL else "csv"}')
        sources.append(FOLDERS['gain']/'predictions/no_gain'/case/'predictions.npz')
    sources += [ROOT/f'data/Claus_steady_state/lhs_valid_{s}.xlsx' for s in ['in_range','soft_ood']]
    sources += [FOLDERS['gain']/'validation_metrics.csv', FOLDERS['gain']/'kci_pair_counts.csv']
    protected = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    # Retained Gain and SS-only analyses must not be overwritten by this run.
    retained = {str(p.relative_to(ROOT)): sha(p) for folder in FOLDERS.values() for p in folder.rglob('*') if p.is_file()}
    protocol = dict(model_weights=MODEL_WEIGHTS, internal=INTERNAL, external=EXTERNAL,
                    seed=42, protected_sha256=protected, retained_comparison_sha256=retained)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    KCI_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/'run_protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == protocol, 'Inputs changed since the cached run; preserve it before starting another run.'
    else:
        write_json(path, protocol)
    return configs, contract, protocol


def dynamic_validation(configs, device):
    data, audit = {}, []
    for case in INTERNAL+EXTERNAL:
        path = ROOT/f'data/Claus_dynamic/Test_dataform_change_air2_R=5-{case.split("-")[1]}.{"xlsx" if case in INTERNAL else "csv"}'
        raw = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
        frame = raw.rolling(10, min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        assert len(frame) >= 36 and np.isfinite(frame.to_numpy()).all() and frame[TARGETS].gt(0).all().all()
        data[case] = frame
        inside = raw.air2_SP.between(140,300) & raw.HEATER2_output_T_SP.between(140,240)
        audit.append(dict(case=case, partition='internal' if case in INTERNAL else 'external', source=str(path.relative_to(ROOT)),
                          raw_rows=len(raw), processed_rows=len(frame), TS_points=len(frame)-18, horizon_windows=len(frame)-35,
                          core_percent=float(inside.mean()*100), air2_min=raw.air2_SP.min(), air2_max=raw.air2_SP.max(),
                          T2_min=raw.HEATER2_output_T_SP.min(), T2_max=raw.HEATER2_output_T_SP.max()))
    selected, horizons, rechecks = [], [], []
    for alias, cfg in configs.items():
        mean, std = kci_native.load_model_stats(cfg['exp_name'], cfg)
        model = get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/'saved_models'/f"{cfg['exp_name']}.pth", map_location=device, weights_only=True))
        model.eval()
        for case, frame in data.items():
            dest = OUTPUT/'predictions'/alias/case
            dest.mkdir(parents=True, exist_ok=True)
            saved = dest/'predictions.npz'
            if saved.exists():
                with np.load(saved) as archive: arrays = dict(archive)
            else:
                arrays = evaluate_case(model, frame, cfg, mean, std, device)
                np.savez_compressed(saved, **arrays)
            partition = 'internal' if case in INTERNAL else 'external'
            rows, full = metric_rows(arrays, alias, case, partition)
            selected.extend(rows); horizons.extend(full)
            rechecks.extend(independent_rmse_rows(arrays, alias, case, partition))
            if alias == 'no_gain':
                with np.load(FOLDERS['gain']/'predictions/no_gain'/case/'predictions.npz') as reference:
                    for key, value in arrays.items(): np.testing.assert_allclose(value, reference[key], rtol=2e-5, atol=1e-10)
            pd.DataFrame({'time_min': np.arange(18,len(frame))*10,
                          **{f'{target}_true': arrays['ts_true'][:,j] for j,target in enumerate(TARGETS)},
                          **{f'{target}_pred': arrays['ts_pred'][:,j] for j,target in enumerate(TARGETS)}}).to_csv(dest/'time_series.csv',index=False)
            print(f'[Dynamic] {alias}: {case} complete ({device})',flush=True)
        del model
    for records, name in [(selected,'validation_metrics'), (horizons,'horizon_metrics'), (rechecks,'RMSE_recheck'), (audit,'data_audit')]:
        pd.DataFrame(records).to_csv(OUTPUT/f'{name}.csv',index=False)
    values, full, checked = map(pd.DataFrame, [selected,horizons,rechecks])
    assert (len(values),len(full),len(checked)) == (216,1296,1368)
    merged = checked.merge(values,on=['model']+KEYS,suffixes=('_recomputed','_csv'),validate='one_to_one')
    np.testing.assert_allclose(merged.RMSE_recomputed,merged.RMSE_csv,rtol=1e-12)
    hcheck = checked[checked.evaluation!='TS'].copy()
    hcheck['horizon'] = hcheck.evaluation.str[1:].astype(int)
    hcheck = hcheck.merge(full,on=['model','partition','case','target','horizon'],suffixes=('_recomputed','_csv'),validate='one_to_one')
    assert len(hcheck)==1296
    np.testing.assert_allclose(hcheck.RMSE_recomputed,hcheck.RMSE_csv,rtol=1e-12)
    prior = pd.read_csv(FOLDERS['gain']/'validation_metrics.csv').query("model == 'no_gain'")
    baseline = values.query("model == 'no_gain'").merge(prior,on=['model']+KEYS,suffixes=('_fresh','_prior'),validate='one_to_one')
    assert len(baseline)==36
    np.testing.assert_allclose(baseline.RMSE_fresh,baseline.RMSE_prior,rtol=2e-5,atol=1e-10)
    baseline.to_csv(OUTPUT/'baseline_metrics_recheck.csv',index=False)
    baseline.query("partition == 'internal'").to_csv(OUTPUT/'internal_metrics_recheck.csv',index=False)
    print('PASS: dynamic 216 metrics, 1296 horizon metrics, 1368 independent RMSE checks; all six No-Gain references match.',flush=True)


def numpy_counts(teacher, dynamic, threshold, targets, mvs):
    valid = np.broadcast_to((np.abs(teacher)>=threshold)[None,None], dynamic.shape)
    positive = np.broadcast_to((teacher>0)[None,None], dynamic.shape)
    negative = np.broadcast_to((teacher<0)[None,None], dynamic.shape)
    records = []
    for i, target in enumerate(targets):
        for j, mv in enumerate(mvs):
            mask, pos, neg, pred = valid[...,i,j], positive[...,i,j], negative[...,i,j], dynamic[...,i,j]
            records.append(dict(target=target,mv=mv,possible_comparisons=mask.size,valid_comparisons=int(mask.sum()),
                                tp=int((mask&pos&(pred>0)).sum()),tn=int((mask&neg&(pred<0)).sum()),
                                fp=int((mask&neg&(pred>0)).sum()),fn=int((mask&pos&(pred<0)).sum()),
                                zero_gain=int((mask&(pred==0)).sum())))
    return pd.DataFrame(records)


def steady_state_validation(configs, contract, device, batch_size):
    probe_sets = {}
    for distribution, count in [('in_range',4404),('soft_ood',2585)]:
        path = ROOT/f'data/Claus_steady_state/lhs_valid_{distribution}.xlsx'
        frame = kci_native.load_steady_state_source(str(path), contract[-1], log_target_cols=kci_native.FULL_TARGET_COLS)
        frame = kci_native.validate_distribution_file(frame, distribution)
        assert len(frame)==count
        probe_sets[distribution]=frame.reset_index(drop=True)
    teacher_artifacts = kci_native.load_teacher(device)
    all_rows=[]
    de, ys, en, targets, mvs, _ = contract
    for alias, weight in MODEL_WEIGHTS.items():
        cfg=configs[alias]
        mean,std=kci_native.load_model_stats(cfg['exp_name'],cfg)
        model=get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/'saved_models'/f"{cfg['exp_name']}.pth",map_location=device,weights_only=True))
        model.eval()
        threshold=float(cfg['training'].get('gain_valid_delta_threshold',1e-5))
        for distribution,data in probe_sets.items():
            dest=KCI_OUTPUT/alias/distribution
            dest.mkdir(parents=True,exist_ok=True)
            sentinel=dest/'evaluation_validation.json'
            if sentinel.exists():
                report=json.loads(sentinel.read_text(encoding='utf-8'))
                assert report['passed'] and all(sha(dest/name)==h for name,h in report['output_sha256'].items())
                all_rows.extend(pd.read_csv(dest/'gain_evaluation_detail.csv').to_dict('records'))
                print(f'[KCI cached] {alias}: {distribution}',flush=True)
                continue
            counts=kci_native.empty_counts(targets,mvs)
            teachers,dynamics=[],[]
            diagnostics=dest/'gain_probe_diagnostics.csv'
            for start in range(0,len(data),batch_size):
                batch=data.iloc[start:start+batch_size].reset_index(drop=True)
                teacher_gain=kci_native.teacher_gain_matrix(batch,*teacher_artifacts,targets,mvs,device)
                dynamic_gain,first_step=kci_native.dynamic_gain_matrix(model,batch,mean,std,de,ys,en,targets,mvs,cfg,device)
                assert first_step==1 and dynamic_gain.shape[:2]==(2,18)
                teachers.append(teacher_gain.cpu().numpy());dynamics.append(dynamic_gain.cpu().numpy())
                diagnostic_rows=[]
                for step in range(18):
                    kci_native.update_counts(counts,teacher_gain,dynamic_gain[:,step],targets,mvs,threshold)
                    rows=kci_native.probe_rows_for_batch(cfg['exp_name'],distribution,start,batch,teacher_gain,dynamic_gain[:,step],targets,mvs,threshold)
                    for row in rows:row.update(training_type='gain_ss',model=alias,gain_weight=weight,ss_weight=weight,step_index=step+1,post_step_index=step+1)
                    diagnostic_rows.extend(rows)
                pd.DataFrame(diagnostic_rows).to_csv(diagnostics,mode='w' if start==0 else 'a',header=start==0,index=False)
                if start==0 or (start//batch_size+1)%8==0 or start+batch_size>=len(data):
                    print(f'[KCI] {alias}: {distribution} {min(start+batch_size,len(data))}/{len(data)} probes',flush=True)
            teacher_array=np.concatenate(teachers,axis=0);dynamic_array=np.concatenate(dynamics,axis=2)
            assert np.isfinite(teacher_array).all() and np.isfinite(dynamic_array).all()
            np.savez_compressed(dest/'gain_arrays.npz',teacher_gain=teacher_array,dynamic_gain=dynamic_array,
                                physical_ANN_inputs=data[kci_native.TAB_INPUT_COLS].to_numpy(),target_cols=targets,mv_cols=mvs)
            details=[]
            for target in targets:
                for mv in mvs:
                    row=kci_native.count_row(cfg['exp_name'],distribution,target,mv,counts[(target,mv)],len(data))
                    row.update(model=alias,gain_weight=weight,ss_weight=weight,training_type='gain_ss',post_step_start=1,post_step_end=18)
                    row['correct']=row['tp']+row['tn'];details.append(row)
            details=pd.DataFrame(details)
            independent=numpy_counts(teacher_array,dynamic_array,threshold,targets,mvs)
            compare=details.merge(independent,on=['target','mv'],suffixes=('_native','_numpy'),validate='one_to_one')
            for col in ['possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain']:
                np.testing.assert_array_equal(compare[f'{col}_native'],compare[f'{col}_numpy'])
            # Check all diagnostic CSV rows independently, streaming to bound memory.
            recount=[]
            for chunk in pd.read_csv(diagnostics,chunksize=100_000,usecols=['target','mv','valid_teacher_gain','teacher_sign','dynamic_sign']):
                chunk['tp']=chunk.valid_teacher_gain & chunk.teacher_sign.eq(1) & chunk.dynamic_sign.eq(1)
                chunk['tn']=chunk.valid_teacher_gain & chunk.teacher_sign.eq(-1) & chunk.dynamic_sign.eq(-1)
                chunk['valid']=chunk.valid_teacher_gain.astype(int)
                recount.append(chunk.groupby(['target','mv'])[['tp','tn','valid']].sum())
            recount=pd.concat(recount).groupby(level=[0,1]).sum().reset_index()
            compare=details.merge(recount,on=['target','mv'],suffixes=('_native','_csv'),validate='one_to_one')
            np.testing.assert_array_equal(compare.tp_native,compare.tp_csv)
            np.testing.assert_array_equal(compare.tn_native,compare.tn_csv)
            np.testing.assert_array_equal(compare.valid_comparisons,compare.valid)
            details.to_csv(dest/'gain_evaluation_detail.csv',index=False)
            write_json(sentinel,dict(passed=True,probe_points=len(data),diagnostic_rows=len(data)*2*18*len(targets)*len(mvs),
                                    numpy_confusion_recount=True,CSV_confusion_recount=True,threshold=threshold,
                                    output_sha256={name:sha(dest/name) for name in ['gain_arrays.npz','gain_evaluation_detail.csv','gain_probe_diagnostics.csv']}))
            all_rows.extend(details.to_dict('records'))
            del teachers,dynamics,teacher_array,dynamic_array,diagnostic_rows,rows
            gc.collect()
        del model
    all_rows=pd.DataFrame(all_rows)
    assert len(all_rows)==40
    reference=pd.read_csv(FOLDERS['gain']/'kci_pair_counts.csv').query("model == 'gain005'")
    for alias in MODEL_WEIGHTS:
        joined=all_rows[all_rows.model==alias].merge(reference,on=['distribution','target','mv'],suffixes=('_new','_reference'),validate='one_to_one')
        for col in ['probe_points','possible_comparisons','valid_comparisons']:
            np.testing.assert_array_equal(joined[f'{col}_new'],joined[f'{col}_reference'])
    all_rows.to_csv(KCI_OUTPUT/'gain_ss_pair_kci_comparison.csv',index=False)
    all_rows.to_csv(OUTPUT/'gain_ss_kci_components.csv',index=False)
    kci_native.build_summary(all_rows).to_csv(KCI_OUTPUT/'gain_evaluation_summary.csv',index=False)
    print('PASS: fresh KCI on the same 4404/2585 probes; 40 pair counts match NumPy/CSV recounts and prior validation denominators.',flush=True)


def comparison_report(protocol, device):
    metrics=pd.read_csv(OUTPUT/'validation_metrics.csv')
    selected=metrics[metrics.model.isin(MODEL_WEIGHTS)].copy()
    selected['gain_weight']=selected.model.map(MODEL_WEIGHTS)
    selected['ss_weight']=selected.gain_weight
    base=metrics[metrics.model=='no_gain'][KEYS+['RMSE','n']].rename(columns={'RMSE':'NoGain_RMSE','n':'NoGain_n'})
    components=selected.merge(base,on=KEYS,validate='many_to_one')
    assert len(components)==180 and components.n.eq(components.NoGain_n).all() and components.NoGain_RMSE.gt(0).all()
    components['relative_RMSE']=components.RMSE/components.NoGain_RMSE
    components['overall_log_weight']=.5/18
    assert np.isfinite(components.relative_RMSE).all() and components.relative_RMSE.gt(0).all()
    regional=aggregate(components,['model','gain_weight','ss_weight','partition'])
    overall=regional.pivot(index=['model','gain_weight','ss_weight'],columns='partition',values='AvgRelRMSE').rename(columns={'internal':'internal_AvgRelRMSE','external':'external_AvgRelRMSE'}).reset_index()
    overall['AvgRelRMSE']=np.sqrt(overall.internal_AvgRelRMSE*overall.external_AvgRelRMSE)
    overall['component_count']=36
    kci=pd.read_csv(OUTPUT/'gain_ss_kci_components.csv')
    counts=kci.groupby('model').agg(correct=('correct','sum'),valid=('valid_comparisons','sum')).reset_index()
    counts['overall_micro_KCI_percent']=100*counts.correct/counts.valid
    overall=overall.merge(counts,on='model',validate='one_to_one').sort_values('AvgRelRMSE')
    overall.insert(0,'rank',range(1,6))
    kcireg=kci.groupby(['model','distribution']).agg(correct=('correct','sum'),valid=('valid_comparisons','sum')).reset_index()
    kcireg['partition']=kcireg.distribution.map({'in_range':'internal','soft_ood':'external'})
    kcireg['micro_kci_percent']=100*kcireg.correct/kcireg.valid
    regional=regional.merge(kcireg[['model','partition','micro_kci_percent']],on=['model','partition'],validate='one_to_one')
    per_region=aggregate(components,['model','gain_weight','ss_weight','partition','evaluation']).merge(kcireg[['model','partition','micro_kci_percent']],on=['model','partition'],validate='many_to_one')
    per_evaluation=per_region.pivot(index=['model','gain_weight','ss_weight','evaluation'],columns='partition',values='AvgRelRMSE').reset_index()
    per_evaluation['AvgRelRMSE']=np.sqrt(per_evaluation.internal*per_evaluation.external)
    summary=selected.groupby(['partition','model','gain_weight','ss_weight','target','evaluation']).agg(cases=('case','size'),mean_case_RMSE=('RMSE','mean')).reset_index()
    for frame,name in [(overall,'avgrelrmse_overall'),(regional,'avgrelrmse_by_region'),(per_region,'avgrelrmse_by_region_evaluation'),
                       (per_evaluation,'avgrelrmse_by_evaluation'),(summary,'validation_rmse_summary'),(components,'relative_rmse_components')]:
        frame.to_csv(OUTPUT/f'{name}.csv',index=False)
    write_json(OUTPUT/'metric_protocol.json',dict(model_family='gain_ss',internal_validation=INTERNAL,external_validation=EXTERNAL,
               weights=WEIGHTS,seed=42,region_weights=[.5,.5],internal_ratios=18,external_ratios=18,
               internal_component_log_weight=.5/18,external_component_log_weight=.5/18,
               formula='sqrt(GM_internal(same-case RMSE / NoGain RMSE) * GM_external(same-case RMSE / NoGain RMSE))',
               regional_RMSE_arithmetic_preaveraging=False,SS_target_scope='all_48_outputs',
               preprocessing='10-row median; H2S/SO2 log; saved training z-score statistics',
               KCI='Fresh ANN-reference all18 strict sign agreement; zero dynamic gain is incorrect; same validation probes, teacher, scaling and threshold as retained Gain/SS-only; overall KCI is pooled micro, not region50',
               KCI_sources=['data/Claus_steady_state/lhs_valid_in_range.xlsx','data/Claus_steady_state/lhs_valid_soft_ood.xlsx'],
               KCI_probe_points=[4404,2585],optimizer_updates=0,output_directory=OUTPUT.relative_to(ROOT).as_posix(),
               full_rerun_command='.\\.venv\\Scripts\\python.exe -m scripts.prediction.validate_gain_ss_hyperparameters'))
    rebuild='''"""Rebuild the Gain+SS report from saved validation CSVs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.prediction.validation_comparison_report import build_report
if __name__=='__main__':
    build_report(Path(__file__).resolve().parent)
'''
    (OUTPUT/'rebuild_figures.py').write_text(rebuild,encoding='utf-8')
    write_json(OUTPUT/'kci_provenance.json',dict(passed=True,fresh_evaluation=True,training_KCI_history_used=False,
               source_directory=KCI_OUTPUT.relative_to(ROOT).as_posix(),probe_points=[4404,2585],
               independent_NumPy_and_diagnostic_CSV_counts_checked=True,identical_denominators_to_Gain_validation=True,
               protected_sha256=protocol['protected_sha256']))
    build_report(OUTPUT)
    # A compact cross-family table uses exactly the same cases and ratio definition.
    frames=[]
    for family,folder in {**FOLDERS,'gain_ss':OUTPUT}.items():
        p=json.loads((folder/'metric_protocol.json').read_text(encoding='utf-8'))
        assert p['internal_validation']==INTERNAL and p['external_validation']==EXTERNAL and p['region_weights']==[.5,.5]
        frame=pd.read_csv(folder/'avgrelrmse_overall.csv')
        if family=='gain':frame['ss_weight']=0
        if family=='ss_only':frame['gain_weight']=0
        frame['family']=family
        frames.append(frame[['family','model','gain_weight','ss_weight','internal_AvgRelRMSE','external_AvgRelRMSE','AvgRelRMSE','overall_micro_KCI_percent']])
    pd.concat(frames).sort_values('AvgRelRMSE').to_csv(OUTPUT/'three_family_validation_comparison.csv',index=False)
    assert all(sha(ROOT/name)==h for name,h in protocol['protected_sha256'].items()),'Protected inputs changed.'
    assert all(sha(ROOT/name)==h for name,h in protocol['retained_comparison_sha256'].items()),'Retained Gain/SS-only output changed.'
    report=dict(passed=True,device=str(device),model_count=5,baseline_model_count=1,case_count=6,
                selected_metric_rows=216,horizon_metric_rows=1296,independent_RMSE_rechecks=1368,paired_RMSE_ratios=180,
                baseline_reference_checks=36,raw_truth_alignment_checked=True,KCI_recomputed=True,KCI_probe_points=[4404,2585],
                optimizer_updates=0,protected_inputs_unchanged=True,retained_comparisons_unchanged=True,
                protected_sha256=protocol['protected_sha256'],visual_review='pending')
    write_json(OUTPUT/'evaluation_validation.json',report)
    write_json(KCI_OUTPUT/'evaluation_validation.json',dict(passed=True,model_count=5,probe_points=[4404,2585],
               diagnostic_rows_recounted=5*(4404+2585)*2*18*4,protected_sha256=protocol['protected_sha256']))
    print(overall[['model','internal_AvgRelRMSE','external_AvgRelRMSE','AvgRelRMSE','overall_micro_KCI_percent']].to_string(index=False),flush=True)
    print('PASS: Gain+SS validation complete; retained analyses and frozen model inputs unchanged.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','dynamic','kci','report'],default='all')
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args()
    assert args.batch_size>0
    torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs,contract,protocol=preflight()
    print(f'[Preflight] five equal-weight Gain+SS checkpoints ready; {device}; no optimizer updates.',flush=True)
    if args.phase in {'all','dynamic'}:dynamic_validation(configs,device)
    if args.phase in {'all','kci'}:steady_state_validation(configs,contract,device,args.batch_size)
    if args.phase in {'all','report'}:comparison_report(protocol,device)


if __name__=='__main__':
    main()
