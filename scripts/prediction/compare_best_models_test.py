"""Compare validation-selected Gain, SS-only and Gain+SS on the existing test set."""
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.experiment_paths import experiment_results_dir,teacher_results_dir
from src.models import get_model
from scripts.evaluation import evaluate_gain_in_out_distribution as native
from scripts.prediction.validate_hyperparameters import evaluate_case,metric_rows,independent_rmse_rows,sha,write_json,TARGETS,KEYS
from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts
from scripts.plotting.plot_best_models_test_tables import build_report

OUTPUT=ROOT/'results/visualization/best_gain_ss_gainss_test_seed42'
KCI_OUTPUT=ROOT/'results/evaluation/gain_eval_all18/test/best_family_models_seed42_20261106'
PRIOR=ROOT/'results/visualization/three_model_no_gain_gain250_ss2p5_test_seed42'
INTERNAL=['R5','R5-1','R5-2','R5-6']
EXTERNAL=[f'R5-{i}' for i in range(19,27)]
VALIDATION={
    'gain':ROOT/'results/visualization/gain005_gain2p5_gain25_gain250_gain2500_casewise_region50_validation_seed42',
    'ss_only':ROOT/'results/visualization/ss_only_all_weights_casewise_region50_validation_seed42',
    'gain_ss':ROOT/'results/visualization/gain_ss_all_weights_casewise_region50_validation_seed42',
}
LABELS={'no_gain':'No-Gain','gain':'Gain (250)','ss_only':'SS-only (2.5)','gain_ss':'Gain+SS (2.5/2.5)'}


def preflight():
    selection=[dict(family='no_gain',alias='no_gain',gain_weight=0.,ss_weight=0.,validation_AvgRelRMSE=1.)]
    for family,folder in VALIDATION.items():
        metric=json.loads((folder/'metric_protocol.json').read_text(encoding='utf-8'))
        assert metric['internal_validation']==['R5-3','R5-4','R5-5']
        assert metric['external_validation']==['R5-27','R5-28','R5-29'] and metric['region_weights']==[.5,.5]
        best=pd.read_csv(folder/'avgrelrmse_overall.csv').sort_values(['AvgRelRMSE','model']).iloc[0]
        selection.append(dict(family=family,alias=best.model,gain_weight=0. if family=='ss_only' else float(best.gain_weight),
                              ss_weight=0. if family=='gain' else float(best.ss_weight),validation_AvgRelRMSE=float(best.AvgRelRMSE)))
    assert [r['alias'] for r in selection]==['no_gain','gain250','ss2p5_only','gain2p5_ss2p5'],'Validation winners changed; review the model labels before proceeding.'
    configs={}
    sources=[Path(__file__),ROOT/'scripts/plotting/plot_best_models_test_tables.py',
             ROOT/'scripts/evaluation/evaluate_gain_in_out_distribution.py',ROOT/'scripts/prediction/validate_hyperparameters.py',
             ROOT/'scripts/prediction/validate_gain_ss_hyperparameters.py',ROOT/'scripts/prediction/predict.py',
             ROOT/'scripts/training/train_pgin_stepwise_gain.py',*(ROOT/'src').rglob('*.py'),
             ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',*(teacher_results_dir()/f'zscore_{v}.csv' for v in ['mean','std'])]
    for row in selection:
        path=ROOT/f"configs/pgin/{row['family']}/transformer_layerwise_63var_decoder_input_sp_pgin_{row['alias']}_seed42.yaml"
        cfg=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        native.model_contract(cfg)
        assert cfg['training']['seed']==42
        assert cfg['training']['gain_loss_weight']==row['gain_weight'] and cfg['training']['ss_loss_weight']==row['ss_weight']
        if row['family'] in {'ss_only','gain_ss'}:assert cfg['training']['ss_target_scope']=='all_48_outputs'
        row.update(label=LABELS[row['family']],config=path.relative_to(ROOT).as_posix(),checkpoint=f"saved_models/{cfg['exp_name']}.pth")
        sources += [path,ROOT/row['checkpoint'],*(experiment_results_dir(cfg)/f'zscore_{v}.csv' for v in ['mean','std'])]
        configs[row['family']]=cfg
    base=configs['no_gain'];contract=native.model_contract(base)
    for family,cfg in configs.items():
        assert cfg['model']==base['model'] and cfg['window']==base['window']
        assert cfg['data']['training_files']==base['data']['training_files']
        native.assert_same_contract(contract,native.model_contract(cfg),family)
    native.assert_same_scaling([(r['config'],configs[r['family']]) for r in selection],contract[0]+contract[1]+contract[2])
    for case in INTERNAL+EXTERNAL:
        sources.append(ROOT/f'data/Claus_dynamic/Test_dataform_change_air2_R={case.replace("R5","5")}{"_test_split" if case in INTERNAL else ""}.csv')
        sources.extend(PRIOR/'predictions'/alias/case/'predictions.npz' for alias in ['no_gain','gain250','ss2p5_only'])
    sources += [ROOT/f'data/Claus_steady_state/lhs_test_{s}.xlsx' for s in ['in_range','soft_ood']]
    sources += [folder/name for folder in VALIDATION.values() for name in ['avgrelrmse_overall.csv','metric_protocol.json']]
    sources += [ROOT/'results/evaluation/gain_eval_all18/test/no_gain_seed42/gain_evaluation_detail.csv']
    protected={str(p.relative_to(ROOT)):sha(p) for p in sources}
    retained={str(p.relative_to(ROOT)):sha(p) for folder in [*VALIDATION.values(),PRIOR] for p in folder.rglob('*') if p.is_file()}
    protocol=dict(selected_models=selection,selection_rule='Minimum validation AvgRelRMSE within each family; test never selects weights',
                  internal_test=INTERNAL,external_test=EXTERNAL,steady_state_test_sources=['data/Claus_steady_state/lhs_test_in_range.xlsx','data/Claus_steady_state/lhs_test_soft_ood.xlsx'],
                  steady_state_test_probe_points=[4385,2692],delta_percent='100*(RMSE_model/RMSE_NoGain-1); negative means improvement',
                  RMSE_display_unit='10^-3 kmol/kmol (raw RMSE multiplied by 1000)',seed=42,
                  data_roles='Existing test cases; not a new blind test; no validation cases used for test metrics',
                  protected_sha256=protected,retained_comparison_sha256=retained)
    OUTPUT.mkdir(parents=True,exist_ok=True);KCI_OUTPUT.mkdir(parents=True,exist_ok=True)
    p=OUTPUT/'run_protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocol,'Protected inputs changed; preserve the previous result before rerunning.'
    else:write_json(p,protocol)
    pd.DataFrame(selection).to_csv(OUTPUT/'selected_models.csv',index=False)
    return configs,contract,protocol


def dynamic_test(configs,selection,device):
    frames={};audits=[]
    for case in INTERNAL+EXTERNAL:
        path=ROOT/f'data/Claus_dynamic/Test_dataform_change_air2_R={case.replace("R5","5")}{"_test_split" if case in INTERNAL else ""}.csv'
        raw=pd.read_csv(path)
        frame=raw.rolling(10,min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        assert len(frame)>=36 and np.isfinite(frame.to_numpy()).all() and frame[TARGETS].gt(0).all().all()
        frames[case]=frame
        audits.append(dict(case=case,partition='internal' if case in INTERNAL else 'external',source=path.relative_to(ROOT).as_posix(),
                           raw_rows=len(raw),processed_rows=len(frame),TS_points=len(frame)-18,horizon_windows=len(frame)-35))
    selected=[];horizons=[];checks=[];prior_cases_checked=0
    for row in selection:
        family,alias=row['family'],row['alias'];cfg=configs[family]
        mean,std=native.load_model_stats(cfg['exp_name'],cfg)
        model=get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/row['checkpoint'],map_location=device,weights_only=True));model.eval()
        for case,frame in frames.items():
            dest=OUTPUT/'predictions'/family/case;dest.mkdir(parents=True,exist_ok=True)
            p=dest/'predictions.npz'
            if p.exists():
                with np.load(p) as archive:arrays=dict(archive)
            else:
                arrays=evaluate_case(model,frame,cfg,mean,std,device);np.savez_compressed(p,**arrays)
            a,b=metric_rows(arrays,family,case,'internal' if case in INTERNAL else 'external')
            selected.extend(a);horizons.extend(b);checks.extend(independent_rmse_rows(arrays,family,case,'internal' if case in INTERNAL else 'external'))
            if family!='gain_ss':
                with np.load(PRIOR/'predictions'/alias/case/'predictions.npz') as old:
                    for name,value in arrays.items():np.testing.assert_allclose(value,old[name],rtol=2e-5,atol=1e-10)
                prior_cases_checked+=1
            pd.DataFrame({'time_min':np.arange(18,len(frame))*10,
                          **{f'{t}_true':arrays['ts_true'][:,j] for j,t in enumerate(TARGETS)},
                          **{f'{t}_pred':arrays['ts_pred'][:,j] for j,t in enumerate(TARGETS)}}).to_csv(dest/'time_series.csv',index=False)
            print(f'[Dynamic] {family}: {case} done ({device})',flush=True)
        del model
    metrics,full,recheck=map(pd.DataFrame,[selected,horizons,checks])
    assert (len(metrics),len(full),len(recheck),prior_cases_checked)==(288,1728,1824,36)
    match=recheck.merge(metrics,on=['model']+KEYS,suffixes=('_direct','_metric'),validate='one_to_one')
    np.testing.assert_allclose(match.RMSE_direct,match.RMSE_metric,rtol=1e-12)
    h=recheck[recheck.evaluation!='TS'].copy();h['horizon']=h.evaluation.str[1:].astype(int)
    h=h.merge(full,on=['model','partition','case','target','horizon'],suffixes=('_direct','_metric'),validate='one_to_one')
    assert len(h)==1728
    np.testing.assert_allclose(h.RMSE_direct,h.RMSE_metric,rtol=1e-12)
    for frame,name in [(metrics,'test_metrics'),(full,'horizon_metrics'),(recheck,'RMSE_recheck'),(pd.DataFrame(audits),'data_audit')]:frame.to_csv(OUTPUT/f'{name}.csv',index=False)
    print('PASS: 1824 independent RMSE checks; 36 previously evaluated model/case predictions reproduced.',flush=True)


def kci_test(configs,contract,selection,device):
    data={}
    for distribution,expected in [('in_range',4385),('soft_ood',2692)]:
        source=ROOT/f'data/Claus_steady_state/lhs_test_{distribution}.xlsx'
        frame=native.load_steady_state_source(str(source),contract[-1],log_target_cols=native.FULL_TARGET_COLS)
        frame=native.validate_distribution_file(frame,distribution).reset_index(drop=True)
        assert len(frame)==expected
        data[distribution]=frame
    teacher=native.load_teacher(device)
    de,ys,en,targets,mvs,_=contract
    combined=[]
    for chosen in selection:
        family=chosen['family'];cfg=configs[family]
        model=get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/chosen['checkpoint'],map_location=device,weights_only=True));model.eval()
        mean,std=native.load_model_stats(cfg['exp_name'],cfg)
        threshold=float(cfg['training'].get('gain_valid_delta_threshold',1e-5))
        for distribution,frame in data.items():
            dest=KCI_OUTPUT/family/distribution;dest.mkdir(parents=True,exist_ok=True)
            sentinel=dest/'evaluation_validation.json'
            if sentinel.exists():
                report=json.loads(sentinel.read_text(encoding='utf-8'))
                assert report['passed'] and all(sha(dest/name)==digest for name,digest in report['output_sha256'].items())
                combined.extend(pd.read_csv(dest/'gain_evaluation_detail.csv').to_dict('records'));continue
            counts=native.empty_counts(targets,mvs);teachers=[];dynamics=[]
            csv=dest/'gain_probe_diagnostics.csv'
            for start in range(0,len(frame),64):
                batch=frame.iloc[start:start+64].reset_index(drop=True)
                reference=native.teacher_gain_matrix(batch,*teacher,targets,mvs,device)
                gain,first=native.dynamic_gain_matrix(model,batch,mean,std,de,ys,en,targets,mvs,cfg,device)
                assert first==1 and gain.shape[:2]==(2,18)
                teachers.append(reference.cpu().numpy());dynamics.append(gain.cpu().numpy())
                records=[]
                for step in range(18):
                    native.update_counts(counts,reference,gain[:,step],targets,mvs,threshold)
                    rows=native.probe_rows_for_batch(cfg['exp_name'],distribution,start,batch,reference,gain[:,step],targets,mvs,threshold)
                    for r in rows:r.update(training_type=family,model=family,step_index=step+1,post_step_index=step+1)
                    records.extend(rows)
                pd.DataFrame(records).to_csv(csv,mode='w' if start==0 else 'a',header=start==0,index=False)
                if start==0 or (start//64+1)%8==0 or start+64>=len(frame):print(f'[KCI] {family}: {distribution} {min(start+64,len(frame))}/{len(frame)}',flush=True)
            reference=np.concatenate(teachers,axis=0);gain=np.concatenate(dynamics,axis=2)
            assert np.isfinite(reference).all() and np.isfinite(gain).all()
            np.savez_compressed(dest/'gain_arrays.npz',teacher_gain=reference,dynamic_gain=gain,
                                physical_ANN_inputs=frame[native.TAB_INPUT_COLS].to_numpy(),target_cols=targets,mv_cols=mvs)
            detail=[]
            for target in targets:
                for mv in mvs:
                    row=native.count_row(cfg['exp_name'],distribution,target,mv,counts[(target,mv)],len(frame))
                    row.update(model=family,training_type=family,post_step_start=1,post_step_end=18,correct=row['tp']+row['tn']);detail.append(row)
            detail=pd.DataFrame(detail)
            independent=numpy_counts(reference,gain,threshold,targets,mvs)
            check=detail.merge(independent,on=['target','mv'],suffixes=('_native','_numpy'),validate='one_to_one')
            for col in ['possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain']:np.testing.assert_array_equal(check[f'{col}_native'],check[f'{col}_numpy'])
            recount=[];csv_rows=0
            for chunk in pd.read_csv(csv,chunksize=100_000,usecols=['target','mv','valid_teacher_gain','teacher_sign','dynamic_sign']):
                csv_rows+=len(chunk)
                chunk['tp']=chunk.valid_teacher_gain&chunk.teacher_sign.eq(1)&chunk.dynamic_sign.eq(1)
                chunk['tn']=chunk.valid_teacher_gain&chunk.teacher_sign.eq(-1)&chunk.dynamic_sign.eq(-1)
                chunk['valid']=chunk.valid_teacher_gain.astype(int)
                recount.append(chunk.groupby(['target','mv'])[['tp','tn','valid']].sum())
            assert csv_rows==len(frame)*144
            recount=pd.concat(recount).groupby(level=[0,1]).sum().reset_index()
            check=detail.merge(recount,on=['target','mv'],suffixes=('_native','_csv'),validate='one_to_one')
            np.testing.assert_array_equal(check.tp_native,check.tp_csv);np.testing.assert_array_equal(check.tn_native,check.tn_csv)
            np.testing.assert_array_equal(check.valid_comparisons,check.valid)
            detail.to_csv(dest/'gain_evaluation_detail.csv',index=False)
            write_json(sentinel,dict(passed=True,probe_points=len(frame),diagnostic_rows=csv_rows,threshold=threshold,
                                    NumPy_recount_passed=True,CSV_recount_passed=True,
                                    output_sha256={name:sha(dest/name) for name in ['gain_arrays.npz','gain_evaluation_detail.csv','gain_probe_diagnostics.csv']}))
            combined.extend(detail.to_dict('records'))
            del reference,gain,teachers,dynamics,records,rows
            gc.collect()
        del model
    detail=pd.DataFrame(combined);assert len(detail)==32
    prior=pd.read_csv(ROOT/'results/evaluation/gain_eval_all18/test/no_gain_seed42/gain_evaluation_detail.csv')
    check=detail.query("model == 'no_gain'").merge(prior,on=['distribution','target','mv'],suffixes=('_fresh','_prior'),validate='one_to_one')
    for col in ['probe_points','possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain']:np.testing.assert_array_equal(check[f'{col}_fresh'],check[f'{col}_prior'])
    for family in configs:
        c=detail[detail.model==family].merge(detail[detail.model=='no_gain'],on=['distribution','target','mv'],suffixes=('_model','_base'),validate='one_to_one')
        np.testing.assert_array_equal(c.valid_comparisons_model,c.valid_comparisons_base)
    detail.to_csv(KCI_OUTPUT/'gain_evaluation_detail.csv',index=False)
    detail.to_csv(OUTPUT/'gain_pair_kci.csv',index=False)
    summary=native.build_summary(detail)
    summary.to_csv(KCI_OUTPUT/'gain_evaluation_summary.csv',index=False)
    overall=detail.groupby(['model','distribution']).agg(correct=('correct','sum'),valid=('valid_comparisons','sum'),probe_points=('probe_points','first')).reset_index()
    overall['micro_KCI_percent']=100*overall.correct/overall.valid
    overall.to_csv(OUTPUT/'overall_kci.csv',index=False)
    print('PASS: all four models have fresh test KCI; baseline counts exactly reproduce the existing test KCI.',flush=True)


def report(protocol,device):
    metrics=pd.read_csv(OUTPUT/'test_metrics.csv');kci=pd.read_csv(OUTPUT/'gain_pair_kci.csv')
    base=metrics[metrics.model=='no_gain'][KEYS+['RMSE','n']].rename(columns={'RMSE':'NoGain_RMSE','n':'NoGain_n'})
    pairs=metrics.merge(base,on=KEYS,validate='many_to_one')
    assert len(pairs)==288 and pairs.n.eq(pairs.NoGain_n).all() and pairs.NoGain_RMSE.gt(0).all()
    pairs['relative_RMSE']=pairs.RMSE/pairs.NoGain_RMSE
    pairs['RMSE_change_percent']=100*(pairs.relative_RMSE-1)
    pairs.to_csv(OUTPUT/'paired_case_comparison.csv',index=False)
    build_report(OUTPUT)
    (OUTPUT/'rebuild_figures.py').write_text('''"""Rebuild all five test comparison tables from CSVs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.plotting.plot_best_models_test_tables import build_report
if __name__=='__main__':build_report(Path(__file__).resolve().parent)
''',encoding='utf-8')
    assert all(sha(ROOT/name)==h for name,h in protocol['protected_sha256'].items())
    assert all(sha(ROOT/name)==h for name,h in protocol['retained_comparison_sha256'].items())
    write_json(OUTPUT/'evaluation_validation.json',dict(passed=True,device=str(device),model_count=4,case_count=12,
               selection_uses_validation_only=True,test_metric_rows=288,horizon_rows=1728,independent_RMSE_rechecks=1824,
               prior_dynamic_cases_reproduced=36,prior_NoGain_KCI_counts_exact=True,KCI_probe_points=[4385,2692],
               KCI_diagnostic_rows_recounted=4*(4385+2692)*144,protected_inputs_unchanged=True,retained_outputs_unchanged=True,
               optimizer_updates=0,protected_sha256=protocol['protected_sha256'],visual_review='pending'))
    write_json(KCI_OUTPUT/'evaluation_validation.json',dict(passed=True,model_count=4,probe_points=[4385,2692],
               diagnostic_rows_recounted=4*(4385+2692)*144,protected_sha256=protocol['protected_sha256']))
    print('PASS: five requested test tables completed; protected inputs and previous comparisons unchanged.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','dynamic','kci','report'],default='all')
    args=parser.parse_args();torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs,contract,protocol=preflight()
    print('[Selection] Gain=250; SS-only=2.5; Gain+SS=2.5/2.5; No-Gain baseline; test only.',flush=True)
    if args.phase in {'all','dynamic'}:dynamic_test(configs,protocol['selected_models'],device)
    if args.phase in {'all','kci'}:kci_test(configs,contract,protocol['selected_models'],device)
    if args.phase in {'all','report'}:report(protocol,device)


if __name__=='__main__':main()
