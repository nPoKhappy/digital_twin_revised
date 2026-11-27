"""Build the full-trajectory validation report from 63 frozen evaluations."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.analysis import build_expanded_dg_validation_report as prior
from scripts.analysis import plot_jtice_validation as journal
from scripts.prediction.validate_hyperparameters import sha, write_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline_rawsplit'
OLD=ROOT/'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'
KEYS=['partition','case','target','evaluation']


def main():
    check=json.loads((OUT/'evaluation_validation.json').read_text(encoding='utf8'))
    protocol=json.loads((OUT/'run_protocol.json').read_text(encoding='utf8'))
    if not (check['passed'] and check['model_count']==63 and check['dynamic_metrics']==2268 and check['kci_components']==504):
        raise ValueError('63-model validation not complete')
    audit=json.loads((OUT/'independent_npz_audit.json').read_text(encoding='utf8'))
    if not (audit.get('passed') and audit.get('model_count')==63 and audit.get('rmse_checks')==2268 and audit.get('kci_pair_checks')==504 and audit.get('teacher_arrays_consistent')):
        raise ValueError('Independent NPZ validation has not passed')
    for path,digest in protocol['sources_sha256'].items():
        if sha(ROOT/path)!=digest: raise ValueError(f'Validation source changed: {path}')
    dependency=json.loads((OUT/'dependency_snapshot.json').read_text(encoding='utf8'))
    if sha(ROOT/dependency['path'])!=dependency['sha256']:
        raise ValueError('Imported teacher/loader dependency changed during validation')
    meta_path=OLD/'model_parameter_metadata.csv'
    old_prov=json.loads((OLD/'provenance.json').read_text(encoding='utf8'))
    if sha(meta_path)!=old_prov['outputs_sha256'][meta_path.name]: raise ValueError('Parameter labels mismatch')
    meta=pd.read_csv(meta_path)
    jobs=protocol['jobs']
    if len(meta)!=21 or set(meta.model)!={j['alias'] for j in jobs}: raise ValueError('Wrong model metadata')
    for job in jobs:
        cfg=yaml.safe_load((ROOT/job['config']).read_text(encoding='utf8'))
        item=meta.loc[meta.model.eq(job['alias'])].iloc[0]
        if item.family!=job['family']: raise ValueError('Family metadata mismatch')
        family=job['family']
        if family=='difficult_gradnorm': value=float(cfg['training']['dg_target_ratio'])
        elif family=='ss_only': value=float(cfg['training']['ss_loss_weight'])
        elif family in ('gain','gain_ss'): value=float(cfg['training']['gain_loss_weight'])
        else: value=0.0
        if not math.isclose(float(item.value),value): raise ValueError(f'Weight metadata mismatch {job["alias"]}')
        # Derive display labels from the actual config, not alias parsing.
        if job['seed']==42:
            mask=meta.model.eq(job['alias'])
            gain=float(cfg['training'].get('gain_loss_weight',0))
            ss=float(cfg['training'].get('ss_loss_weight',0))
            if family=='no_gain': short,full='NG','No-Gain: Gain = 0, SS = 0'
            elif family=='difficult_gradnorm':
                short=f'D ρ={value:g}'
                full=f'Difficult-GradNorm: ρ = {value:g}, Gain = {gain:g}, SS = {ss:g}'
            else:
                prefix={'gain':'G','ss_only':'S','gain_ss':'GS'}[family]
                label={'gain':'Gain','ss_only':'SS-only','gain_ss':'Gain+SS'}[family]
                short=f'{prefix} {value:g}'
                full=f'{label}: Gain = {gain:g}, SS = {ss:g}'
            meta.loc[mask,'short']=short; meta.loc[mask,'full']=full
    metrics=pd.read_csv(OUT/'validation_metrics.csv')
    raw=pd.read_csv(OUT/'kci_components.csv')
    baseline=(metrics[(metrics.seed==42)&metrics.model.eq('no_gain')][KEYS+['n','RMSE']]
              .rename(columns={'n':'NoGain_seed42_n','RMSE':'NoGain_seed42_RMSE'}))
    if len(baseline)!=36 or baseline.duplicated(KEYS).any(): raise ValueError('Baseline coverage')
    components=metrics.merge(baseline,on=KEYS,validate='many_to_one')
    if len(components)!=63*36 or not components.n.eq(components.NoGain_seed42_n).all() or not components.NoGain_seed42_RMSE.gt(0).all(): raise ValueError('RMSE alignment error')
    components['relative_RMSE']=components.RMSE/components.NoGain_seed42_RMSE
    family={j['alias']:j['family'] for j in jobs}
    components['family']=components.model.map(family)
    cols=['seed','model','family',*KEYS,'n','RMSE','NoGain_seed42_n','NoGain_seed42_RMSE','relative_RMSE']
    components=components[cols]
    if components.duplicated(['seed','model',*KEYS]).any() or not np.isfinite(components.relative_RMSE).all() or not components.relative_RMSE.gt(0).all(): raise ValueError('Invalid relative RMSE')
    if len(raw)!=63*8 or raw.duplicated(['seed','model','distribution','target','mv']).any(): raise ValueError('KCI coverage')
    if not raw.groupby(['distribution','target','mv']).valid_comparisons.nunique().eq(1).all(): raise ValueError('KCI denominators differ')
    raw['correct']=raw.tp+raw.tn
    pooled=raw.groupby(['seed','model','distribution'],as_index=False).agg(correct=('correct','sum'),valid=('valid_comparisons','sum'))
    pooled['KCI_percent']=100*pooled.correct/pooled.valid
    pooled['region']=pooled.distribution.map({'in_range':'internal','soft_ood':'external'})
    region=components.groupby(['seed','model','family','partition'],as_index=False).agg(AvgRelRMSE=('relative_RMSE',lambda x:float(np.exp(np.log(x).mean()))),component_count=('relative_RMSE','size'))
    if not region.component_count.eq(18).all(): raise ValueError('Expected 18 components per region')
    coords=region.rename(columns={'partition':'region'}).merge(pooled,on=['seed','model','region'],validate='one_to_one')
    coords=coords[['seed','model','family','region','KCI_percent','AvgRelRMSE','correct','valid']]
    if len(coords)!=126: raise ValueError('Wrong regional coverage')
    baseline_coords=coords[(coords.seed==42)&coords.model.eq('no_gain')]
    np.testing.assert_allclose(baseline_coords.AvgRelRMSE,1.0,rtol=1e-12)
    wide=coords.pivot(index=['seed','model','family'],columns='region',values='AvgRelRMSE').reset_index()
    wide['overall_50_50']=np.sqrt(wide.internal*wide.external)
    wide=wide.sort_values(['overall_50_50','model','seed']).reset_index(drop=True)
    wide.insert(0,'rank_single_run',range(1,64))
    summary=wide.groupby(['family','model'],as_index=False).agg(mean=('overall_50_50','mean'),sd=('overall_50_50','std'),minimum=('overall_50_50','min'),maximum=('overall_50_50','max'),n_seeds=('seed','nunique'))
    summary=summary.sort_values(['mean','model']).reset_index(drop=True); summary.insert(0,'rank_mean',range(1,22))
    for frame,name in [(baseline,'baseline_raw_rmse.csv'),(components,'relative_rmse_components.csv'),(raw,'kci_components.csv'),(coords,'regional_coordinates_63_models.csv'),(wide,'scores_by_model_seed.csv'),(summary,'scores_by_family_model.csv'),(meta,'model_parameter_metadata.csv')]: frame.to_csv(OUT/name,index=False,float_format='%.17g')
    models=meta.astype(object).where(meta.notna(),None).to_dict('records')
    html=prior.make_html(coords,raw,models)
    old_axis='[Math.max(0,lo-p),y?hi+p:Math.min(100,hi+p)]'
    if html.count(old_axis)!=1: raise ValueError('Interactive axis mismatch')
    html=html.replace(old_axis,'[Math.max(0,lo-p),y?hi+p:100]')
    (OUT/'regional_validation_nogain42_by_seed.html').write_text(html,encoding='utf8')
    xlim=(max(0.,min(85.8,math.floor(float(coords.KCI_percent.min()))-1)),100.2)
    ylim=(max(0.,min(.35,math.floor(float(coords.AvgRelRMSE.min())*.9*20)/20)),max(2.85,math.ceil(float(coords.AvgRelRMSE.max())*1.05*20)/20))
    bounds={'x':xlim,'y':ylim}
    fig=OUT/'figures';fig.mkdir(exist_ok=True); journal.OUT=fig
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'pdf.fonttype':42,'ps.fonttype':42})
    for expanded in (False,True):
        suffix='expanded_y' if expanded else 'linear'
        marks=journal.draw(coords,meta,journal.FAMILIES,f'nogain42_4x2_{suffix}',expanded,bounds)
        if len(marks)!=144: raise ValueError('Wrong marker count')
        for part,group in [(1,journal.FAMILIES[:2]),(2,journal.FAMILIES[2:])]: journal.draw(coords,meta,group,f'nogain42_part{part}_{suffix}',expanded,bounds)
    jtice=OUT/'jtice_figures'; jtice.mkdir(exist_ok=True); journal.OUT=jtice
    for expanded in (False,True):
        suffix='expanded_y' if expanded else 'linear'
        for label,group in [('all_families',journal.FAMILIES),('part1',journal.FAMILIES[:2]),('part2',journal.FAMILIES[2:])]:
            journal.draw(coords,meta,group,f'validation_{label}_{suffix}',expanded,bounds)
            (jtice/f'validation_{label}_{suffix}_caption.txt').write_text('Validation of three seeds. AvgRelRMSE is normalized to the same-case, same-target, same-horizon No-Gain seed 42 RMSE; KCI is agreement with the shared ANN teacher. '+('A piecewise-linear y axis is used.' if expanded else 'All panels use a shared linear y axis.')+'\n',encoding='utf8')
    report={'passed':True,'model_count':63,'regional_points':126,'components':2268,'kci_components':504,'seeds':[42,43,44],'newly_evaluated_models':63,'baseline_seed':42,'same_x_y_axes_all_panels':True,'no_jitter':True,'optimizer_updates':0}
    write_json(OUT/'analysis_validation.json',report)
    (OUT/'README.md').write_text('# Raw-split validation\n\n63 個重新訓練模型使用相同 validation 案例與 ANN-reference 穩態探點重新推論；AvgRelRMSE 使用新 No-Gain seed 42 的相同案例、輸出、時域 RMSE 作基準，每區域對18個比值取幾何平均。KCI 和動態 RMSE 使用不同資料；KCI 衡量相對 ANN 的方向一致，並非直接證明 Aspen 真實方向。R5-5 約 33.19% 原始時間點超出核心範圍。\n\n動態預測使用已記錄的未來 decoder 操作變數；H1/H18 的視窗使用真實過去歷史，TS 從最初18點真實歷史遞迴至結束。此驗證不證明閉迴路表現。\n',encoding='utf8')
    names=['validation_metrics.csv','kci_components.csv','baseline_raw_rmse.csv','relative_rmse_components.csv','regional_coordinates_63_models.csv','scores_by_model_seed.csv','scores_by_family_model.csv','model_parameter_metadata.csv','regional_validation_nogain42_by_seed.html','analysis_validation.json','independent_npz_audit.json','README.md']
    names += [str(p.relative_to(OUT)).replace('\\','/') for d in (fig,jtice) for p in sorted(d.rglob('*')) if p.is_file()]
    audit_script=ROOT/'scripts/analysis/audit_rawsplit_validation.py'
    worker_script=ROOT/'scripts/prediction/validate_rawsplit_seed_worker.py'
    provenance={'sources_sha256':{str((OUT/'run_protocol.json').relative_to(ROOT)):sha(OUT/'run_protocol.json'),str((OUT/'evaluation_validation.json').relative_to(ROOT)):sha(OUT/'evaluation_validation.json'),str((OUT/'dependency_snapshot.json').relative_to(ROOT)):sha(OUT/'dependency_snapshot.json'),dependency['path']:dependency['sha256'],str(meta_path.relative_to(ROOT)):sha(meta_path),str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__).resolve()),str(audit_script.relative_to(ROOT)):sha(audit_script),str(worker_script.relative_to(ROOT)):sha(worker_script)},'outputs_sha256':{name:sha(OUT/name) for name in names},'plot_bounds':bounds,'baseline':'new rawsplit No-Gain seed42','model_count':63}
    write_json(OUT/'provenance.json',provenance)
    print(f'Completed validation report: {OUT}',flush=True)


if __name__=='__main__': main()
