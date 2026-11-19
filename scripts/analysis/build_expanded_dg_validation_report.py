"""Build the fixed-No-Gain-42 validation report after the DG grid expansion.

Uses the immutable historical summary for 48 non-DG and 3 retained DG runs,
and the independently validated 12 new DG runs. No predictions are recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from scripts.plotting.render_retrain_validation_html import PAGE
from scripts.plotting import retrain_parameter_ui as ui

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/visualization'
HISTORICAL = BASE / 'retrain_lhs10_23_63_models_validation_nogain42_baseline_previous_before_dg_expansion'
EXPANDED = BASE / 'retrain_lhs10_23_expanded_dg_validation'
DEFAULT_OUT = BASE / 'retrain_lhs10_23_63_models_validation_nogain42_expanded_staging'
OLD_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
NEW_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
HISTORICAL_TEST = BASE / 'retrain_lhs10_23_selected_test_nogain42'
HISTORICAL_BASELINE_NAME = 'retrain_lhs10_23_63_models_validation_nogain42_baseline'
SEEDS = (42, 43, 44)
NEW_ALIASES = ('dg_ratio0p15', 'dg_ratio1p5', 'dg_ratio15', 'dg_ratio150')
KEEP_DG = 'dg_ratio0p015'
REGIONS = {'internal': 'in_range', 'external': 'soft_ood'}
CASES = {'internal': ('R5-3','R5-4','R5-5'), 'external': ('R5-27','R5-28','R5-29')}
TARGETS = ('B35_H2S','B35_SO2')
MVS = ('air2_SP','HEATER2_output_T_SP')
EVALUATIONS = ('H1','H18','TS')
KEYS = ['partition','case','target','evaluation']
FAMILIES = [('gain','Gain'),('ss_only','SS-only'),('gain_ss','Gain + SS'),('difficult_gradnorm','Difficult-GradNorm')]
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
MARKERS = {42:'o',43:'s',44:'^'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_meta():
    old = read_json(OLD_MANIFEST)
    new = read_json(NEW_MANIFEST)
    active = [j for j in old if j['family'] != 'difficult_gradnorm' or j['alias'] == KEEP_DG] + new
    expected_new = {(s,a) for s in SEEDS for a in NEW_ALIASES}
    if len(new) != 12 or {(j['seed'],j['alias']) for j in new} != expected_new or len(active) != 63:
        raise ValueError('Unexpected active configuration roster')
    seed_models = {}
    for seed in SEEDS:
        jobs = [j for j in active if j['seed'] == seed]
        items = {}
        for j in jobs:
            cfgpath=ROOT/j['config']; cfg=yaml.safe_load(cfgpath.read_text(encoding='utf-8-sig'))
            if sha(cfgpath)!=j['config_sha256'] or cfg['exp_name']!=j['exp_name'] or cfg['training']['seed']!=seed:
                raise ValueError(f'Active configuration differs: {cfgpath}')
            family=j['family']; gain=float(cfg['training'].get('gain_loss_weight',0)); ss=float(cfg['training'].get('ss_loss_weight',0)); cap=None
            if family=='difficult_gradnorm':
                args=j['cli_args']; ratio=float(args[args.index('--target-ratio')+1]); cap=float(args[args.index('--max-scale')+1])
                expected_cap=50 if j['alias']==KEEP_DG else 1e7
                if cap!=expected_cap or gain!=.05 or ss!=0:
                    raise ValueError(f'Unexpected DG training setup: {j["alias"]}')
                value=ratio; short=f'D ρ={ratio:g}'; full=f'Difficult-GradNorm · ρ = {ratio:g}（Gain = {gain:g}，SS = {ss:g}）'
            elif family=='no_gain':
                if gain!=0 or ss!=0: raise ValueError('Unexpected No-Gain weights')
                value,short,full=0.,'NG','No-Gain · Gain = 0，SS = 0'
            else:
                value=ss if family=='ss_only' else gain
                if family=='gain_ss' and gain!=ss: raise ValueError('Gain+SS weights differ')
                prefix={'gain':'G','ss_only':'S','gain_ss':'GS'}[family]
                label={'gain':'Gain','ss_only':'SS-only','gain_ss':'Gain+SS'}[family]
                short=f'{prefix} {value:g}'; full=f'{label} · Gain = {gain:g}，SS = {ss:g}'
            if j['alias'] in items: raise ValueError('Duplicate model alias')
            items[j['alias']]=dict(model=j['alias'],family=family,value=value,short=short,full=full,max_scale=cap)
        if len(jobs)!=21 or len(items)!=21: raise ValueError(f'Wrong model count for seed {seed}')
        seed_models[seed]=items
    if seed_models[42]!=seed_models[43] or seed_models[42]!=seed_models[44]:
        raise ValueError('Parameter settings differ across seeds')
    order={'no_gain':0,'gain':1,'ss_only':2,'gain_ss':3,'difficult_gradnorm':4}
    models=sorted(seed_models[42].values(),key=lambda m:(order[m['family']],m['value']))
    for i,item in enumerate(models,1): item['number']=i
    return models,active


def validate_inputs(models,active):
    old_prov=read_json(HISTORICAL/'provenance.json'); old_report=read_json(HISTORICAL/'analysis_validation.json')
    if not old_report.get('passed') or old_report.get('model_count')!=63 or old_prov.get('baseline_seed')!=42:
        raise ValueError('Historical fixed-baseline analysis is not complete')
    for name,digest in old_prov['outputs_sha256'].items():
        if sha(HISTORICAL/name).lower()!=digest.lower(): raise ValueError(f'Historical summary changed: {name}')
    for name,digest in old_prov['sources_sha256'].items():
        if sha(ROOT/name).lower()!=digest.lower(): raise ValueError(f'Historical validation source changed: {name}')
    new_report=read_json(EXPANDED/'evaluation_validation.json'); protocol=read_json(EXPANDED/'run_protocol.json')
    if not (new_report.get('passed') and new_report.get('model_count')==12 and new_report.get('seeds')==list(SEEDS)
            and new_report.get('dynamic_metrics')==432 and new_report.get('kci_components')==96
            and new_report.get('source_hashes_unchanged')):
        raise ValueError('Expanded DG validation is incomplete')
    for name,digest in protocol['sources_sha256'].items():
        if sha(ROOT/name).lower()!=digest.lower(): raise ValueError(f'Expanded validation input changed: {name}')
    checked=[]; protocol_sha=sha(EXPANDED/'run_protocol.json')
    for j in protocol['jobs']:
        folder=EXPANDED/'model_data'/f"seed{j['seed']}"/j['alias']
        marker=read_json(folder/'complete.json')
        required={'validation_metrics.csv','kci_components.csv','gain_in_range.npz','gain_soft_ood.npz',
                  *[f'{case}_prediction.npz' for cases in CASES.values() for case in cases]}
        if marker['run_protocol_sha256']!=protocol_sha or not required.issubset(marker['outputs']):
            raise ValueError(f'Expanded DG result incomplete: {folder}')
        for name,digest in marker['outputs'].items():
            if sha(folder/name)!=digest: raise ValueError(f'Expanded DG output changed: {folder/name}')
        checked.append((j['seed'],j['alias']))
    if set(checked)!={(s,a) for s in SEEDS for a in NEW_ALIASES} or len(checked)!=12:
        raise ValueError('Expanded DG model roster differs')
    return old_prov,protocol


def aggregate(models):
    aliases={m['model']:m['family'] for m in models}; retained=set(aliases)-set(NEW_ALIASES)
    old=pd.read_csv(HISTORICAL/'relative_rmse_components.csv')
    old=old[old.model.isin(retained)].copy()
    if len(old)!=51*36: raise ValueError('Historical component subset is incomplete')
    baseline=pd.read_csv(HISTORICAL/'baseline_raw_rmse.csv')
    if len(baseline)!=36 or baseline.duplicated(KEYS).any(): raise ValueError('Historical reference is invalid')
    new=pd.read_csv(EXPANDED/'validation_metrics.csv')
    if len(new)!=432 or set(new.model)!=set(NEW_ALIASES): raise ValueError('Expanded metric rows missing')
    new=new.merge(baseline,on=KEYS,validate='many_to_one')
    if not new.n.eq(new.NoGain_seed42_n).all() or not new.NoGain_seed42_RMSE.gt(0).all():
        raise ValueError('Expanded baseline matching failed')
    new['relative_RMSE']=new.RMSE/new.NoGain_seed42_RMSE
    new['family']='difficult_gradnorm'
    columns=['seed','model','family',*KEYS,'n','RMSE','NoGain_seed42_n','NoGain_seed42_RMSE','relative_RMSE']
    components=pd.concat([old[columns],new[columns]],ignore_index=True)
    expected={(s,m,p,c,t,e) for s in SEEDS for m in aliases for p,cases in CASES.items() for c in cases for t in TARGETS for e in EVALUATIONS}
    actual=set(components[['seed','model',*KEYS]].itertuples(index=False,name=None))
    if len(components)!=2268 or actual!=expected or not np.isfinite(components.relative_RMSE).all() or not components.relative_RMSE.gt(0).all():
        raise ValueError('Combined RMSE component coverage is invalid')
    historical_kci=[]
    for seed in SEEDS:
        path=BASE/f'retrain_lhs10_23_21_models_validation_seed{seed}'/'kci_components.csv'
        df=pd.read_csv(path); df=df[df.model.isin(retained)].copy(); df['seed']=seed
        if len(df)!=17*8: raise ValueError(f'Historical KCI component coverage differs for seed {seed}')
        historical_kci.append(df)
    kci=pd.concat([*historical_kci,pd.read_csv(EXPANDED/'kci_components.csv')],ignore_index=True)
    if len(kci)!=504 or kci.duplicated(['seed','model','distribution','target','mv']).any(): raise ValueError('Combined KCI component coverage is invalid')
    expected_kci={(s,m,d,t,mv) for s in SEEDS for m in aliases for d in REGIONS.values() for t in TARGETS for mv in MVS}
    if set(kci[['seed','model','distribution','target','mv']].itertuples(index=False,name=None))!=expected_kci:
        raise ValueError('Unexpected model, seed or output/MV KCI pairing')
    if not kci.groupby(['distribution','target','mv']).valid_comparisons.nunique().eq(1).all():
        raise ValueError('Teacher-sign KCI denominators differ across models or seeds')
    for row in kci.itertuples(index=False):
        expected_probes={'in_range':4404,'soft_ood':2585}[row.distribution]
        if int(row.probe_points)!=expected_probes or int(row.possible_comparisons)!=expected_probes*2*18:
            raise ValueError('Unexpected KCI probe or possible comparison count')
        if not 0<int(row.valid_comparisons)<=int(row.possible_comparisons): raise ValueError('Invalid KCI denominator')
        if int(row.tp)+int(row.tn)+int(row.fp)+int(row.fn)+int(row.zero_gain)!=int(row.valid_comparisons):
            raise ValueError('KCI counts do not sum to valid')
    kci['correct']=kci.tp+kci.tn
    grouped=kci.groupby(['seed','model','distribution'],as_index=False).agg(correct=('correct','sum'),valid=('valid_comparisons','sum'))
    grouped['KCI_percent']=100*grouped.correct/grouped.valid
    grouped['region']=grouped.distribution.map({v:k for k,v in REGIONS.items()})
    regions=components.groupby(['seed','model','family','partition'],as_index=False).agg(
        AvgRelRMSE=('relative_RMSE',lambda x:float(np.exp(np.log(x).mean()))),component_count=('relative_RMSE','size'))
    if not regions.component_count.eq(18).all(): raise ValueError('Wrong number of per-region RMSE ratios')
    coords=regions.rename(columns={'partition':'region'}).merge(grouped,on=['seed','model','region'],validate='one_to_one')
    coords=coords[['seed','model','family','region','KCI_percent','AvgRelRMSE','correct','valid']]
    old_coord=pd.read_csv(HISTORICAL/'regional_coordinates_63_models.csv')
    old_coord=old_coord[old_coord.model.isin(retained)]
    match=coords[coords.model.isin(retained)].merge(old_coord,on=['seed','model','family','region'],suffixes=('_new','_old'),validate='one_to_one')
    if len(match)!=102: raise ValueError('Retained old coordinates do not match')
    for col in ('KCI_percent','AvgRelRMSE','correct','valid'):
        np.testing.assert_allclose(match[f'{col}_new'],match[f'{col}_old'],rtol=1e-12,atol=1e-12)
    wide=coords.pivot(index=['seed','model','family'],columns='region',values='AvgRelRMSE').reset_index()
    wide['overall_50_50']=np.sqrt(wide.internal*wide.external)
    wide=wide.sort_values(['overall_50_50','model','seed']).reset_index(drop=True);wide.insert(0,'rank_single_run',range(1,64))
    summary=wide.groupby(['family','model'],as_index=False).agg(mean=('overall_50_50','mean'),sd=('overall_50_50','std'),minimum=('overall_50_50','min'),maximum=('overall_50_50','max'),n_seeds=('seed','nunique'))
    summary=summary.sort_values(['mean','model']).reset_index(drop=True);summary.insert(0,'rank_mean',range(1,22))
    if len(coords)!=126 or len(wide)!=63 or len(summary)!=21 or not summary.n_seeds.eq(3).all(): raise ValueError('Invalid combined summary sizes')
    return baseline,components,kci,coords,wide,summary


def historical_test_mapping():
    """Validate old test's frozen selection against the archived validation files."""
    plan=read_json(HISTORICAL_TEST/'validation_selection_plan.json')
    selection=read_json(HISTORICAL_TEST/'selected_models.json')
    protocol=read_json(HISTORICAL_TEST/'run_protocol.json')
    if (selection['selection_plan']!=plan or selection['selection_plan_sha256']!=sha(HISTORICAL_TEST/'validation_selection_plan.json')
            or protocol['selection_sha256']!=sha(HISTORICAL_TEST/'selected_models.json')):
        raise ValueError('Historical test frozen selection hash chain differs')
    sources=[('validation_selection_plan.json',plan['provenance']['sources_sha256']),
             ('selected_models.json embedded plan',selection['selection_plan']['provenance']['sources_sha256']),
             ('run_protocol.json',protocol['sources_sha256'])]
    mapping=[]
    for document,entries in sources:
        found=0
        for name,digest in entries.items():
            normalized=name.replace('\\','/')
            marker=f'results/visualization/{HISTORICAL_BASELINE_NAME}/'
            if not normalized.startswith(marker): continue
            relative=normalized[len(marker):];archived=HISTORICAL/relative
            if sha(archived).lower()!=digest.lower(): raise ValueError(f'Historical test archived source differs: {archived}')
            mapping.append(dict(document=document,previous_path=normalized,
                                archived_path=str(archived.relative_to(ROOT)).replace('\\','/'),sha256=digest))
            found+=1
        if found==0: raise ValueError(f'Historical test source references missing in {document}')
    return dict(passed=True,previous_prefix=f'results/visualization/{HISTORICAL_BASELINE_NAME}/',
                archive_prefix=str(HISTORICAL.relative_to(ROOT)).replace('\\','/')+'/',
                frozen_selection_hash_chain_valid=True,source_references_checked=len(mapping),
                historical_test_files_modified=False,mappings=mapping)


def make_html(data,kci,models):
    datasets={}
    for seed in SEEDS:
        points=[dict(model=r.model,family=r.family,partition=r.region,x=float(r.KCI_percent),y=float(r.AvgRelRMSE),correct=int(r.correct),valid=int(r.valid)) for r in data[data.seed.eq(seed)].itertuples(index=False)]
        details=[dict(model=r.model,partition={v:k for k,v in REGIONS.items()}[r.distribution],mv=r.mv,target=r.target,correct=int(r.tp+r.tn),valid=int(r.valid_comparisons)) for r in kci[kci.seed.eq(seed)].itertuples(index=False)]
        datasets[str(seed)]=dict(points=points,details=details)
    previous=ui.model_parameters;ui.model_parameters=lambda root:models
    try: page=ui.render_parameter_page(PAGE,datasets,ROOT)
    finally: ui.model_parameters=previous
    replacements={
      '準確度與增益方向':'準確度與增益方向（共同基準）',
      '左側內插、右側外插：兩個座標分別來自不同的驗證資料集。':'所有 seed 均以 No-Gain seed 42 為基準。左側內插、右側外插：兩個座標分別來自不同的驗證資料集。',
      '相同案例及目前所選 seed 的 No-Gain RMSE':'相同案例、輸出及評估方式的 No-Gain seed 42 RMSE',
      'No-Gain = 1，較低表示相對誤差較小。':'No-Gain seed 42 = 1；seed 43、44 仍各自以 seed 42 為分母，可能不等於 1。較低表示相對誤差較小。',
      '圖上虛線為 No-Gain 相對誤差基準。':'圖上虛線為 No-Gain seed 42 的共同相對誤差基準。',
      'ANN-reference KCI (%)':'KCI(%)'}
    for old,new in replacements.items():
        if old not in page: raise ValueError(f'Expected report wording not found: {old}')
        page=page.replace(old,new)
    return page


def figure(data,models,out,start,stop,stem,expanded,xlim,ylim):
    families=FAMILIES[start:stop];rows=len(families)
    fig=plt.figure(figsize=(9.2,rows*2.8+.9))
    grid=fig.add_gridspec(rows,3,width_ratios=(1,1,.31),left=.08,right=.99,top=.86,
                          bottom=.055 if rows==4 else .085,wspace=.16,hspace=.40)
    split=1.35; lower,upper=ylim
    if expanded:
        if not lower < split < upper: raise ValueError('Expanded scale split is outside y limits')
        slope=(split-lower)/(2.6*(upper-split))
        def forward(v):
            v=np.asarray(v);return np.where(v<=split,v,split+(v-split)*slope)
        def inverse(v):
            v=np.asarray(v);return np.where(v<=split,v,split+(v-split)/slope)
    plotted=[]
    for row,(family,name) in enumerate(families):
        index=start+row; choices=sorted((m for m in models if m['family']==family),key=lambda m:m['value'])
        if len(choices)!=5: raise ValueError(f'Expected five parameters for {family}')
        for col,region in enumerate(('internal','external')):
            ax=fig.add_subplot(grid[row,col]);ax.set_xlim(*xlim);ax.set_ylim(*ylim)
            if expanded:
                ax.set_yscale('function',functions=(forward,inverse));ax.set_ylim(*ylim)
                lo=[v for v in (.4,.6,.8,1.,1.2,1.35) if lower<=v<=upper]
                hi=[round(float(v),1) for v in np.linspace(split,upper,4)[1:] if round(float(v),1)<=upper]
                ticks=sorted(set(lo+hi));ax.set_yticks(ticks);ax.set_yticklabels([f'{x:g}' for x in ticks])
                ax.axhline(split,color='#999999',linewidth=.65,linestyle=':',zorder=1)
            ax.set_title(f'({"abcdefgh"[index*2+col]})  {name}  |  {"Interpolation" if col==0 else "Extrapolation"}',loc='left',fontsize=9,fontweight='bold',pad=6)
            ax.axhline(1,color='#777777',linewidth=.7,linestyle='--',zorder=1)
            ax.grid(color='#d4d4d4',linewidth=.45,alpha=.7,zorder=0)
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7.5,length=3,pad=2)
            if col==0: ax.set_ylabel('AvgRelRMSE',fontsize=9)
            if row==rows-1: ax.set_xlabel('KCI(%)',fontsize=9)
            subset=data[data.region.eq(region)]
            for item,color in [(m,c) for m,c in zip(choices,COLORS)]+[(dict(model='no_gain'),'#555555')]:
                for seed in SEEDS:
                    point=subset[subset.model.eq(item['model']) & subset.seed.eq(seed)]
                    if len(point)!=1: raise ValueError('Plot data missing a model/seed/region')
                    x=float(point.KCI_percent.iloc[0]);y=float(point.AvgRelRMSE.iloc[0])
                    if not(xlim[0]<=x<=xlim[1] and ylim[0]<=y<=ylim[1]): raise ValueError('Plot limit omits a point')
                    ax.scatter(x,y,marker=MARKERS[seed],s=35 if item['model']=='no_gain' else 32,facecolor=color,edgecolor='white',linewidth=.5,zorder=4)
                    plotted.append((family,region,item['model'],seed,x,y))
        legend=fig.add_subplot(grid[row,2]);legend.axis('off')
        legend.text(.03,.98,'ρ' if family=='difficult_gradnorm' else 'Loss weight',fontsize=8,fontweight='bold',va='top',transform=legend.transAxes)
        for j,(item,color) in enumerate(zip(choices,COLORS)):
            yy=.79-j*.135;legend.plot([.14],[yy],color=color,marker='o',markersize=6,linestyle='',transform=legend.transAxes)
            legend.text(.28,yy,f"{item['value']:g}",fontsize=8,va='center',transform=legend.transAxes)
    handles=[Line2D([],[],marker=MARKERS[s],color='#555555',markersize=6,ls='',markeredgecolor='white',markeredgewidth=.4,label=f'Seed {s}') for s in SEEDS]
    handles.append(Line2D([],[],marker='o',color='#555555',markersize=6,ls='',markeredgecolor='white',markeredgewidth=.4,label='No-Gain reference'))
    fig.legend(handles=handles,loc='upper center',ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(.5,.985),handletextpad=.4,columnspacing=1.3)
    fig.text(.5,.938,'Baseline: No-Gain seed 42',fontsize=8,ha='center',va='center')
    fig.text(.5,.902,'Common nonlinear Y: expanded below 1.35; no values omitted' if expanded else 'Common linear X and Y scales across all panels',fontsize=8,ha='center',va='center')
    if len(plotted)!=rows*2*18: raise ValueError('Unexpected number of displayed markers')
    for extension in ('png','pdf'): fig.savefig(out/f'{stem}.{extension}',dpi=350,facecolor='white')
    plt.close(fig)
    return plotted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT.resolve()): raise ValueError('Output must remain in the workspace')
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Refusing to overwrite nonempty directory: {out}')
    models,active=read_meta();old_prov,new_protocol=validate_inputs(models,active)
    historical_mapping=historical_test_mapping()
    baseline,components,kci,coords,wide,summary=aggregate(models)
    source_paths=[HISTORICAL/n for n in ('provenance.json','analysis_validation.json','baseline_raw_rmse.csv','relative_rmse_components.csv','regional_coordinates_63_models.csv')]
    source_paths += [EXPANDED/n for n in ('run_protocol.json','evaluation_validation.json','validation_metrics.csv','kci_components.csv')]
    source_paths += [OLD_MANIFEST,NEW_MANIFEST,Path(__file__),ROOT/'scripts/plotting/retrain_parameter_ui.py',ROOT/'scripts/plotting/render_retrain_validation_html.py']
    source_paths += [BASE/f'retrain_lhs10_23_21_models_validation_seed{s}'/'kci_components.csv' for s in SEEDS]
    source_paths += [HISTORICAL_TEST/n for n in ('validation_selection_plan.json','selected_models.json','run_protocol.json')]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    out.mkdir(parents=True,exist_ok=True);fig_dir=out/'figures';fig_dir.mkdir()
    frames=[(baseline,'baseline_raw_rmse.csv'),(components,'relative_rmse_components.csv'),(kci,'kci_components.csv'),
            (coords,'regional_coordinates_63_models.csv'),(wide,'scores_by_model_seed.csv'),(summary,'scores_by_family_model.csv')]
    for frame,name in frames:frame.to_csv(out/name,index=False,float_format='%.17g')
    pd.DataFrame(models).to_csv(out/'model_parameter_metadata.csv',index=False,float_format='%.17g')
    (out/'historical_path_mapping_verification.json').write_text(json.dumps(historical_mapping,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'regional_validation_nogain42_by_seed.html').write_text(make_html(coords,kci,models),encoding='utf-8')
    xlim=(max(0.,min(85.8,math.floor(float(coords.KCI_percent.min()))-1)),100.2)
    ylim=(max(0.,min(.35,math.floor(float(coords.AvgRelRMSE.min())*.9*20)/20)),max(2.85,math.ceil(float(coords.AvgRelRMSE.max())*1.05*20)/20))
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    marks=figure(coords,models,fig_dir,0,4,'nogain42_4x2_linear',False,xlim,ylim)
    for start,stop,stem in [(0,2,'nogain42_part1_linear'),(2,4,'nogain42_part2_linear')]:figure(coords,models,fig_dir,start,stop,stem,False,xlim,ylim)
    figure(coords,models,fig_dir,0,4,'nogain42_4x2_expanded_y',True,xlim,ylim)
    for start,stop,stem in [(0,2,'nogain42_part1_expanded_y'),(2,4,'nogain42_part2_expanded_y')]:figure(coords,models,fig_dir,start,stop,stem,True,xlim,ylim)
    unique={(seed,model,region,x,y) for _,region,model,seed,x,y in marks}
    expected_points={(int(r.seed),r.model,r.region,float(r.KCI_percent),float(r.AvgRelRMSE)) for r in coords.itertuples(index=False)}
    if len(marks)!=144 or len(unique)!=126 or unique!=expected_points:
        raise ValueError('Figure did not preserve all unique model coordinates')
    for name,digest in source_hashes.items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Source changed during report generation: {name}')
    readme=('''# 擴充 DG 權重後的驗證圖（共同 No-Gain seed 42 基準）

這份結果共 63 個動態模型、126 個區域座標。沿用歷史結果中 48 個 No-Gain／Gain／SS-only／Gain+SS 模型及 3 個 DG ρ = 0.015 模型；另外 12 個 DG 模型使用新 checkpoint 重新驗證。僅重新整理已完成的驗證輸出，未在本步驟重新訓練模型。

DG 的 ρ = 0.015 訓練時 max-scale = 50；ρ = 0.15、1.5、15、150 使用 max-scale = 10,000,000，因此不同 ρ 並非只有比例參數不同，不能將差異單獨歸因於 ρ。此設定保留於 `model_parameter_metadata.csv` 和 provenance 中。

橫軸為 ANN-reference KCI（穩態 in-range／soft-OOD probes），縱軸為對應動態案例各 18 項相對 No-Gain seed 42 RMSE 的幾何平均；內插 R5-3/4/5、外插 R5-27/28/29，兩軸使用不同資料。R5-5 約 33.19% 原始時間點超出核心範圍。KCI 為 ANN 增益方向一致性，不是 Aspen 物理增益或閉迴路驗證。

兩種 4×2 圖及對應分頁各使用一致的全圖 X、Y 範圍。linear 是線性尺度；expanded_y 在 1.35 以下展開，沒有省略數值，但距離比例於 1.35 改變，需在圖說說明。No-Gain 參考點跨家族重複顯示；真實座標不作抖動，部分點可能重疊。

歷史版本固定保留於 `retrain_lhs10_23_63_models_validation_nogain42_baseline_previous_before_dg_expansion`；新 DG 原始驗證保留於 `retrain_lhs10_23_expanded_dg_validation`。此前選定的 test 評估仍是歷史結果，並未因本報告重選或重跑。舊 test 原本指向 `retrain_lhs10_23_63_models_validation_nogain42_baseline/` 的歷史基準檔，現應對應到相同檔名的上述備份目錄；`historical_path_mapping_verification.json` 逐項記錄對應路徑並核對原始 SHA，舊 test 檔案沒有修改。

重建到新空資料夾：`.\\.venv\\Scripts\\python.exe -m scripts.analysis.build_expanded_dg_validation_report --output-dir results/visualization/retrain_lhs10_23_63_models_validation_nogain42_expanded_staging`。
''')
    (out/'README.md').write_text(readme,encoding='utf-8')
    report=dict(passed=True,model_count=63,regional_points=126,components=2268,kci_components=504,seeds=list(SEEDS),
                reused_historical_models=51,newly_evaluated_dg_models=12,baseline_seed=42,retained_coordinates_checked=102,
                same_x_y_axes_all_panels=True,no_jitter=True,optimizer_updates=0)
    (out/'analysis_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    outputs={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'}
    provenance=dict(sources_sha256=source_hashes,outputs_sha256=outputs,previous_fixed_baseline_outputs_sha256=old_prov['outputs_sha256'],
                    expanded_validation_source_sha256=new_protocol['sources_sha256'],baseline_seed=42,
                    reused_historical_runs=51,new_dg_runs=12,plot_bounds=dict(x=xlim,y=ylim),
                    DG_max_scale={'rho_0.015':50,'rho_0.15_1.5_15_150':10000000},
                    historical_selected_test='unchanged; not reselected or rerun',
                    historical_test_path_mapping='historical_path_mapping_verification.json',jitter=False)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),models=63,points=126,reused=51,new=12,xlim=xlim,ylim=ylim),indent=2))


if __name__=='__main__': main()
