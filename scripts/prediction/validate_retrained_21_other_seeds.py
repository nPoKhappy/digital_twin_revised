"""Recompute seed-43 or seed-44 regional validation for the 21 LHS-retrained models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiment_paths import experiment_results_dir
from src.models import get_model
from scripts.evaluation import evaluate_gain_in_out_distribution as native
from scripts.prediction.validate_hyperparameters import (
    INTERNAL, EXTERNAL, TARGETS, KEYS, evaluate_case, metric_rows,
    independent_rmse_rows, sha, write_json,
)
from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts

SEED = None
OUT = None
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
REGIONS = {'internal': 'in_range', 'external': 'soft_ood'}
CASE_PATHS = {c: ROOT / f'data/Claus_dynamic/Test_dataform_change_air2_R=5-{c.split("-")[1]}.{ "xlsx" if c in INTERNAL else "csv"}' for c in INTERNAL + EXTERNAL}
PROBES = {name: ROOT / f'data/Claus_steady_state/lhs_valid_{name}.xlsx' for name in REGIONS.values()}


def preflight():
    jobs = [j for j in json.loads(MANIFEST.read_text(encoding='utf-8')) if j['seed'] == SEED]
    batch_path = ROOT / f'results/pgin/retrain_lhs10_23/seed{SEED}/_batch_run.json'
    batch = json.loads(batch_path.read_text(encoding='utf-8-sig'))
    assert batch['status'] == 'complete' and len(jobs) == len(batch['completed']) == 21
    completed = {j['exp_name']: j for j in batch['completed']}
    assert len(completed) == 21 and len({j['alias'] for j in jobs}) == 21
    configs = {}
    sources = [Path(__file__), MANIFEST, batch_path,
               ROOT/'saved_models/Tabular_MLP_5in_2out_QV.pth',
               ROOT/'results/steady_state/Tabular_MLP_New/zscore_mean.csv',
               ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv',
               ROOT/'scripts/evaluation/evaluate_gain_in_out_distribution.py',
               ROOT/'scripts/prediction/validate_hyperparameters.py',
               ROOT/'scripts/prediction/validate_gain_ss_hyperparameters.py',
               ROOT/'scripts/prediction/predict.py', *list((ROOT/'src').rglob('*.py')),
               *CASE_PATHS.values(), *PROBES.values()]
    for p in sources[3:6]:
        assert sha(p).lower() == batch['teacher_and_source_sha256'][str(p)].lower(), f'Teacher artifact changed: {p}'
    for j in jobs:
        cfgpath = ROOT / j['config']
        cfg = yaml.safe_load(cfgpath.read_text(encoding='utf-8-sig'))
        assert sha(cfgpath) == j['config_sha256'] and cfg['training']['seed'] == SEED
        assert cfg['exp_name'] == j['exp_name']
        native.model_contract(cfg)
        completed_job = completed[cfg['exp_name']]
        checkpoint = ROOT / j['checkpoint']
        result = experiment_results_dir(cfg)
        assert result == ROOT / j['results']
        for path, key in [(checkpoint, 'checkpoint_sha256'), (result/'zscore_mean.csv', 'zscore_mean_sha256'), (result/'zscore_std.csv', 'zscore_std_sha256')]:
            assert sha(path).lower() == completed_job[key].lower(), f'Training artifact changed: {path}'
            sources.append(path)
        sources.append(cfgpath)
        configs[j['alias']] = cfg
    contract = native.model_contract(configs['no_gain'])
    for alias, cfg in configs.items():
        native.assert_same_contract(contract, native.model_contract(cfg), alias)
    native.assert_same_scaling([(j['config'], configs[j['alias']]) for j in jobs], contract[0]+contract[1]+contract[2])
    protocol = dict(seed=SEED, jobs=jobs, sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    internal_cases=INTERNAL, external_cases=EXTERNAL,
                    kci_sources={k: str(v.relative_to(ROOT)) for k,v in PROBES.items()},
                    rmse=f'Each region geometric mean over 3 cases x 2 outputs x H1/H18/TS of same-case model RMSE divided by seed{SEED} No-Gain RMSE.',
                    kci='Micro sign agreement pooled over 18 post-step times, both step directions and 2 MV x 2 target pairs; teacher magnitude threshold 1e-5; zero dynamic gain incorrect.',
                    note='RMSE and KCI use separate dynamic and steady-state validation datasets. R5-5 crosses the core boundary. Repeated seed uncertainty not evaluated.', optimizer_updates=0)
    OUT.mkdir(parents=True, exist_ok=True)
    proto = OUT/'run_protocol.json'
    if proto.exists():
        assert json.loads(proto.read_text(encoding='utf-8')) == protocol, 'Inputs changed; use a new output directory.'
    else:
        write_json(proto, protocol)
    return jobs, configs, contract, protocol


def main():
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jobs, configs, contract, protocol = preflight()
    print(f'Preflight: {len(jobs)} models; device={device}', flush=True)
    frames = {}
    for case, path in CASE_PATHS.items():
        raw = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
        frame = raw.rolling(10, min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        assert len(frame) >= 36 and np.isfinite(frame.to_numpy()).all() and frame[TARGETS].gt(0).all().all()
        frames[case] = frame
    data = {}
    for name, path in PROBES.items():
        f = native.load_steady_state_source(str(path), contract[-1], log_target_cols=native.FULL_TARGET_COLS)
        data[name] = native.validate_distribution_file(f, name).reset_index(drop=True)
    assert len(data['in_range']) == 4404 and len(data['soft_ood']) == 2585
    teacher = native.load_teacher(device)
    de, ys, en, targets, mvs, _ = contract
    # The physical teacher gradients are independent of dynamic model/scaler.
    teacher_gain = {}
    for name, frame in data.items():
        teacher_gain[name] = np.concatenate([native.teacher_gain_matrix(frame.iloc[i:i+128], *teacher, targets, mvs, device).cpu().numpy() for i in range(0,len(frame),128)])
        assert np.isfinite(teacher_gain[name]).all()
    for index, job in enumerate(jobs, 1):
        alias = job['alias']; cfg = configs[alias]
        dest = OUT/'model_data'/alias
        dest.mkdir(parents=True, exist_ok=True)
        sentinel = dest/'complete.json'
        if sentinel.exists():
            cached = json.loads(sentinel.read_text(encoding='utf-8'))
            assert cached['run_protocol_sha256'] == sha(OUT/'run_protocol.json')
            for n,h in cached['outputs'].items(): assert sha(dest/n) == h
            print(f'[{index}/21] {alias}: verified cache',flush=True)
            continue
        print(f'[{index}/21] {alias}: dynamic validation',flush=True)
        mean,std=native.load_model_stats(cfg['exp_name'],cfg)
        model=get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/job['checkpoint'],map_location=device,weights_only=True)); model.eval()
        rows=[]; checks=[]
        for case,frame in frames.items():
            arrays=evaluate_case(model,frame,cfg,mean,std,device)
            np.savez_compressed(dest/f'{case}_prediction.npz',**arrays)
            partition='internal' if case in INTERNAL else 'external'
            selected,_=metric_rows(arrays,alias,case,partition); rows.extend(selected)
            checks.extend(independent_rmse_rows(arrays,alias,case,partition))
        metrics=pd.DataFrame(rows);check=pd.DataFrame(checks)
        check=check.merge(metrics,on=['model']+KEYS,suffixes=('_direct','_reported'),validate='one_to_one')
        np.testing.assert_allclose(check.RMSE_direct,check.RMSE_reported,rtol=1e-12)
        metrics.to_csv(dest/'validation_metrics.csv',index=False)
        details=[]
        for name, frame in data.items():
            counts=native.empty_counts(targets,mvs); values=[]
            for i in range(0,len(frame),128):
                batch=frame.iloc[i:i+128].reset_index(drop=True)
                reference=torch.from_numpy(teacher_gain[name][i:i+len(batch)]).to(device)
                gain,first=native.dynamic_gain_matrix(model,batch,mean,std,de,ys,en,targets,mvs,cfg,device)
                assert first==1 and gain.shape[:2]==(2,18) and torch.isfinite(gain).all()
                for step in range(18):native.update_counts(counts,reference,gain[:,step],targets,mvs,1e-5)
                values.append(gain.cpu().numpy())
                if i==0 or i+128>=len(frame) or i//128%10==0: print(f'[{index}/21] {alias}: {name} {min(i+128,len(frame))}/{len(frame)}',flush=True)
            arr=np.concatenate(values,axis=2)
            np.savez_compressed(dest/f'gain_{name}.npz',teacher_gain=teacher_gain[name],dynamic_gain=arr)
            independent=numpy_counts(teacher_gain[name],arr,1e-5,targets,mvs)
            for target in targets:
                for mv in mvs:
                    result=native.count_row(cfg['exp_name'],name,target,mv,counts[(target,mv)],len(frame))
                    result.update(model=alias,family=job['family'])
                    independent_row=independent.query('target == @target and mv == @mv').iloc[0]
                    for col in ['possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain']:
                        assert result[col] == independent_row[col],(alias,name,target,mv,col)
                    details.append(result)
        pd.DataFrame(details).to_csv(dest/'kci_components.csv',index=False)
        names=[p.name for p in dest.iterdir() if p.is_file() and p.name != 'complete.json']
        write_json(sentinel,dict(run_protocol_sha256=sha(OUT/'run_protocol.json'),outputs={n:sha(dest/n) for n in names}))
        del model
    metrics=pd.concat([pd.read_csv(OUT/'model_data'/j['alias']/'validation_metrics.csv') for j in jobs],ignore_index=True)
    kci=pd.concat([pd.read_csv(OUT/'model_data'/j['alias']/'kci_components.csv') for j in jobs],ignore_index=True)
    baseline=metrics[metrics.model.eq('no_gain')][KEYS+['RMSE','n']].rename(columns={'RMSE':'baseline_RMSE','n':'baseline_n'})
    components=metrics.merge(baseline,on=KEYS,validate='many_to_one')
    assert len(components)==21*36 and components.n.eq(components.baseline_n).all() and components.baseline_RMSE.gt(0).all()
    components['relative_RMSE']=components.RMSE/components.baseline_RMSE
    assert (components.relative_RMSE>0).all()
    regional=components.groupby(['model','partition'],as_index=False).agg(AvgRelRMSE=('relative_RMSE',lambda x:float(np.exp(np.log(x).mean()))),components=('relative_RMSE','size'))
    assert regional.components.eq(18).all()
    kci['correct']=kci.tp+kci.tn
    counts=kci.groupby(['model','distribution'],as_index=False).agg(correct=('correct','sum'),valid=('valid_comparisons','sum'))
    counts['ANN_reference_KCI_percent']=100*counts.correct/counts.valid
    counts['partition']=counts.distribution.map({v:k for k,v in REGIONS.items()})
    coordinates=regional.merge(counts,on=['model','partition'],validate='one_to_one')
    coordinates['family']=coordinates.model.map({j['alias']:j['family'] for j in jobs})
    for frame,name in [(metrics,'validation_metrics.csv'),(components,'relative_rmse_components.csv'),(kci,'kci_components.csv'),(coordinates,'regional_plot_coordinates.csv')]:frame.to_csv(OUT/name,index=False)
    assert len(coordinates)==42 and np.isfinite(coordinates[['AvgRelRMSE','ANN_reference_KCI_percent']]).all().all()
    colors={'no_gain':'#202020','gain':'#0072B2','ss_only':'#D55E00','gain_ss':'#009E73','difficult_gradnorm':'#8C4BB5'}
    markers={'no_gain':'*','gain':'o','ss_only':'s','gain_ss':'^','difficult_gradnorm':'D'}
    labels={'no_gain':'No-Gain','gain':'Gain','ss_only':'SS-only','gain_ss':'Gain+SS','difficult_gradnorm':'Difficult-GradNorm'}
    fig,axes=plt.subplots(1,2,figsize=(16,8),sharey=True)
    for ax,part,title in zip(axes,['internal','external'],['Interpolation validation','Extrapolation validation']):
        sub=coordinates[coordinates.partition.eq(part)]
        for fam in colors:
            s=sub[sub.family.eq(fam)]
            ax.scatter(s.ANN_reference_KCI_percent,s.AvgRelRMSE,c=colors[fam],marker=markers[fam],s=90 if fam=='no_gain' else 55,label=labels[fam],edgecolors='white',linewidths=.5,zorder=3)
            for row in s.itertuples():
                ax.annotate(row.model,(row.ANN_reference_KCI_percent,row.AvgRelRMSE),xytext=(4,4),textcoords='offset points',fontsize=6.5,color=colors[fam])
        ax.axhline(1,color='.5',ls='--',lw=1)
        ax.set(title=title,xlabel='ANN-reference KCI (%)')
        ax.grid(alpha=.2); ax.margins(x=.13,y=.12)
    axes[0].set_ylabel(f'AvgRelRMSE (seed {SEED} No-Gain = 1)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,.96))
    fig.suptitle(f'21 retrained models | seed {SEED} validation',fontsize=15,fontweight='bold',y=1.01)
    fig.text(.5,.015,'Y: R5-3/4/5 (left), R5-27/28/29 (right); geometric mean of 18 paired RMSE ratios.\nX: separate steady-state validation probes: in-range (left), soft-OOD (right); ANN-reference micro KCI.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.91))
    for ext in ['png','pdf']:fig.savefig(OUT/f'regional_kci_vs_avgrelrmse.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    for path,digest in protocol['sources_sha256'].items():assert sha(ROOT/path)==digest, f'Input changed during run: {path}'
    write_json(OUT/'evaluation_validation.json',dict(passed=True,seed=SEED,model_count=21,per_region_points=21,source_hashes_unchanged=True,independent_rmse_and_numpy_kci_recounts=True,optimizer_updates=0))
    print(f'COMPLETE: {OUT}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(43,44),required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    SEED=args.seed
    OUT=ROOT / f'results/visualization/retrain_lhs10_23_21_models_validation_seed{SEED}'
    if args.preflight_only:
        jobs, configs, contract, protocol=preflight()
        print(f'Preflight passed: seed {SEED}, {len(jobs)} models; output={OUT}',flush=True)
    else:
        main()
