"""Compare frozen No-Gain, Gain=250 and SS-only=2.5 on the existing test cases."""
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
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiment_paths import experiment_results_dir
from src.models import get_model
from src.variable_selection import variable_selection
from scripts.prediction.validate_hyperparameters import evaluate_case, metric_rows, independent_rmse_rows, aggregate, sha

OUTPUT = ROOT / 'results/visualization/three_model_no_gain_gain250_ss2p5_test_seed42'
INTERNAL = ['R5', 'R5-1', 'R5-2', 'R5-6']
EXTERNAL = [f'R5-{i}' for i in range(19, 27)]
TARGETS = ['B35_H2S', 'B35_SO2']
KEYS = ['partition', 'case', 'target', 'evaluation']
MODELS = {
    'no_gain': ('No-Gain', 'no_gain', 'no_gain', '#0072B2', '--', 'o'),
    'gain250': ('Gain=250', 'gain', 'gain250', '#009E73', '-.', 's'),
    'ss2p5_only': ('SS-only=2.5', 'ss_only', 'ss2p5_only', '#D55E00', '-', '^'),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def draw_figures():
    metrics = pd.read_csv(OUTPUT/'test_metrics.csv')
    horizon = pd.read_csv(OUTPUT/'horizon_metrics.csv')
    overall = pd.read_csv(OUTPUT/'avgrelrmse_overall.csv').set_index('model')
    per_evaluation = pd.read_csv(OUTPUT/'avgrelrmse_by_evaluation.csv')
    summary = pd.read_csv(OUTPUT/'test_summary.csv')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':10,'pdf.fonttype':42})
    names = []

    def save(fig, name):
        fig.canvas.draw()
        for ax in fig.axes:
            for line in ax.lines:
                y=np.asarray(line.get_ydata(),dtype=float)
                assert np.isfinite(y).all()
                assert ((y>=ax.get_ylim()[0]) & (y<=ax.get_ylim()[1])).all(),name
        for extension in ['png','pdf']:
            fig.savefig(OUTPUT/f'{name}.{extension}',dpi=300,bbox_inches='tight')
        plt.close(fig)
        names.append(name)

    def style(ax):
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for alias,(label,_,_,color,ls,marker) in MODELS.items():
        axes[0].plot(range(3),overall.loc[alias,['internal_AvgRelRMSE','external_AvgRelRMSE','AvgRelRMSE']],color=color,ls=ls,marker=marker,label=label)
        rows=per_evaluation[per_evaluation.model==alias].set_index('evaluation')
        axes[1].plot(range(3),rows.loc[['H1','H18','TS'],'AvgRelRMSE'],color=color,ls=ls,marker=marker,label=label)
    for ax in axes:
        ax.axhline(1,color='.5',ls=':',lw=1)
        ax.set_ylabel('AvgRelRMSE (No-Gain = 1)')
        style(ax)
    axes[0].set_xticks(range(3),['Internal','External','Overall'])
    axes[1].set_xticks(range(3),['H1','H18','TS'])
    axes[0].set_title('Region comparison')
    axes[1].set_title('Prediction-scale comparison')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.text(.5,.015,'Existing test cases; same-case RMSE ratios; geometric means; internal/external each 50%; seed42.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.82,bottom=.16,wspace=.3)
    save(fig,'test_avgrelrmse_comparison')

    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,partition in enumerate(['internal','external']):
        for j,target in enumerate(TARGETS):
            ax=axes[i,j]
            for alias,(label,_,_,color,ls,marker) in MODELS.items():
                rows=summary[(summary.partition==partition)&(summary.target==target)&(summary.model==alias)].set_index('evaluation')
                ax.plot(range(3),rows.loc[['H1','H18','TS'],'mean_case_RMSE']*1000,color=color,ls=ls,marker=marker,label=label)
            ax.set(title=f'{partition.capitalize()} | {target.removeprefix("B35_")}',ylabel=r'Mean case RMSE ($10^{-3}$ kmol/kmol)')
            ax.set_xticks(range(3),['H1','H18','TS'])
            style(ax)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.text(.5,.015,'Arithmetic mean of case RMSE within each region; internal n=4 cases, external n=8 cases; seed42.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.89,bottom=.10,hspace=.38,wspace=.32)
    save(fig,'test_rmse_summary')

    fig,axes=plt.subplots(2,2,figsize=(11,8))
    means=horizon.groupby(['partition','model','target','horizon']).RMSE.mean()
    for i,partition in enumerate(['internal','external']):
        for j,target in enumerate(TARGETS):
            ax=axes[i,j]
            for alias,(label,_,_,color,ls,_) in MODELS.items():
                ax.plot(range(1,19),means.loc[(partition,alias,target)].sort_index()*1000,color=color,ls=ls,label=label)
            ax.set(title=f'{partition.capitalize()} | {target.removeprefix("B35_")}',xlabel='Prediction horizon (10 min per step)',ylabel=r'Mean case RMSE ($10^{-3}$ kmol/kmol)')
            ax.set_xticks([1,6,12,18])
            style(ax)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.subplots_adjust(top=.89,bottom=.09,hspace=.45,wspace=.32)
    save(fig,'test_horizon_rmse')

    for name,cases in [('internal',INTERNAL),('external_19_22',EXTERNAL[:4]),('external_23_26',EXTERNAL[4:])]:
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        for j,case in enumerate(cases):
            for i,target in enumerate(TARGETS):
                ax=axes[i,j]
                truth=None
                for alias,(label,_,_,color,ls,_) in MODELS.items():
                    data=pd.read_csv(OUTPUT/'predictions'/alias/case/'time_series.csv')
                    if truth is None:
                        truth=data[f'{target}_true'].to_numpy()
                        ax.plot(data.time_min,truth*1000,color='black',lw=1.8,label='True')
                    else: np.testing.assert_allclose(truth,data[f'{target}_true'],rtol=1e-12,atol=1e-12)
                    ax.plot(data.time_min,data[f'{target}_pred']*1000,color=color,ls=ls,lw=1.3,label=label)
                ax.set_title(f'{case} | {target.removeprefix("B35_")}')
                ax.set_ylabel(r'Concentration ($10^{-3}$ kmol/kmol)')
                if i: ax.set_xlabel('Elapsed time from case start (min)')
                style(ax)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
        fig.text(.5,.015,'TS: only the first 18 points use true history; predictions are then fed back recursively. Known future decoder inputs.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.90,bottom=.11,hspace=.37,wspace=.40)
        save(fig,f'time_series_{name}')
    contact=Image.new('RGB',(1800,1950),'white')
    for i,name in enumerate(names):
        with Image.open(OUTPUT/f'{name}.png') as image:
            thumb=image.convert('RGB')
        thumb.thumbnail((900,650))
        contact.paste(thumb,((i%2)*900+(900-thumb.width)//2,(i//2)*650+(650-thumb.height)//2))
    contact.save(OUTPUT/'contact_sheet.png')
    contact.convert('L').save(OUTPUT/'contact_sheet_grayscale.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        draw_figures()
        return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protected={}
    def protect(path): protected[str(path.relative_to(ROOT))]=sha(path)
    for path in [ROOT/'scripts/prediction/validate_hyperparameters.py',ROOT/'scripts/prediction/predict.py',*(ROOT/'src').rglob('*.py')]: protect(path)
    paths,frames,audits,configurations={},{},[],{}
    for case in INTERNAL+EXTERNAL:
        suffix='_test_split' if case in INTERNAL else ''
        path=ROOT/'data/Claus_dynamic'/f'Test_dataform_change_air2_R={case.replace("R5","5")}{suffix}.csv'
        protect(path)
        raw=pd.read_csv(path)
        frame=raw.rolling(10,min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        assert len(frame)>=36 and np.isfinite(frame.to_numpy()).all() and frame[TARGETS].gt(0).all().all()
        paths[case],frames[case]=path,frame
        audits.append(dict(case=case,partition='internal' if case in INTERNAL else 'external',source=str(path.relative_to(ROOT)),raw_rows=len(raw),processed_rows=len(frame),TS_points=len(frame)-18,horizon_windows=len(frame)-35))
    for alias,(_,category,code,_,_,_) in MODELS.items():
        path=ROOT/f'configs/pgin/{category}/transformer_layerwise_63var_decoder_input_sp_pgin_{code}_seed42.yaml'
        protect(path)
        config=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        assert config['training']['seed']==42
        assert config['training']['gain_loss_weight']==(250 if alias=='gain250' else 0)
        assert config['training']['ss_loss_weight']==(2.5 if alias=='ss2p5_only' else 0)
        assert config['window']['train_window_mins']==180 and config['window']['sampling_interval_min']==10 and config['window']['prediction_length']==18
        for p in [ROOT/'saved_models'/f'{config["exp_name"]}.pth',*(experiment_results_dir(config)/n for n in ['zscore_mean.csv','zscore_std.csv'])]: protect(p)
        configurations[alias]=config
    protocol=dict(internal_test=INTERNAL,external_test=EXTERNAL,seed=42,models=list(MODELS),region_weights=[.5,.5],
                  internal_ratio_count=24,external_ratio_count=48,
                  formula='sqrt(GM_internal(RMSE_model / same-case NoGain RMSE) * GM_external(RMSE_model / same-case NoGain RMSE))',
                  internal_component_log_weight=.5/24,external_component_log_weight=.5/48,
                  preprocessing='10-row median; H2S/SO2 log; saved model-specific training z-score statistics',
                  data_roles='Existing previously used test cases; not a new blind test; no validation cases and no optimizer updates.',
                  KCI='Not recomputed or used in this dynamic test comparison.',protected_sha256=protected)
    protocol_path=OUTPUT/'evaluation_protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text(encoding='utf-8'))==protocol,'Inputs changed; refusing stale cached predictions.'
    else: write_json(protocol_path,protocol)
    protect(Path(__file__).resolve())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    selected,horizons,rechecks=[],[],[]
    fresh_count=0
    for alias,config in configurations.items():
        de,ys,_,en=variable_selection(config['data']['variables_num'])
        config['data'].update(num_en_input=len(en),num_de_input=len(de),num_output=len(ys))
        mean,std=[pd.read_csv(experiment_results_dir(config)/n,index_col=0).squeeze('columns') for n in ['zscore_mean.csv','zscore_std.csv']]
        model=get_model(config).to(device)
        model.load_state_dict(torch.load(ROOT/'saved_models'/f'{config["exp_name"]}.pth',map_location=device,weights_only=True))
        model.eval()
        for case in INTERNAL+EXTERNAL:
            folder=OUTPUT/'predictions'/alias/case
            folder.mkdir(parents=True,exist_ok=True)
            npz=folder/'predictions.npz'
            if npz.exists(): arrays=dict(np.load(npz))
            else:
                arrays=evaluate_case(model,frames[case],config,mean,std,device)
                np.savez_compressed(npz,**arrays)
                fresh_count+=1
            partition='internal' if case in INTERNAL else 'external'
            a,b=metric_rows(arrays,alias,case,partition)
            selected.extend(a);horizons.extend(b)
            rechecks.extend(independent_rmse_rows(dict(np.load(npz)),alias,case,partition))
            pd.DataFrame({'time_min':np.arange(18,len(frames[case]))*10,
                          **{f'{t}_true':arrays['ts_true'][:,j] for j,t in enumerate(TARGETS)},
                          **{f'{t}_pred':arrays['ts_pred'][:,j] for j,t in enumerate(TARGETS)}}).to_csv(folder/'time_series.csv',index=False)
            print(f'{alias}: {case} complete ({device})',flush=True)
        del model
    values,full,checks=pd.DataFrame(selected),pd.DataFrame(horizons),pd.DataFrame(rechecks)
    assert len(values)==216 and len(full)==1296 and len(checks)==1368
    match=checks.merge(values,on=['model']+KEYS,suffixes=('_recomputed','_csv'),validate='one_to_one')
    assert len(match)==216
    np.testing.assert_allclose(match.RMSE_recomputed,match.RMSE_csv,rtol=1e-12)
    hm=checks[checks.evaluation!='TS'].copy();hm['horizon']=hm.evaluation.str[1:].astype(int)
    hm=hm.merge(full,on=['model','partition','case','target','horizon'],suffixes=('_recomputed','_csv'),validate='one_to_one')
    assert len(hm)==1296
    np.testing.assert_allclose(hm.RMSE_recomputed,hm.RMSE_csv,rtol=1e-12)
    base=values[values.model=='no_gain'][KEYS+['RMSE','n']].rename(columns={'RMSE':'NoGain_RMSE','n':'NoGain_n'})
    components=values.merge(base,on=KEYS,validate='many_to_one')
    assert components.n.eq(components.NoGain_n).all() and components.RMSE.gt(0).all() and components.NoGain_RMSE.gt(0).all()
    components['relative_RMSE']=components.RMSE/components.NoGain_RMSE
    components['overall_log_weight']=np.where(components.partition=='internal',.5/24,.5/48)
    regional=aggregate(components,['model','partition'])
    overall=regional.pivot(index='model',columns='partition',values='AvgRelRMSE').rename(columns={'internal':'internal_AvgRelRMSE','external':'external_AvgRelRMSE'}).reset_index()
    overall['AvgRelRMSE']=np.sqrt(overall.internal_AvgRelRMSE*overall.external_AvgRelRMSE)
    overall['label']=overall.model.map(lambda m:MODELS[m][0])
    overall=overall.sort_values('AvgRelRMSE')
    for row in overall.itertuples():
        c=components[components.model==row.model]
        assert len(c)==72
        np.testing.assert_allclose(c.groupby('partition').overall_log_weight.sum(),.5)
        np.testing.assert_allclose(row.AvgRelRMSE,np.exp(np.sum(c.overall_log_weight*np.log(c.relative_RMSE))),rtol=1e-12)
    per_region=aggregate(components,['model','partition','evaluation'])
    per_eval=per_region.pivot(index=['model','evaluation'],columns='partition',values='AvgRelRMSE').reset_index()
    per_eval['AvgRelRMSE']=np.sqrt(per_eval.internal*per_eval.external)
    summary=values.groupby(['partition','model','target','evaluation']).agg(cases=('case','size'),mean_case_RMSE=('RMSE','mean')).reset_index()
    paired=components[components.model!='no_gain'].copy()
    paired['RMSE_change_percent']=(paired.relative_RMSE-1)*100
    counts=paired.groupby(['model','partition']).agg(items=('relative_RMSE','size'),improved=('relative_RMSE',lambda s:int((s<1).sum())),worsened=('relative_RMSE',lambda s:int((s>1).sum()))).reset_index()
    versus=values[values.model=='ss2p5_only'].merge(values[values.model=='gain250'],on=KEYS,suffixes=('_SS','_Gain'),validate='one_to_one')
    versus['SS_to_Gain_RMSE']=versus.RMSE_SS/versus.RMSE_Gain
    versus['RMSE_change_percent']=(versus.SS_to_Gain_RMSE-1)*100
    for frame,name in [(values,'test_metrics'),(full,'horizon_metrics'),(checks,'RMSE_recheck'),(components,'relative_rmse_components'),
                       (paired,'paired_case_comparison'),(overall,'avgrelrmse_overall'),(regional,'avgrelrmse_by_region'),
                       (per_eval,'avgrelrmse_by_evaluation'),(per_region,'avgrelrmse_by_region_evaluation'),(summary,'test_summary'),
                       (counts,'comparison_counts'),(versus,'ss2p5_vs_gain250'),(pd.DataFrame(audits),'data_audit')]:
        frame.to_csv(OUTPUT/f'{name}.csv',index=False)
    draw_figures()
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    lines=['# 三模型 test 比較：No-Gain、Gain=250、SS-only=2.5','',
           '內部：R5、R5-1、R5-2、R5-6 的 test split。外部：R5-19～R5-26 完整 CSV。這些是既有 test 案例；未混入 R5-3／4／5 或 R5-27／28／29 驗證案例。','',
           '三個固定 seed42 checkpoint。只推論，無訓練更新。SS-only 的 Gain=0、SS=2.5、SS target scope=all_48_outputs。','',
           '## AvgRelRMSE','',
           '同案例、同目標、同 H1/H18/TS 的 RMSE 先除以 No-Gain。內部 24 項、外部 48 項各自取幾何平均，再計算 sqrt(I×E)，內外各占 50%。No-Gain=1，越低越好。','',
           '| 模型 | 內部 | 外部 | 整體（50%／50%） |','|---|---:|---:|---:|']
    for row in overall.itertuples(): lines.append(f'| {row.label} | {row.internal_AvgRelRMSE:.9f} | {row.external_AvgRelRMSE:.9f} | {row.AvgRelRMSE:.9f} |')
    lines+=['','## H1／H18／TS：兩區各 50%','','| 模型 | H1 | H18 | TS |','|---|---:|---:|---:|']
    for alias,(label,*_) in MODELS.items():
        rows=per_eval[per_eval.model==alias].set_index('evaluation')
        lines.append(f'| {label} | {rows.loc["H1","AvgRelRMSE"]:.9f} | {rows.loc["H18","AvgRelRMSE"]:.9f} | {rows.loc["TS","AvgRelRMSE"]:.9f} |')
    lines+=['','## 原單位平均案例 RMSE（表內乘以 10⁻³ kmol/kmol）','','各區內的案例 RMSE 等權算術平均，與逐案例比值的幾何平均不同。','',
            '| 區域 | 目標 | 評估 | No-Gain | Gain=250 | SS-only=2.5 |','|---|---|---|---:|---:|---:|']
    grid=summary.pivot(index=['partition','target','evaluation'],columns='model',values='mean_case_RMSE')
    for (partition,target,evaluation),row in grid.iterrows():
        lines.append('| '+' | '.join([partition,target,evaluation]+[f'{row[a]*1000:.6f}' for a in MODELS])+' |')
    lines+=['','## 圖與數據','',
            '- [AvgRelRMSE 圖](test_avgrelrmse_comparison.png)／[PDF](test_avgrelrmse_comparison.pdf)',
            '- [原始 RMSE 摘要](test_rmse_summary.png)／[H1～H18](test_horizon_rmse.png)',
            '- [內部 TS](time_series_internal.png)／[外部 19～22](time_series_external_19_22.png)／[外部 23～26](time_series_external_23_26.png)',
            '- [整體分數](avgrelrmse_overall.csv)／[H1/H18/TS](avgrelrmse_by_evaluation.csv)／[原始平均 RMSE](test_summary.csv)',
            '- [144 項與 No-Gain 配對比較](paired_case_comparison.csv)／[72 項 SS-only 對 Gain 比較](ss2p5_vs_gain250.csv)／[改善項數](comparison_counts.csv)',
            '- [216 項 test 指標](test_metrics.csv)／[1296 項 horizon 指標](horizon_metrics.csv)／[1368 項 RMSE 重算](RMSE_recheck.csv)',
            '- [資料案例](data_audit.csv)／[計算定義](evaluation_protocol.json)／[來源與檢查](evaluation_validation.json)','',
            'TS 僅最初 18 點使用真實歷史，其後回填預測；H1/H18 每個滑動起點重新給真實歷史，再遞迴預測第 1／18 步。皆使用已知未來 decoder inputs。','',
            '相關 H1/H18/TS 項目不是獨立重複實驗，單一 seed 不代表統計顯著性。本次未重算 KCI，亦未使用 test 結果重新選超參數；不宣稱為未曾檢視的新盲測。','',
            '重跑：`.\\.venv\\Scripts\\python.exe scripts/prediction/compare_three_model_test.py`；僅重畫加上 `--plot-only`。','']
    (OUTPUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(OUTPUT/'evaluation_validation.json',dict(passed=True,seed=42,device=device,model_count=3,case_count=12,
               fresh_inference_cases_this_run=fresh_count,optimizer_updates=0,selected_metric_rows=216,horizon_metric_rows=1296,
               independent_RMSE_rechecks=1368,raw_truth_alignment_checked=True,same_case_ratio_matching_checked=True,
               internal_weight=.5,external_weight=.5,weighted_log_reconstruction_checked=True,
               validation_cases_used=False,protected_inputs_unchanged=True,protected_sha256=protected,
               visual_review='pending',output_sha256={p.name:sha(p) for p in OUTPUT.iterdir() if p.suffix in ['.csv','.png','.pdf','.md']}))
    print(overall.to_string(index=False),flush=True)
    print(per_eval.to_string(index=False),flush=True)
    print(counts.to_string(index=False),flush=True)
    print(f'PASS: {OUTPUT}',flush=True)


if __name__=='__main__':
    main()
