"""Add No-Gain 43/44 to the preselected single-run test comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.plotting import plot_retrain_selected_test as main_report

ROOT = Path(__file__).resolve().parents[2]
EXTRA = ROOT / 'results/visualization/retrain_lhs10_23_nogain_seeds43_44_test'
OUT = main_report.SOURCE / 'comparison_report_with_nogain_seeds'
KEYS = ['partition', 'case', 'target', 'evaluation']


def load():
    validation = json.loads((EXTRA/'evaluation_validation.json').read_text(encoding='utf-8'))
    if not (validation.get('passed') and validation.get('model_count') == 2
            and validation.get('protected_sources_unchanged') and validation.get('truth_matches_nogain42')):
        raise ValueError('A completed, checked No-Gain 43/44 evaluation is required.')
    data, summary, selection = main_report.load()
    selected, _ = main_report.main_selection(data)
    extra = pd.read_csv(EXTRA/'regional_coordinates.csv')
    if len(extra) != 4 or set(extra.model) != {'no_gain_seed43','no_gain_seed44'}:
        raise ValueError('Expected two extra No-Gain seeds and two regions.')
    parts = pd.read_csv(EXTRA/'relative_rmse_components.csv')
    metrics = pd.read_csv(main_report.SOURCE/'test_metrics.csv')
    base = metrics[metrics.model.eq('no_gain_seed42')][KEYS+['RMSE']].rename(columns={'RMSE':'actual_baseline'})
    matched = parts.merge(base, on=KEYS, validate='many_to_one')
    if (len(matched) != 144 or not np.allclose(matched.NoGain42_RMSE, matched.actual_baseline, rtol=1e-12)
            or not np.allclose(matched.relative_RMSE, matched.RMSE/matched.actual_baseline, rtol=1e-12)):
        raise ValueError('Extra seed RMSE components do not match the No-Gain 42 benchmark.')
    agg = matched.groupby(['model','partition'],as_index=False).agg(
        computed=('relative_RMSE',lambda s: float(np.exp(np.log(s).mean()))), n=('relative_RMSE','size'))
    checked = extra.merge(agg,on=['model','partition'],validate='one_to_one')
    if len(checked) != 4 or not checked.n.eq(checked.components).all() or not np.allclose(checked.computed,checked.AvgRelRMSE,rtol=1e-12):
        raise ValueError('Extra seed regional RMSE mismatch.')
    kci = pd.read_csv(EXTRA/'kci_components.csv')
    counts = kci.assign(correct=kci.tp+kci.tn).groupby(['model','distribution'],as_index=False).agg(
        correct=('correct','sum'), valid=('valid_comparisons','sum'))
    counts['partition'] = counts.distribution.map({'in_range':'internal','soft_ood':'external'})
    checked = extra.merge(counts,on=['model','partition'],suffixes=('_csv','_check'),validate='one_to_one')
    if (len(kci) != 16 or len(checked) != 4 or not checked.correct_csv.eq(checked.correct_check).all()
            or not checked.valid_csv.eq(checked.valid_check).all()
            or not np.allclose(checked.KCI_percent,100*checked.correct_check/checked.valid_check,rtol=1e-12)):
        raise ValueError('Extra seed KCI component mismatch.')
    extra['family'],extra['alias'],extra['role'] = 'no_gain','no_gain','additional_reference_seed'
    combined = pd.concat([selected,extra],ignore_index=True)
    raw = pd.read_csv(main_report.SOURCE/'relative_rmse_components.csv')
    raw = pd.concat([raw[raw.model.isin(selected.model)],parts],ignore_index=True)
    if len(combined) != 14 or len(raw) != 504:
        raise ValueError('Expected seven models in two regions and 504 raw components.')
    return combined, summary, selection, raw


def plot(data):
    fig, axes = plt.subplots(1,2,figsize=(13,5.2),sharex=True,sharey=True)
    xmin,xmax=data.KCI_percent.min(),data.KCI_percent.max()
    ymin,ymax=min(data.AvgRelRMSE.min(),1),max(data.AvgRelRMSE.max(),1)
    xp,yp=max((xmax-xmin)*.1,.5),max((ymax-ymin)*.12,.05)
    names = {'gain':'Gain 2.5','ss_only':'SS-only 2.5','gain_ss':'Gain+SS 2.5 / 2.5',
             'difficult_gradnorm':'Difficult-GradNorm ρ = 0.015'}
    handles=[]
    for row in data[data.partition.eq('internal')].itertuples(index=False):
        marker='*' if row.model=='no_gain_seed42' else main_report.MARKERS[row.seed]
        label=(f'No-Gain, seed {row.seed}' if row.family=='no_gain' else f'{names[row.family]}, seed {row.seed}')
        handles.append(Line2D([],[],marker=marker,ls='',color=main_report.COLORS[row.family],label=label,markersize=8))
    for ax,region,title in zip(axes,('internal','external'),('Interpolation test','Extrapolation test')):
        for row in data[data.partition.eq(region)].itertuples(index=False):
            marker='*' if row.model=='no_gain_seed42' else main_report.MARKERS[row.seed]
            ax.scatter(row.KCI_percent,row.AvgRelRMSE,marker=marker,s=130 if marker=='*' else 75,
                       c=main_report.COLORS[row.family],edgecolors='white',linewidths=.8,zorder=4)
        ax.axhline(1,color='#777',ls='--',lw=.8)
        ax.set(title=title,xlabel='KCI(%)',xlim=(xmin-xp,xmax+xp),ylim=(max(0,ymin-yp),ymax+yp))
        ax.grid(color='#d0d0d0',alpha=.55,linewidth=.6)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('AvgRelRMSE')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False,fontsize=8)
    fig.suptitle('Validation-selected runs and No-Gain seeds | Existing test',fontsize=14,fontweight='bold',y=1.01)
    fig.tight_layout(rect=(0,0,1,.78))
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'seven_model_test_comparison.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    sources=[main_report.SOURCE/name for name in ['evaluation_validation.json','selected_models.json',
             'regional_coordinates.csv','test_metrics.csv','relative_rmse_components.csv','kci_components.csv']]
    sources += [EXTRA/name for name in ['evaluation_validation.json','regional_coordinates.csv',
                'relative_rmse_components.csv','kci_components.csv']]
    sources += [main_report.VALIDATION]
    hashes={str(p.relative_to(ROOT)):main_report.sha(p) for p in sources}
    data,summary,selection,raw=load()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing report: {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    data.to_csv(OUT/'seven_model_coordinates.csv',index=False,float_format='%.17g')
    raw.to_csv(OUT/'seven_model_raw_rmse.csv',index=False,float_format='%.17g')
    plot(data)
    page=main_report.primary_html(data,summary,selection,raw)
    old='灰色星號是共同基準 No-Gain 42。四個彩色點各為該家族依 validation 選出的最低單次。'
    new='灰色星號為共同基準 No-Gain 42；灰色方形與三角形分別為 No-Gain 43、44，數值同樣相對 seed 42。四個彩色點各為該家族依 validation 選出的最低單次。'
    if old not in page:
        raise ValueError('Unexpected primary template.')
    page=page.replace(old,new).replace('primary_raw_rmse.csv', 'seven_model_raw_rmse.csv')
    page=page.replace('</main><script id="dataset"',
        '<p><a href="../comparison_report/primary_test_comparison.html">原五模型比較</a> · '
        '<a href="../comparison_report/test_comparison.html">四家族三 seed 補充比較</a></p></main><script id="dataset"',1)
    (OUT/'seven_model_test_comparison.html').write_text(page,encoding='utf-8')
    (OUT/'README.md').write_text('七模型比較：四家族 validation 單次最低預選，加上 No-Gain 42、43、44。'
        '所有逐項 RMSE 均以 No-Gain 42 作分母；43、44 並非各自固定為 1。'
        '動態 test 與穩態 ANN-reference KCI 使用不同資料；既有 test 並非未曾檢視的新盲測。\n',encoding='utf-8')
    if any(main_report.sha(ROOT/path)!=digest for path,digest in hashes.items()):
        raise ValueError('An input changed during reporting.')
    result={'passed':True,'source_sha256':hashes,'model_count':7,'regional_points':14,'raw_rmse_components':504,
            'output_sha256':{p.name:main_report.sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT/'report_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Created: {OUT}')


if __name__=='__main__':
    main()
