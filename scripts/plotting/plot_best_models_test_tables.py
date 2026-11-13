"""Five reference-style test tables and a regional KCI/AvgRelRMSE figure."""
from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

FAMILIES=['no_gain','gain','ss_only','gain_ss']
LABELS=['No-Gain','Gain=250','SS-only=2.5','Gain+SS=2.5/2.5']
INTERNAL=['R5','R5-1','R5-2','R5-6']
EXTERNAL=[f'R5-{i}' for i in range(19,27)]
PAIRS=[('B35_H2S','air2_SP','Air2 → H₂S'),('B35_H2S','HEATER2_output_T_SP','T2 → H₂S'),
       ('B35_SO2','air2_SP','Air2 → SO₂'),('B35_SO2','HEATER2_output_T_SP','T2 → SO₂')]


def delta_percent(value,baseline):
    if baseline<=0:raise ValueError('No-Gain RMSE must be positive.')
    return 100*(value/baseline-1)


def signed(value):
    # Avoid showing -0.0 or +0.0 for a value rounded to zero.
    return '0.0' if abs(value)<.05 else f'{value:+.1f}'


def grouped_horizon_rows(metrics,partition,target,cases):
    """One case block contains H1/H18/TS; all formatting uses unrounded RMSE."""
    indexed=metrics.set_index(['partition','case','target','evaluation','model'])
    records=[]
    for case in cases:
        for evaluation in ['H1','H18','TS']:
            values={family:float(indexed.loc[(partition,case,target,evaluation,family),'RMSE']) for family in FAMILIES}
            baseline=values['no_gain']
            row=dict(partition=partition,case=case,target=target,evaluation=evaluation,
                     NoGain_RMSE=baseline,NoGain_RMSE_x1e3=1000*baseline)
            for family in FAMILIES[1:]:
                row[f'{family}_RMSE']=values[family]
                row[f'{family}_delta_percent']=delta_percent(values[family],baseline)
            row['best_models']='|'.join(f for f in FAMILIES if np.isclose(values[f],min(values.values()),rtol=1e-12,atol=0))
            records.append(row)
    return pd.DataFrame(records)


def write_grouped_html(folder,sections):
    parts=['<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Test RMSE 表格</title>',
           '<style>body{font-family:"Times New Roman","Microsoft JhengHei",serif;margin:32px}table{border-collapse:collapse;width:100%;margin:20px 0 45px;border-top:1px solid;border-bottom:1px solid}th{border-bottom:1px solid}th,td{text-align:center;padding:4px 12px}td[rowspan]{vertical-align:middle}.up{color:red}.down{color:green}.best{font-weight:bold}</style>',
           '<body><p>Δ = 100 × (模型 RMSE / No-Gain RMSE − 1)；綠色：改善；紅色：惡化；粗體：每列最低RMSE。</p>']
    for title,frame in sections:
        parts.append(f'<h2>{escape(title)}</h2><table><thead><tr><th>案例</th><th>時域</th><th>No-Gain RMSE<br>(10<sup>−3</sup> kmol/kmol)</th>')
        parts.extend(f'<th>{escape(label)}<br>Δ (%)</th>' for label in LABELS[1:])
        parts.append('</tr></thead><tbody>')
        for i,row in enumerate(frame.to_dict('records')):
            winners=row['best_models'].split('|');parts.append('<tr>')
            if i%3==0:parts.append(f'<td rowspan="3">{escape(row["case"])}</td>')
            parts.append(f'<td>{row["evaluation"]}</td><td class="{"best" if "no_gain" in winners else ""}">{row["NoGain_RMSE_x1e3"]:.4f}</td>')
            for family in FAMILIES[1:]:
                value=row[f'{family}_delta_percent'];color='down' if value<0 else 'up' if value>0 else ''
                parts.append(f'<td class="{color} {"best" if family in winners else ""}">{signed(value)}</td>')
            parts.append('</tr>')
        parts.append('</tbody></table>')
    parts.append('</body></html>')
    (folder/'grouped_rmse_tables.html').write_text('\n'.join(parts),encoding='utf-8')


def build_report(folder,only_figures=None):
    folder=Path(folder)
    metrics=pd.read_csv(folder/'test_metrics.csv')
    details=pd.read_csv(folder/'gain_pair_kci.csv')
    overall=pd.read_csv(folder/'overall_kci.csv')
    paired=pd.read_csv(folder/'paired_case_comparison.csv')
    selection=pd.read_csv(folder/'selected_models.csv')
    assert list(selection.family)==FAMILIES and len(metrics)==288 and len(details)==32 and len(overall)==8
    assert set(metrics.case)==set(INTERNAL+EXTERNAL)
    np.testing.assert_allclose(paired.RMSE_change_percent,100*(paired.RMSE/paired.NoGain_RMSE-1),rtol=1e-12,atol=1e-12)
    # Reconstruct every displayed overall KCI from counts, not arithmetic pair means.
    for row in overall.itertuples():
        d=details[(details.model==row.model)&(details.distribution==row.distribution)]
        assert len(d)==4
        np.testing.assert_allclose(row.micro_KCI_percent,100*(d.tp+d.tn).sum()/d.valid_comparisons.sum(),rtol=1e-12)
    overall_grid=overall.pivot(index='model',columns='distribution',values='micro_KCI_percent').loc[FAMILIES].reset_index()
    overall_grid.to_csv(folder/'overall_kci_table.csv',index=False)
    pair_rows=[]
    for distribution in ['in_range','soft_ood']:
        for target,mv,label in PAIRS:
            d=details[(details.distribution==distribution)&(details.target==target)&(details.mv==mv)].set_index('model')
            row=dict(distribution=distribution,target=target,mv=mv,pair=label)
            for family in FAMILIES:
                value=float(d.loc[family,'kci_percent'])
                np.testing.assert_allclose(value,100*(d.loc[family,'tp']+d.loc[family,'tn'])/d.loc[family,'valid_comparisons'],rtol=1e-12)
                row[family]=value
            pair_rows.append(row)
    pair_grid=pd.DataFrame(pair_rows)
    pair_grid.to_csv(folder/'pair_kci_table.csv',index=False)
    by_key=metrics.set_index(['partition','case','target','evaluation','model'])
    def rmse(partition,case,target,evaluation,family):return float(by_key.loc[(partition,case,target,evaluation,family),'RMSE'])
    core=[]
    for target in ['B35_H2S','B35_SO2']:
        for case in INTERNAL:
            baseline=rmse('internal',case,target,'TS','no_gain')
            row=dict(case=case,target=target,NoGain_TS_RMSE=baseline,NoGain_TS_RMSE_x1e3=baseline*1000)
            for family in FAMILIES[1:]:
                value=rmse('internal',case,target,'TS',family)
                row[f'{family}_TS_RMSE']=value;row[f'{family}_delta_percent']=delta_percent(value,baseline)
            core.append(row)
    core=pd.DataFrame(core);core.to_csv(folder/'internal_ts_rmse_table.csv',index=False)
    external={}
    for target in ['B35_SO2','B35_H2S']:
        records=[]
        for case in EXTERNAL:
            row=dict(case=case,target=target)
            for evaluation in ['H18','TS']:
                baseline=rmse('external',case,target,evaluation,'no_gain')
                row[f'NoGain_{evaluation}_RMSE']=baseline;row[f'NoGain_{evaluation}_RMSE_x1e3']=baseline*1000
                for family in FAMILIES[1:]:
                    value=rmse('external',case,target,evaluation,family)
                    row[f'{family}_{evaluation}_RMSE']=value;row[f'{family}_{evaluation}_delta_percent']=delta_percent(value,baseline)
            records.append(row)
        external[target]=pd.DataFrame(records)
        external[target].to_csv(folder/f'external_{target.removeprefix("B35_").lower()}_rmse_table.csv',index=False)
    core_grouped={t:grouped_horizon_rows(metrics,'internal',t,INTERNAL) for t in ['B35_H2S','B35_SO2']}
    pd.concat(core_grouped.values(),ignore_index=True).to_csv(folder/'internal_horizon_rmse_table.csv',index=False)
    outer_grouped={t:grouped_horizon_rows(metrics,'external',t,EXTERNAL) for t in ['B35_SO2','B35_H2S']}
    for target,frame in outer_grouped.items():frame.to_csv(folder/f'external_{target.removeprefix("B35_").lower()}_horizon_rmse_table.csv',index=False)
    write_grouped_html(folder,[('核心 H₂S',core_grouped['B35_H2S']),('核心 SO₂',core_grouped['B35_SO2']),
                               ('近外插 SO₂',outer_grouped['B35_SO2']),('近外插 H₂S',outer_grouped['B35_H2S'])])
    # Different numbers of test cases: each region is summarized separately.
    region=paired.groupby(['model','partition'],sort=False).agg(
        AvgRelRMSE=('relative_RMSE',lambda s:float(np.exp(np.log(s).mean()))),component_count=('relative_RMSE','size')).reset_index()
    for row in region.itertuples():assert row.component_count==(24 if row.partition=='internal' else 48)
    region=region.merge(overall.assign(partition=overall.distribution.map({'in_range':'internal','soft_ood':'external'}))[
        ['model','partition','micro_KCI_percent']],on=['model','partition'],validate='one_to_one')
    region.to_csv(folder/'avgrelrmse_by_region.csv',index=False)
    region.rename(columns={'micro_KCI_percent':'ANN_reference_KCI_percent'}).to_csv(folder/'regional_plot_coordinates.csv',index=False)
    total=region.pivot(index='model',columns='partition',values='AvgRelRMSE').reset_index()
    total['AvgRelRMSE']=np.sqrt(total.internal*total.external)
    total.to_csv(folder/'avgrelrmse_overall.csv',index=False)
    plt.rcParams.update({'font.family':['DejaVu Sans','Microsoft JhengHei'],
                         'font.size':13,'pdf.fonttype':42,'axes.unicode_minus':False})
    figures=[]
    def save(fig,name):
        if only_figures is not None and name not in only_figures:
            plt.close(fig);figures.append(name)
            return
        fig.canvas.draw()
        for ax in fig.axes:
            if ax.get_gid()!='grouped_horizon_table':continue
            for table in ax.tables:
                for (r,c),cell in table.get_celld().items():
                    box=cell.get_window_extent(fig.canvas.get_renderer());text=cell.get_text().get_window_extent(fig.canvas.get_renderer())
                    assert text.width<=box.width+2,(name,r,c,'text wider than grouped cell')
                    assert text.height<=box.height+2,(name,r,c,'text taller than grouped cell')
        fig.savefig(folder/f'{name}.png',dpi=300,facecolor='white')
        fig.savefig(folder/f'{name}.pdf',facecolor='white')
        plt.close(fig)
        figures.append(name)
    def table_figure(name,title,headers,body,widths,footer,bold=None,height=7.5):
        fig=plt.figure(figsize=(15,height),facecolor='white')
        fig.text(.035,.95,title,fontsize=27,va='top')
        ax=fig.add_axes([.035,.23,.93,.57]);ax.axis('off')
        table=ax.table(cellText=body,colLabels=headers,cellLoc='center',colLoc='center',colWidths=widths,bbox=[0,0,1,1])
        table.auto_set_font_size(False);table.set_fontsize(16)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('white');cell.set_facecolor('white');cell.PAD=.035
            if r==0:cell.set_text_props(weight='bold',fontsize=15)
            if c==0:cell.set_text_props(ha='left')
        header_height=1/(len(body)+1)
        for y in [1,1-header_height,0]:ax.plot([0,1],[y,y],color='.2',lw=1,transform=ax.transAxes,clip_on=False)
        if bold:
            for r,c in bold:table[r+1,c].set_text_props(weight='bold')
        fig.text(.035,.055,footer,fontsize=11.2,color='#B22222',va='bottom',linespacing=1.55)
        # Verify cell text extents after drawing, including multi-line headers.
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for (r,c),cell in table.get_celld().items():
            text=cell.get_text().get_window_extent(renderer);box=cell.get_window_extent(renderer)
            assert text.width<=box.width+2,(name,r,c,'text wider than cell')
            assert text.height<=box.height+2,(name,r,c,'text taller than cell')
        save(fig,name)
    body=[]
    for family,label in zip(FAMILIES,LABELS):
        r=overall_grid[overall_grid.model==family].iloc[0]
        body.append([label,f'{r.in_range:.2f}%',f'{r.soft_ood:.2f}%'])
    table_figure('01_overall_kci','整體 KCI 比較（Test）',['模型','核心操作區域 KCI','近外插評估子集 KCI'],body,[.4,.3,.3],
                 '依 validation AvgRelRMSE 選定三個模型；seed42。\nANN-reference micro KCI；test 核心 4,385 點、近外插 2,692 點。',height=5.6)
    body=[];bold=[]
    for index,r in enumerate(pair_grid.to_dict('records')):
        values=[r[f] for f in FAMILIES]
        body.append(['核心' if r['distribution']=='in_range' else '近外插',r['pair'],*[f'{v:.2f}%' for v in values]])
        bold.extend((index,j+2) for j,v in enumerate(values) if np.isclose(v,max(values),rtol=0,atol=1e-10))
    table_figure('02_pair_kci','各增益配對的 KCI（Test）',['區域','增益配對',*LABELS],body,[.075,.185,.14,.17,.19,.24],
                 '粗體為各列最高 KCI；每個配對以 correct / valid 計算。\n使用相同 ANN teacher、scaler、增益有效門檻與 all18 定義。',bold=bold,height=7.5)
    def grouped_table(ax,frame):
        ax.axis('off')
        headers=['案例','時域','No-Gain RMSE\n(10⁻³ kmol/kmol)',*[label+'\nΔ (%)' for label in LABELS[1:]]]
        body=[]
        for i,row in enumerate(frame.to_dict('records')):
            body.append([row['case'] if i%3==1 else '',row['evaluation'],f"{row['NoGain_RMSE_x1e3']:.4f}",
                         *[signed(row[f'{f}_delta_percent']) for f in FAMILIES[1:]]])
        table=ax.table(cellText=body,colLabels=headers,cellLoc='center',colLoc='center',colWidths=[.09,.09,.25,.18,.18,.21],bbox=[0,0,1,1])
        table.auto_set_font_size(False);table.set_fontsize(15)
        for (r,c),cell in table.get_celld().items():
            cell.set_facecolor('white');cell.set_edgecolor('white');cell.PAD=.03
            if r==0:
                cell.set_text_props(weight='bold',fontsize=14)
                cell.set_height(cell.get_height()*1.5)
        for i,row in enumerate(frame.to_dict('records')):
            winners=row['best_models'].split('|')
            if 'no_gain' in winners:table[i+1,2].set_text_props(weight='bold')
            for c,family in enumerate(FAMILIES[1:],3):
                value=row[f'{family}_delta_percent']
                color='#008000' if value<0 else '#FF0000' if value>0 else 'black'
                table[i+1,c].set_text_props(color=color,weight='bold' if family in winners else 'normal')
        header_height=1.5/(len(frame)+1.5)
        for y in [1,1-header_height,0]:ax.plot([0,1],[y,y],color='.2',lw=1,transform=ax.transAxes,clip_on=False)
        # Case text is placed in the middle row; no inner lines split its three-row block.
        ax.set_gid('grouped_horizon_table')
        return table
    fig=plt.figure(figsize=(15,14.3),facecolor='white')
    fig.text(.035,.972,'核心區域動態預測：H1／H18／TS（Test）',fontsize=26,va='top')
    for target,label,bottom in [('B35_H2S','H₂S',.535),('B35_SO2','SO₂',.105)]:
        fig.text(.035,bottom+.365,label,fontsize=19,weight='bold')
        grouped_table(fig.add_axes([.035,bottom,.93,.345]),core_grouped[target])
    fig.text(.035,.028,'Δ = 100 × (模型 RMSE / No-Gain RMSE − 1)。綠色：改善；紅色：惡化；粗體：同一案例／時域的最低 RMSE。\n每個案例依序列出 H1、H18、TS；Gain=250，SS-only=2.5，Gain+SS=(2.5,2.5)，seed42。',fontsize=11,color='.2',linespacing=1.6)
    save(fig,'03_internal_ts_rmse')
    for target,name in [('B35_SO2','04_external_so2_rmse'),('B35_H2S','05_external_h2s_rmse')]:
        fig=plt.figure(figsize=(15,13.5),facecolor='white')
        species='SO₂' if target=='B35_SO2' else 'H₂S'
        fig.text(.035,.968,f'近外插 {species}：H1／H18／TS（Test）',fontsize=26,va='top')
        grouped_table(fig.add_axes([.035,.13,.93,.745]),outer_grouped[target])
        fig.text(.035,.035,'Δ = 100 × (模型 RMSE / No-Gain RMSE − 1)。綠色：改善；紅色：惡化；粗體：同一案例／時域的最低 RMSE。\n每個案例依序列出 H1、H18、TS；Gain=250，SS-only=2.5，Gain+SS=(2.5,2.5)，seed42。',fontsize=11,color='.2',linespacing=1.6)
        save(fig,name)
    # Sixth figure: actual No-Gain KCI points, as well as the y=1 reference.
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.8))
    styles=[('.3','X'),('#0072B2','o'),('#D55E00','^'),('#009E73','s')]
    for ax,partition,title in zip(axes,['internal','external'],['Internal test','External test']):
        d=region[region.partition==partition].set_index('model')
        for family,label,(color,marker) in zip(FAMILIES,LABELS,styles):
            r=d.loc[family]
            ax.plot(r.micro_KCI_percent,r.AvgRelRMSE,color=color,marker=marker,ms=9,ls='none',label=label)
        ax.axhline(1,color='.35',ls='--',lw=1)
        low,high=min(d.AvgRelRMSE.min(),1),max(d.AvgRelRMSE.max(),1)
        pad=max((high-low)*.18,.05)
        ax.set(ylim=(max(0,low-pad),high+pad),xlabel='ANN-reference KCI (%)',ylabel='AvgRelRMSE (No-Gain = 1)',title=title)
        ax.margins(x=.16);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.ticklabel_format(axis='x',useOffset=False,style='plain')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,fontsize=10)
    fig.text(.5,.025,'Test cases: 4 internal / 8 external. Regional geometric means of case x target x H1/H18/TS ratios.\nKCI uses corresponding steady-state test distributions; panel scales vary; seed42.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.22,wspace=.3)
    save(fig,'06_regional_kci_vs_avgrelrmse')
    contact=Image.new('RGB',(1800,2100),'white')
    for i,name in enumerate(figures):
        with Image.open(folder/f'{name}.png') as im:thumb=im.convert('RGB')
        thumb.thumbnail((900,700));contact.paste(thumb,((i%2)*900+(900-thumb.width)//2,(i//2)*700+(700-thumb.height)//2))
    contact.save(folder/'contact_sheet.png')
    contact.convert('L').save(folder/'contact_sheet_grayscale.png')
    lines=['# 三家族最佳模型與 No-Gain：Test 比較','',
           '依各家族 validation AvgRelRMSE 最低值固定選定 Gain=250、SS-only=2.5、Gain+SS=(2.5,2.5)，seed42；test 分數未參與選擇。保留 No-Gain 基準。','',
           '動態核心：R5／R5-1／R5-2／R5-6 的 test split；動態近外插：R5-19～26。穩態 KCI：lhs_test_in_range.xlsx（4385點）、lhs_test_soft_ood.xlsx（2692點，與先前 _3.xlsx 內容相同）。','',
           '## 五張表格與第六張內外插圖','']
    for name in figures:lines.append(f'- [{name} PNG]({name}.png)／[PDF]({name}.pdf)')
    lines += ['','## 資料與計算定義','',
              'Δ (%) = 100 × (RMSE_model / RMSE_NoGain − 1)，負值表示改善。No-Gain 原單位 RMSE 乘以1000，表頭標示10⁻³ kmol/kmol；百分比以未四捨五入的 RMSE 計算。','',
              'KCI 整體以四個增益配對的 correct／valid 合併，不是四個百分比的算術平均。零動態增益視為錯誤；有效 ANN gain 門檻1e-5，兩方向各0.5個訓練標準差，warmup35步後逐點判斷全部18步。','',
              '區域 AvgRelRMSE 先計算同案例、同目標、同H1/H18/TS的模型／No-Gain比值。核心24項、近外插48項分別取幾何平均；整體sqrt(I×E)，內外各50%。第六張圖呈現區域分數，No-Gain有實際KCI座標上的點及y=1參考線。','',
              '- [選模來源](selected_models.csv)／[來源與保護雜湊](run_protocol.json)',
              '- [整體KCI](overall_kci.csv)／[配對KCI與計數](gain_pair_kci.csv)',
              '- [核心H1/H18/TS表](internal_horizon_rmse_table.csv)／[外插SO2 H1/H18/TS表](external_so2_horizon_rmse_table.csv)／[外插H2S H1/H18/TS表](external_h2s_horizon_rmse_table.csv)',
              '- [可複製編輯的合併儲存格表格（HTML）](grouped_rmse_tables.html)',
              '- [原核心TS摘要](internal_ts_rmse_table.csv)／[原外插SO2 H18/TS摘要](external_so2_rmse_table.csv)／[原外插H2S H18/TS摘要](external_h2s_rmse_table.csv)',
              '- [288項指標](test_metrics.csv)／[1728項horizon指標](horizon_metrics.csv)／[1824項RMSE重核](RMSE_recheck.csv)',
              '- [與No-Gain逐項比較](paired_case_comparison.csv)／[區域繪圖座標](regional_plot_coordinates.csv)／[原始預測](predictions/)',
              '- [整體AvgRelRMSE](avgrelrmse_overall.csv)／[資料案例](data_audit.csv)／[數值核對](evaluation_validation.json)／[圖表核對](plot_validation.json)','',
              '第3、4、5張已改為每個案例分列H1/H18/TS的表格，案例欄在三列中置中。綠色代表RMSE低於No-Gain，紅色代表高於No-Gain；每列最低RMSE（包含No-Gain）以粗體標示。第3張原檔名保留internal_ts以維持既有連結，內容現已包含兩個品質的全部三種時域。','',
              '本次只評估固定checkpoint。這些是既有test案例，不宣稱未曾檢視的新盲測；單一seed、相關horizon不是獨立重複實驗。KCI是ANN-reference方向一致性，不代表物理增益正確性或閉迴路性能。','',
              '完整重跑：`python -m scripts.prediction.compare_best_models_test`。只重畫：`python results/visualization/best_gain_ss_gainss_test_seed42/rebuild_figures.py`。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    validation=dict(passed=True,figure_count=6,figure_names=figures,table_rows=[4,8,24,24,24],
                    grouped_horizons=['H1','H18','TS'],grouped_delta_cells=216,green_improvement_red_deterioration=True,
                    selection_uses_validation_only=True,delta_computed_from_unrounded_RMSE=True,
                    KCI_micro_reconstructed=True,NoGain_point_in_both_regional_panels=True,
                    regional_ratio_counts={'internal':24,'external':48},visual_review='pending',
                    shared_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix in {'.csv','.png','.pdf','.md','.html'}})
    (folder/'plot_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print('PASS: five test tables and one regional test comparison figure built.',flush=True)
