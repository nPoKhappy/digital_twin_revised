"""Identical numerical checks, figures and documentation for Gain/SS validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

WEIGHTS = [.05, 2.5, 25, 250, 2500]
STYLES = [('#0072B2', 'o'), ('#D55E00', '^'), ('#6A3D9A', 'D'), ('#009E73', 's'), ('#CC79A7', 'P')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(folder):
    folder = Path(folder)
    tables = {name: pd.read_csv(folder/f'{name}.csv') for name in [
        'avgrelrmse_overall', 'avgrelrmse_by_region', 'avgrelrmse_by_evaluation',
        'avgrelrmse_by_region_evaluation', 'relative_rmse_components',
        'validation_metrics', 'validation_rmse_summary']}
    overall, regional, horizons = [tables[n] for n in ['avgrelrmse_overall', 'avgrelrmse_by_region', 'avgrelrmse_by_region_evaluation']]
    components, metrics = tables['relative_rmse_components'], tables['validation_metrics']
    protocol = json.loads((folder/'metric_protocol.json').read_text(encoding='utf-8'))
    combined = protocol.get('model_family') == 'gain_ss'
    gain = 'gain_weight' in overall.columns
    weight_col = 'gain_weight' if gain else 'ss_weight'
    title, prefix, fixed = ('Gain-only', 'Gain', 'SS=0') if gain else ('SS-only', 'SS', 'Gain=0')
    if combined:
        title, prefix, fixed = 'Gain+SS', 'Gain=SS', 'Gain=SS; SS targets: all 48 outputs'
        np.testing.assert_array_equal(overall.gain_weight, overall.ss_weight)
    models = overall.sort_values(weight_col)[['model', weight_col]].values.tolist()
    assert [float(w) for _, w in models] == WEIGHTS and len(components) == 180
    assert components.groupby('model').size().eq(36).all()
    assert components.n.eq(components.NoGain_n).all()
    np.testing.assert_allclose(components.relative_RMSE, components.RMSE/components.NoGain_RMSE, rtol=1e-12)
    kci_file = 'gain_ss_kci_components.csv' if combined else ('kci_pair_counts.csv' if gain else 'ss_only_kci_components.csv')
    kci = pd.read_csv(folder/kci_file)
    coordinates = regional[['model', weight_col, 'partition', 'micro_kci_percent', 'AvgRelRMSE']].rename(columns={weight_col:'loss_weight', 'micro_kci_percent':'ANN_reference_KCI_percent'})
    for row in overall.itertuples():
        c = components[components.model == row.model]
        region_scores = {}
        for region in ['internal', 'external']:
            r = c[c.partition == region]
            assert len(r) == 18 and np.isclose(r.overall_log_weight.sum(), .5)
            region_scores[region] = np.exp(np.log(r.relative_RMSE).mean())
            selected = regional[(regional.model == row.model) & (regional.partition == region)].iloc[0]
            np.testing.assert_allclose(selected.AvgRelRMSE, region_scores[region], rtol=1e-12)
            counts = kci[(kci.model == row.model) & (kci.distribution == ('in_range' if region == 'internal' else 'soft_ood'))]
            np.testing.assert_allclose(selected.micro_kci_percent, 100*(counts.tp+counts.tn).sum()/counts.valid_comparisons.sum(), rtol=1e-12)
        np.testing.assert_allclose(row.AvgRelRMSE, np.sqrt(region_scores['internal']*region_scores['external']), rtol=1e-12)
        counts = kci[kci.model == row.model]
        np.testing.assert_allclose(row.overall_micro_KCI_percent, 100*(counts.tp+counts.tn).sum()/counts.valid_comparisons.sum(), rtol=1e-12)
    for row in horizons.itertuples():
        selected = components[(components.model == row.model) & (components.partition == row.partition) & (components.evaluation == row.evaluation)]
        assert len(selected) == 6
        np.testing.assert_allclose(row.AvgRelRMSE, np.exp(np.log(selected.relative_RMSE).mean()), rtol=1e-12)
    for row in tables['avgrelrmse_by_evaluation'].itertuples():
        selected = horizons[(horizons.model == row.model) & (horizons.evaluation == row.evaluation)]
        assert len(selected) == 2
        np.testing.assert_allclose(row.AvgRelRMSE, np.sqrt(selected.AvgRelRMSE.prod()), rtol=1e-12)
    coordinates.to_csv(folder/'regional_plot_coordinates.csv', index=False)
    overall[['model', weight_col, 'overall_micro_KCI_percent', 'AvgRelRMSE']].rename(columns={weight_col:'loss_weight', 'overall_micro_KCI_percent':'ANN_reference_KCI_percent'}).to_csv(folder/'overall_plot_coordinates.csv', index=False)
    raw = metrics.groupby(['partition', 'model', 'target', 'evaluation']).agg(cases=('case', 'size'), mean_case_RMSE=('RMSE', 'mean')).reset_index()
    raw['loss_weight'] = raw.model.map(dict(models)).fillna(0)
    raw = raw[['partition', 'model', 'loss_weight', 'target', 'evaluation', 'cases', 'mean_case_RMSE']]
    prior = tables['validation_rmse_summary'].merge(raw, on=['partition','model','target','evaluation'], suffixes=('_saved','_recomputed'), validate='one_to_one')
    assert len(prior) == 60
    np.testing.assert_allclose(prior.mean_case_RMSE_saved, prior.mean_case_RMSE_recomputed, rtol=1e-12)
    raw.to_csv(folder/'rmse_summary_with_no_gain.csv', index=False)
    assert protocol['internal_validation'] == ['R5-3', 'R5-4', 'R5-5']
    assert protocol['external_validation'] == ['R5-27', 'R5-28', 'R5-29']
    assert protocol['region_weights'] == [.5, .5]
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'], 'font.size':10, 'pdf.fonttype':42})
    names = []

    def save(fig, name):
        fig.canvas.draw()
        for ax in fig.axes:
            for line in ax.lines:
                x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
                assert np.isfinite(x).all() and np.isfinite(y).all()
                assert ((y >= ax.get_ylim()[0]) & (y <= ax.get_ylim()[1])).all()
        for extension in ['png', 'pdf']:
            fig.savefig(folder/f'{name}.{extension}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        with Image.open(folder/f'{name}.png') as image:
            image.convert('L').save(folder/f'{name}_grayscale.png')
        names.append(name)

    def scatter(ax, rows, x_column, annotate=False):
        for (alias, weight), (color, marker) in zip(models, STYLES):
            row = rows[rows.model == alias].iloc[0]
            ax.plot(row[x_column], row.AvgRelRMSE, color=color, marker=marker, ms=8, ls='none', label=f'{prefix}={weight:g}')
            if annotate:
                ax.annotate(f'{weight:g}: {row.AvgRelRMSE:.4f}', (row[x_column], row.AvgRelRMSE), xytext=(5, 9), textcoords='offset points', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=.3))
        ax.axhline(1, color='.35', ls='--', lw=1, label='No-Gain reference')
        low, high = min(rows.AvgRelRMSE.min(), 1), max(rows.AvgRelRMSE.max(), 1)
        pad = max((high-low)*.2, .05)
        ax.set(ylim=(max(0, low-pad), high+pad), xlabel='ANN-reference KCI (%)', ylabel='AvgRelRMSE (No-Gain = 1)')
        ax.margins(x=.25)
        ax.ticklabel_format(axis='x', useOffset=False, style='plain')
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)

    fig, ax = plt.subplots(figsize=(10,6.5))
    scatter(ax, overall, 'overall_micro_KCI_percent', True)
    ax.set_title(f'{title} validation | Casewise ratios; regions weighted 50% / 50%', fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False)
    fig.text(.5,.025,f'Y: same-case RMSE ratios; geometric mean within regions, then sqrt(internal x external).\nX: micro KCI pooled across steady-state validation distributions. {fixed}; seed42; no error bars.',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.10,right=.80,top=.88,bottom=.19)
    save(fig,'overall_kci_vs_avgrelrmse')
    fig, axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax, region in zip(axes,['internal','external']):
        scatter(ax, regional[regional.partition == region], 'micro_kci_percent')
        ax.set_title(f'{region.capitalize()} validation',fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
    fig.text(.5,.02,f'Geometric mean of case x target x H1/H18/TS ratios. Panel scales vary. {fixed}; seed42.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.19,wspace=.30)
    save(fig,'regional_kci_vs_avgrelrmse')
    fig, axes = plt.subplots(2,3,figsize=(12,8))
    for i, region in enumerate(['internal','external']):
        for j, evaluation in enumerate(['H1','H18','TS']):
            ax=axes[i,j]
            scatter(ax,horizons[(horizons.partition==region)&(horizons.evaluation==evaluation)],'micro_kci_percent')
            ax.set_title(f'{region.capitalize()} | {evaluation}')
            if j: ax.set_ylabel('')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=6,frameon=False,fontsize=9)
    fig.text(.5,.015,'Same-case ratios; geometric mean over cases x two targets in each panel. Related trajectories; panel scales vary.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,top=.87,bottom=.12,hspace=.5,wspace=.35)
    save(fig,'horizon_kci_vs_avgrelrmse')
    fig, axes = plt.subplots(2,2,figsize=(11,8))
    for i, region in enumerate(['internal','external']):
        for j, target in enumerate(['B35_H2S','B35_SO2']):
            ax=axes[i,j]
            baseline=raw[(raw.partition==region)&(raw.target==target)&(raw.model=='no_gain')].set_index('evaluation')
            ax.plot(range(3),baseline.loc[['H1','H18','TS'],'mean_case_RMSE']*1000,color='.35',ls='--',label='No-Gain',lw=1.5)
            for (alias, weight),(color, marker) in zip(models,STYLES):
                rows=raw[(raw.partition==region)&(raw.target==target)&(raw.model==alias)].set_index('evaluation')
                ax.plot(range(3),rows.loc[['H1','H18','TS'],'mean_case_RMSE']*1000,color=color,marker=marker,label=f'{prefix}={weight:g}',lw=1.3)
            ax.set_title(f'{region.capitalize()} | {target.removeprefix("B35_")}')
            ax.set_xticks(range(3),['H1','H18','TS'])
            ax.set_ylabel(r'Mean case RMSE ($10^{-3}$ kmol/kmol)')
            ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.text(.5,.015,f'Raw RMSE: arithmetic case means; AvgRelRMSE: geometric means of same-case ratios. {fixed}; seed42.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,top=.86,bottom=.09,hspace=.40,wspace=.30)
    save(fig,'validation_rmse_summary')
    contact=Image.new('RGB',(1800,1400),'white')
    for i,name in enumerate(names):
        with Image.open(folder/f'{name}.png') as image: thumb=image.convert('RGB')
        thumb.thumbnail((900,700))
        contact.paste(thumb,((i%2)*900+(900-thumb.width)//2,(i//2)*700))
    contact.save(folder/'contact_sheet.png')
    contact.convert('L').save(folder/'contact_sheet_grayscale.png')
    label='Gain+SS' if combined else ('Gain' if gain else 'SS-only')
    relative_folder=protocol.get('output_directory',f'results/visualization/{folder.name}')
    lines=[f'# 五個 {label} 超參數 validation 比較','',
           ('Gain 與 SS 權重同步為 0.05、2.5、25、250、2500；seed42；SS 使用全部 48 個輸出。與五個 Gain、五個 SS-only 使用相同分析、檔案結構及產圖程式。' if combined else '權重：0.05、2.5、25、250、2500；seed42。五個 Gain 與五個 SS-only 使用相同分析、檔案結構及產圖程式。'),'',
           '**內插：R5-3、R5-4、R5-5；外插：R5-27、R5-28、R5-29。這是 validation，比三模型 test 使用不同案例。**','',
           '## 內插／外插比較圖','',
           '[內外插雙面板 PNG](regional_kci_vs_avgrelrmse.png)／[PDF](regional_kci_vs_avgrelrmse.pdf)／[灰階](regional_kci_vs_avgrelrmse_grayscale.png)／[10 個繪圖座標](regional_plot_coordinates.csv)','',
           'X 是對應穩態 validation 分布的 ANN-reference micro KCI；Y 是該區域內 18 項同案例 RMSE 比值的幾何平均。No-Gain=1；同一權重在兩組圖使用相同顏色與標記，各面板座標範圍依資料調整。','',
           '## 一致的計算定義','',
           '先以同案例、同目標、同 H1/H18/TS 的模型 RMSE 除以 No-Gain RMSE。內外各 18 項分別取幾何平均 I、E；整體 AvgRelRMSE=sqrt(I×E)，內外各 50%。沒有先平均原始 RMSE 再計算比值。','',
           '整體 KCI 合併兩個穩態 validation 分布的正確數／有效比較數；區域圖使用各自的計數。KCI 並不採內外各 50%，且不是從 R5-27/28/29 動態案例計算。','',
           '## 整體與區域分數','','| 權重 | 內部 AvgRelRMSE | 外部 AvgRelRMSE | 整體 AvgRelRMSE | ANN-reference KCI (%) |','|---:|---:|---:|---:|---:|']
    for row in overall.to_dict('records'):
        lines.append(f'| {row[weight_col]:g} | {row["internal_AvgRelRMSE"]:.9f} | {row["external_AvgRelRMSE"]:.9f} | {row["AvgRelRMSE"]:.9f} | {row["overall_micro_KCI_percent"]:.6f} |')
    lines+=['','## H1／H18／TS：內外各 50%','','| 權重 | H1 | H18 | TS |','|---:|---:|---:|---:|']
    for alias,weight in models:
        rows=tables['avgrelrmse_by_evaluation'][tables['avgrelrmse_by_evaluation'].model==alias].set_index('evaluation')
        lines.append(f'| {weight:g} | {rows.loc["H1","AvgRelRMSE"]:.9f} | {rows.loc["H18","AvgRelRMSE"]:.9f} | {rows.loc["TS","AvgRelRMSE"]:.9f} |')
    lines+=['','## 共用圖表與檔案','',
            '- [整體 KCI × AvgRelRMSE](overall_kci_vs_avgrelrmse.png)／[PDF](overall_kci_vs_avgrelrmse.pdf)／[座標](overall_plot_coordinates.csv)',
            '- [內外插雙面板](regional_kci_vs_avgrelrmse.png)／[PDF](regional_kci_vs_avgrelrmse.pdf)／[座標](regional_plot_coordinates.csv)',
            '- [H1/H18/TS × 內外插六面板](horizon_kci_vs_avgrelrmse.png)／[PDF](horizon_kci_vs_avgrelrmse.pdf)',
            '- [原單位 RMSE 四面板，含 No-Gain](validation_rmse_summary.png)／[PDF](validation_rmse_summary.pdf)／[CSV](rmse_summary_with_no_gain.csv)',
            '- [整體分數](avgrelrmse_overall.csv)／[區域分數](avgrelrmse_by_region.csv)／[H1/H18/TS 分數](avgrelrmse_by_evaluation.csv)',
            '- [區域 × H1/H18/TS](avgrelrmse_by_region_evaluation.csv)／[180 個逐案例比值與原始 RMSE](relative_rmse_components.csv)',
            '- [216 項指標，含 No-Gain](validation_metrics.csv)／[1296 項 H1～H18 指標](horizon_metrics.csv)／[1368 項 RMSE 重算](RMSE_recheck.csv)',
            f'- [資料來源](data_audit.csv)／[{"No-Gain 內插重現核對" if combined else "內插重現核對"}](internal_metrics_recheck.csv)／[預測陣列](predictions/)',
            f'- [KCI 計數]({kci_file})／[計算協定](metric_protocol.json)／[評估核對](evaluation_validation.json)／[圖表核對](plot_validation.json)','',
            f'目標權重欄位為 `{weight_col}`；共用繪圖座標統一使用 `loss_weight`。Gain、SS-only 與 Gain+SS 是不同訓練目標，因此數值與排名可以不同。','',
            '## 重畫與重跑','',
            '僅重畫（不重跑模型）：','', '```powershell',f'.\\.venv\\Scripts\\python.exe {relative_folder}/rebuild_figures.py','```','',
            ('完整重跑這五組 Gain+SS validation（固定 checkpoint 推論，不訓練）：' if combined else '完整重跑兩組 validation：'),'', '```powershell',protocol.get('full_rerun_command','.\\.venv\\Scripts\\python.exe scripts/prediction/validate_hyperparameters.py'),'```','',
            'H1/H18 每個起點重新給 18 點真實歷史，再遞迴至第 1／18 步；TS 僅最初 18 點使用真實歷史，其後持續回填預測。皆使用已知未來 decoder inputs。R5-5 沿用內部分類，部分操作點超出 core。','',
            ('KCI 使用 lhs_valid_in_range.xlsx 與 lhs_valid_soft_ood.xlsx 全部有效探測點重新評估，沒有使用訓練 KCI history。' if combined else 'KCI 沿用已核對來源的穩態 validation 結果。')+'單一 seed42、相關軌跡不是獨立重複實驗；未新增誤差棒。ANN-reference KCI 不代表物理正確性或閉迴路控制驗證。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    report=dict(passed=True,shared_report_module='scripts/prediction/validation_comparison_report.py',shared_report_sha256=sha(Path(__file__)),
                model_count=5,component_count_per_model=36,regional_components_per_model=18,matched_ratios=180,
                regional_plot_point_count=10,geometric_aggregation_recomputed=True,KCI_counts_recomputed=True,
                horizon_aggregation_checked=True,raw_RMSE_summary_checked=True,
                region_weights=[.5,.5],common_weight_colors_and_markers=True,figure_names=names,errorbars=False,
                raw_RMSE_includes_NoGain=True,visual_review='pending',
                output_sha256={p.name:sha(p) for p in folder.iterdir() if p.suffix in ['.csv','.png','.pdf','.py','.md']})
    (folder/'plot_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'PASS: {label}; same four figure families, 180 ratios checked, 10 regional plot coordinates checked.',flush=True)
