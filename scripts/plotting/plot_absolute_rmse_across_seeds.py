"""Plot case-wise absolute TS RMSE for Gain and Difficult-GradNorm, seeds 42-44."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/visualization/absolute_rmse_seed42_43_44'
SEEDS = [42, 43, 44]
INTERNAL = ['R5', 'R5-1', 'R5-2', 'R5-6']
EXTERNAL = [f'R5-{i}' for i in range(19, 27)]
TARGETS = ['B35_H2S', 'B35_SO2']
STYLES = {'Gain': {'color': '#0072B2', 'marker': 'o', 'linestyle': '-'},
          'Difficult-GradNorm': {'color': '#D81B60', 'marker': 'D', 'linestyle': '--'}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(frame):
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(str(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected, frames = {}, []
    verified = 0
    for seed in SEEDS:
        source = ROOT / f'results/visualization/four_model_rmse_visualization_seed{seed}/four_model_rmse_values.csv'
        protected[source.relative_to(ROOT).as_posix()] = sha(source)
        data = pd.read_csv(source)
        data = data[(data.evaluation == 'TS') & data.model.isin(STYLES)].copy()
        assert len(data) == 48 and not data.duplicated(['model', 'case', 'target']).any()
        assert set(data['case']) == set(INTERNAL + EXTERNAL)
        data['seed'] = seed
        data['distribution'] = np.where(data['case'].isin(INTERNAL), 'internal_test', 'external_all')
        frames.append(data[['seed', 'model', 'case', 'target', 'distribution', 'evaluation', 'RMSE']])
        for model in STYLES:
            suffix = '_difficult_group_gradnorm_no_ss' if model == 'Difficult-GradNorm' else ''
            experiment = f'transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed{seed}{suffix}'
            for case in INTERNAL + EXTERNAL:
                case_dir = case + ('_test_split' if case in INTERNAL else '')
                source = ROOT / 'results' / experiment / case_dir / 'evaluation_metrics.csv'
                protected[source.relative_to(ROOT).as_posix()] = sha(source)
                original = pd.read_csv(source).set_index('Variable')
                for target in TARGETS:
                    row = data[(data.model == model) & (data['case'] == case) & (data.target == target)]
                    assert len(row) == 1
                    np.testing.assert_allclose(row.RMSE.iloc[0], original.loc[target, 'RMSE'], rtol=1e-12, atol=1e-16)
                    verified += 1
    data = pd.concat(frames, ignore_index=True)
    assert np.isfinite(data.RMSE).all() and (data.RMSE > 0).all()
    data.to_csv(OUT / 'absolute_ts_rmse_values.csv', index=False)
    winners = []
    for target in TARGETS:
        for case in INTERNAL + EXTERNAL:
            subset = data[(data.target == target) & (data['case'] == case)]
            overall = subset.loc[subset.RMSE.idxmin()]
            difficult = subset[subset.model == 'Difficult-GradNorm']
            best_difficult = difficult.loc[difficult.RMSE.idxmin()]
            winners.append({'target': target, 'case': case, 'distribution': overall.distribution,
                            'lowest_model': overall.model, 'lowest_seed': int(overall.seed), 'lowest_rmse': overall.RMSE,
                            'lowest_difficult_seed': int(best_difficult.seed), 'lowest_difficult_rmse': best_difficult.RMSE})
    winners = pd.DataFrame(winners)
    winners.to_csv(OUT / 'case_minimum_rmse.csv', index=False)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    for target, label, stem in [(TARGETS[0], r'H$_2$S', 'absolute_ts_rmse_h2s'),
                                (TARGETS[1], r'SO$_2$', 'absolute_ts_rmse_so2')]:
        figure, axes = plt.subplots(3, 4, figsize=(12.6, 9.2))
        for i, (axis, case) in enumerate(zip(axes.flat, INTERNAL + EXTERNAL)):
            subset = data[(data.target == target) & (data['case'] == case)]
            for model, style in STYLES.items():
                values = subset[subset.model == model].set_index('seed').loc[SEEDS, 'RMSE'].to_numpy()
                axis.plot(SEEDS, values, **style, markersize=5.3, linewidth=1.4, label=model)
            best = subset.loc[subset.RMSE.idxmin()]
            axis.scatter([best.seed], [best.RMSE], s=125, facecolors='none', edgecolors='#222222', linewidths=1, zorder=5)
            name = 'Gain' if best.model == 'Gain' else 'Difficult'
            axis.text(.04, .95, f'Lowest: {name}, seed {int(best.seed)}', transform=axis.transAxes, fontsize=8, va='top', color='#444444')
            domain = 'Internal test' if case in INTERNAL else 'External'
            axis.set_title(f'{domain}: {case}', loc='left', fontweight='bold')
            axis.set_ylim(0, 1.24 * subset.RMSE.max())
            axis.set_xlim(41.65, 44.35)
            axis.set_xticks(SEEDS)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
            axis.grid(axis='y', color='#DDDDDD', linewidth=.6)
            axis.spines[['top', 'right']].set_visible(False)
            axis.tick_params(length=0, pad=4)
            if i % 4 == 0: axis.set_ylabel('Absolute RMSE')
            if i >= 8: axis.set_xlabel('Training seed')
        handles = [Line2D([], [], **style, linewidth=1.5, markersize=6, label=model) for model, style in STYLES.items()]
        handles.append(Line2D([], [], marker='o', linestyle='none', color='#222222', markerfacecolor='none', markersize=9, label='Lowest of the six results'))
        figure.suptitle(f'Absolute TS RMSE across training seeds: {label}', fontsize=15, fontweight='bold', y=.99)
        figure.text(.5,.947,'Gain and Difficult-GradNorm; lower RMSE is better',ha='center',fontsize=11)
        figure.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.927),ncol=3,frameon=False)
        figure.text(.5,.027,'Each panel is one case. Y-axis ranges vary by case; compare seeds and methods within the same panel.',ha='center',fontsize=9)
        figure.subplots_adjust(top=.84,bottom=.09,left=.07,right=.985,hspace=.50,wspace=.33)
        for extension in ['png','pdf']:
            figure.savefig(OUT / f'{stem}.{extension}', dpi=300, bbox_inches='tight')
        plt.close(figure)

    report = '''# 三個 seed 的絕對 TS RMSE 比較

比較對象為 Gain 與 Difficult-GradNorm 的 seed42、43、44。內部使用 R5、R5-1、R5-2、R5-6 的 test split；外部使用 R5-19～R5-26 全資料。此處呈現原始尺度 RMSE，沒有換算改善百分比，也沒有跨案例平均。

判讀方式：同一小圖內，點越低表示絕對誤差越小。黑色空心圓圈出該案例六個結果的最低值；這是測試結果的描述，不是用測試集挑選部署模型的程序。各案例的 y 軸範圍不同，因此不要用跨小圖的視覺高度比較誤差大小。

## H₂S

[PNG](absolute_ts_rmse_h2s.png) · [向量 PDF](absolute_ts_rmse_h2s.pdf)

![H₂S 絕對 TS RMSE](absolute_ts_rmse_h2s.png)

## SO₂

[PNG](absolute_ts_rmse_so2.png) · [向量 PDF](absolute_ts_rmse_so2.pdf)

![SO₂ 絕對 TS RMSE](absolute_ts_rmse_so2.png)

## 逐案例原始數值

以下保留原始數值的科學記號。同一列比較相同案例與輸出；不以改善百分比替代絕對預測誤差。

'''
    for target in TARGETS:
        report += f'### {target}\n\n'
        rows = []
        for case in INTERNAL + EXTERNAL:
            values = data[(data.target==target)&(data['case']==case)].set_index(['model','seed']).RMSE
            row = {'案例':case}
            for seed in SEEDS:
                row[f'Gain {seed}']=f'{values.loc[("Gain",seed)]:.5e}'
                row[f'Difficult {seed}']=f'{values.loc[("Difficult-GradNorm",seed)]:.5e}'
            rows.append(row)
        report += table(pd.DataFrame(rows))+'\n'
    report += '''## 判斷種子與方法

- 判斷同一方法哪個 seed 較準：比較同案例、同輸出、同 TS 評估的絕對 RMSE。
- 判斷 Difficult-GradNorm 是否帶來改善：比較同 seed 的 Gain 與 Difficult-GradNorm，再完整報告三個 seed 的差異。
- Seed43 的相對改善較一致，不代表其絕對 RMSE 全面最低。例如內部 SO₂ 的四個案例，Difficult seed44 的 RMSE 都低於 Difficult seed43。
- 這兩張圖只有 TS；不據此推論所有 H1、H18 或 KCI 的優劣。

[144 筆絕對 RMSE](absolute_ts_rmse_values.csv) · [逐案例最低值](case_minimum_rmse.csv) · [來源與驗證](validation_and_provenance.json)
'''
    (OUT/'comparison_report.md').write_text(report,encoding='utf-8')
    for name,digest in protected.items(): assert sha(ROOT/name)==digest,name
    provenance={'passed':True,'seeds':SEEDS,'models':list(STYLES),'evaluation':'TS','rmse_values_verified':verified,
                'panels_per_target':12,'cross_case_averaging':False,'y_axis_ranges':'Case-specific; shared by six results within each panel; all start at zero.',
                'protected_sha256':protected,'script_sha256':sha(Path(__file__)),'visual_review':'pending',
                'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='validation_and_provenance.json'}}
    (OUT/'validation_and_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('PASS: 144 original TS RMSE values and 24 case minima verified; original sources preserved.')
    print(winners.to_string(index=False))


if __name__ == '__main__':
    main()
