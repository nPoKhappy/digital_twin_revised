"""Compare existing H1/H18/TS metrics; add H1/H18 figures without inference."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_absolute_rmse_across_seeds import ROOT, SEEDS, INTERNAL, EXTERNAL, TARGETS, STYLES, sha, table

OUT = ROOT / 'results/visualization/h1_h18_rmse_seed42_43_44'


def plot(data, evaluation, target):
    label = r'H$_2$S' if target == TARGETS[0] else r'SO$_2$'
    suffix = 'h2s' if target == TARGETS[0] else 'so2'
    fig, axes = plt.subplots(3, 4, figsize=(12.6, 9.2))
    for i, (axis, case) in enumerate(zip(axes.flat, INTERNAL + EXTERNAL)):
        subset = data[(data.evaluation == evaluation) & (data.target == target) & (data['case'] == case)]
        for model, style in STYLES.items():
            values = subset[subset.model == model].set_index('seed').loc[SEEDS, 'RMSE']
            axis.plot(SEEDS, values, **style, markersize=5.3, linewidth=1.4)
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
        if i % 4 == 0:
            axis.set_ylabel('Absolute RMSE')
        if i >= 8:
            axis.set_xlabel('Training seed')
    handles = [Line2D([], [], **style, linewidth=1.5, markersize=6, label=model) for model, style in STYLES.items()]
    handles.append(Line2D([], [], marker='o', linestyle='none', color='#222222', markerfacecolor='none', markersize=9, label='Lowest of the six results'))
    fig.suptitle(f'Absolute {evaluation} RMSE across training seeds: {label}', fontsize=15, fontweight='bold', y=.99)
    fig.text(.5, .947, 'Gain and Difficult-GradNorm; lower RMSE is better', ha='center', fontsize=11)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .927), ncol=3, frameon=False)
    fig.text(.5, .027, 'Each panel is one case. Y-axis ranges vary by case; compare seeds and methods within the same panel.', ha='center', fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.09, left=.07, right=.985, hspace=.50, wspace=.33)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT / f'absolute_{evaluation.lower()}_rmse_{suffix}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected, frames = {}, []
    verified = 0
    for seed in SEEDS:
        source = ROOT / f'results/visualization/four_model_rmse_visualization_seed{seed}/four_model_rmse_values.csv'
        protected[source.relative_to(ROOT).as_posix()] = sha(source)
        frame = pd.read_csv(source)
        frame = frame[frame.model.isin(STYLES) & frame.evaluation.isin(['H1', 'H18', 'TS'])].copy()
        assert len(frame) == 144 and not frame.duplicated(['model', 'case', 'target', 'evaluation']).any()
        assert set(frame['case']) == set(INTERNAL + EXTERNAL)
        frame['seed'] = seed
        frame['distribution'] = np.where(frame['case'].isin(INTERNAL), 'internal_test', 'external_all')
        frames.append(frame[['seed', 'model', 'case', 'target', 'distribution', 'evaluation', 'RMSE']])
        for model in STYLES:
            suffix = '_difficult_group_gradnorm_no_ss' if model == 'Difficult-GradNorm' else ''
            experiment = f'transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed{seed}{suffix}'
            for case in INTERNAL + EXTERNAL:
                directory = ROOT / 'results' / experiment / (case + ('_test_split' if case in INTERNAL else ''))
                paths = [directory / 'horizon_metrics_t1_to_t18.csv', directory / 'evaluation_metrics.csv']
                for path in paths:
                    protected[path.relative_to(ROOT).as_posix()] = sha(path)
                horizons = pd.read_csv(paths[0]).set_index(['Horizon', 'Variable'])
                ts = pd.read_csv(paths[1]).set_index('Variable')
                for target in TARGETS:
                    for evaluation in ['H1', 'H18', 'TS']:
                        rows = frame[(frame.model == model) & (frame['case'] == case) & (frame.target == target) & (frame.evaluation == evaluation)]
                        assert len(rows) == 1
                        expected = ts.loc[target, 'RMSE'] if evaluation == 'TS' else horizons.loc[(int(evaluation[1:]), target), 'RMSE']
                        np.testing.assert_allclose(rows.RMSE.iloc[0], expected, rtol=1e-12, atol=1e-16)
                        verified += 1
    data = pd.concat(frames, ignore_index=True)
    assert len(data) == 432 and np.isfinite(data.RMSE).all() and (data.RMSE > 0).all()
    data.to_csv(OUT / 'absolute_rmse_values.csv', index=False)
    keys = ['seed', 'case', 'target', 'distribution', 'evaluation']
    paired = data.pivot(index=keys, columns='model', values='RMSE').reset_index()
    paired['change_percent'] = 100 * (paired['Difficult-GradNorm'] / paired.Gain - 1)
    paired['improved'] = paired['Difficult-GradNorm'] < paired.Gain
    assert len(paired) == 216
    paired.to_csv(OUT / 'case_rmse_changes.csv', index=False)
    per_seed = paired.groupby(['evaluation', 'target', 'distribution', 'seed'], sort=False).agg(
        mean_change_percent=('change_percent', 'mean'), median_change_percent=('change_percent', 'median'),
        improved_cases=('improved', 'sum'), cases=('case', 'size')).reset_index()
    per_seed.to_csv(OUT / 'per_seed_summary.csv', index=False)
    summary = paired.groupby(['evaluation', 'target', 'distribution'], sort=False).agg(
        mean_change_percent=('change_percent', 'mean'), improved_seed_case_pairs=('improved', 'sum'),
        seed_case_pairs=('case', 'size')).reset_index()
    stable = paired.groupby(['evaluation', 'target', 'distribution', 'case'], sort=False).improved.all()
    counts = stable.groupby(level=['evaluation', 'target', 'distribution']).agg(['sum', 'size']).reset_index()
    summary = summary.merge(counts.rename(columns={'sum': 'cases_improved_all_seeds', 'size': 'cases'}))
    summary.to_csv(OUT / 'all_seed_summary.csv', index=False)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    for evaluation in ['H1', 'H18']:
        for target in TARGETS:
            plot(data, evaluation, target)
    report = '# H1、H18 與 TS 跨種子比較\n\n'
    report += '比較 Gain 與 Difficult-GradNorm（no SS），使用 seed 42、43、44。H1 是第 1 步，H18 是第 18 步；TS 沿用原有 rolling time-series 評估。三者分開計算，不能視為獨立重複實驗。\n\n'
    report += '變化率 = 100 × (Difficult-GradNorm RMSE / 同一 seed 的 Gain RMSE − 1)，負值表示改善。先算各案例百分比，再於各 seed、各資料分布內等權平均；跨 seed 統計亦等權平均，並非合併時序後重算的 RMSE。internal 有 4 個案例，external 有 8 個案例。\n\n'
    report += '## 主要觀察\n\n'
    report += '- External H₂S：H1、H18、TS 的平均 RMSE 分別降低 16.83%、25.17%、27.39%；改善的 seed × case 配對分別為 20/24、23/24、23/24。在這三個評估下，H18 與 TS 的改善幅度和一致性較高，但尚不能據此推論所有中間 horizon 都逐步改善。\n'
    report += '- Internal H₂S：H1、H18、TS 平均增加 6.71%、7.26%、7.60%。seed43 三項平均改善，seed42 與 seed44 三項平均退步。\n'
    report += '- External SO₂：H1、H18、TS 平均降低 6.75%、10.21%、8.10%，但改善配對只有 14/24、16/24、17/24；不能以平均值宣稱所有案例受益。seed44 的 H18 雖平均降低 10.45%，仍只有 4/8 個案例改善。\n'
    report += '- Internal SO₂：H1、H18、TS 平均增加 10.58%、2.56%、0.78%。H1 的精度代價較明顯；seed44 的 H1 平均增加 19.99%，TS 則平均降低 5.87%。\n\n'
    report += '## 三個種子彙整\n\n' + table(summary.round(4)) + '\n'
    report += '## 各種子\n\n' + table(per_seed.round(4)) + '\n'
    report += '## 原始 RMSE 圖\n\n每張圖有 12 個案例；各面板的 y 軸範圍不同。黑圈是六筆測試結果的事後最小值，不能當作模型選擇規則。\n\n'
    for evaluation in ['H1', 'H18']:
        for target in ['h2s', 'so2']:
            stem = f'absolute_{evaluation.lower()}_rmse_{target}'
            report += f'### {evaluation} {target.upper()}\n\n![{evaluation} {target}]({stem}.png)\n\n[向量 PDF]({stem}.pdf)\n\n'
    report += '## 判讀限制\n\n改善次數是描述性的 seed × case 計數，並非獨立訓練次數；三個種子不足以證明普遍優越。這些圖評估預測誤差，不能直接判定 KCI、物理正確性或誤差來源。\n\n[逐筆原始值](absolute_rmse_values.csv) · [逐案例配對變化](case_rmse_changes.csv) · [來源與驗證](validation_and_provenance.json)\n'
    (OUT / 'comparison_report.md').write_text(report, encoding='utf-8')
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest, name
    provenance = {'passed': True, 'rmse_values_verified': verified, 'h1_h18_values_verified': 288,
                  'protected_sha256': protected, 'script_sha256': sha(Path(__file__)),
                  'visual_review': 'pending', 'new_inference': False,
                  'change_percent_definition': '100 * (Difficult-GradNorm / same-seed Gain - 1)',
                  'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'validation_and_provenance.json'}}
    (OUT / 'validation_and_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('PASS:', verified, 'RMSE values verified;', len(protected), 'source files unchanged.')
    print(summary.to_string(index=False))
    print(per_seed.to_string(index=False))


if __name__ == '__main__':
    main()
