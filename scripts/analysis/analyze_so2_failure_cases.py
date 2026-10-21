"""Diagnose existing SO2 rolling predictions without training or changing models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/so2_failure_diagnostics_seed42_44_20261021'
CASES = [(42, 'R5-23'), (44, 'R5-21'), (44, 'R5-22'), (44, 'R5-25'), (44, 'R5-26')]
TARGET = 'B35_SO2'
COLORS = {'Truth': '#222222', 'Gain': '#0072B2', 'Difficult-GradNorm': '#D81B60'}
protected = {}


def protect(path):
    path = Path(path)
    protected[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def config_for(seed, difficult=False):
    group = 'difficult_gradnorm' if difficult else 'gain'
    suffix = '_difficult_group_gradnorm_no_ss' if difficult else ''
    path = ROOT / f'configs/pgin/{group}/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed{seed}{suffix}.yaml'
    return yaml.safe_load(protect(path).read_text(encoding='utf-8'))


def safe_corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float('nan')


def markdown(frame):
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)) + '\n'


def shade(axis, time, mask, dt):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        axis.axvspan(max(0, time[start] - dt), time[end - 1], color='#999999', alpha=0.12, zorder=0)


def style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', color='#DDDDDD', linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=0, pad=5)


def write_report():
    summary = pd.read_csv(OUT/'case_diagnosis_summary.csv')
    metrics = pd.read_csv(OUT/'per_model_diagnostics.csv')
    phases = pd.read_csv(OUT/'rapid_and_other_metrics.csv')
    sensitivity = pd.read_csv(OUT/'rapid_threshold_sensitivity.csv')
    display, shapes, rapid_rows = [], [], []
    for r in summary.itertuples(index=False):
        display.append({'案例':f'{r.seed} / {r.case}', 'RMSE 變化':f'{r.rmse_change_percent:+.2f}%',
                        'Gain 平均誤差':f'{r.gain_bias:+.4e}', 'Difficult 平均誤差':f'{r.difficult_bias:+.4e}',
                        '扣除平均誤差後 RMSE 變化':f'{100*(r.difficult_centered_rmse/r.gain_centered_rmse-1):+.2f}%',
                        '新增 MSE 主要項':'平均偏移平方' if r.dominant_added_mse_component=='bias_squared' else '剩餘誤差'})
        shapes.append({'案例':f'{r.seed} / {r.case}', 'Gain 幅度比':f'{r.gain_p90_range_ratio:.3f}',
                       'Difficult 幅度比':f'{r.difficult_p90_range_ratio:.3f}', 'Gain 相關係數':f'{r.gain_correlation:+.3f}',
                       'Difficult 相關係數':f'{r.difficult_correlation:+.3f}',
                       '時序方向一致率 Gain／Difficult':f'{r.gain_temporal_direction_percent:.1f}% / {r.difficult_temporal_direction_percent:.1f}%'})
        sens = sensitivity[(sensitivity.seed==r.seed)&(sensitivity['case']==r.case)]
        rapid_rows.append({'案例':f'{r.seed} / {r.case}', '快速區段點數占比':f'{r.rapid_point_percent:.1f}%',
                           '快速區段占新增 SSE':f'{r.rapid_share_of_additional_sse_percent:.1f}%',
                           'P70／P75／P80 門檻下占新增 SSE':' / '.join(f'{x:.1f}%' for x in sens.rapid_share_of_additional_sse_percent)})
    text = '''# SO₂ 失敗案例診斷：seed42 與 seed44

日期：2026-10-21。分析使用既有預測與原始測試資料，沒有重新訓練、修改權重或重新推論。範圍為 seed42 的 R5-23，以及 seed44 的 R5-21、R5-22、R5-25、R5-26；每例 126 點，共 630 個對齊的時序點。

## 主要發現

五例中有四例，Difficult-GradNorm 相對 Gain 增加的 MSE，主要來自平均低估加重。R5-25 的退步則主要來自扣除平均偏移後仍存在的剩餘誤差，且預測變化幅度進一步縮小。這是誤差的數學分解與曲線診斷，尚未識別造成偏移的訓練機制。

R5-26 同時存在明顯時序形狀問題：兩個模型與真值的相關係數均為負，因此不能把這五例一律描述成「時序方向已學對，只差輸出水準」。

[五案例總覽 PNG](figures/five_case_so2_overview.png) · [向量 PDF](figures/five_case_so2_overview.pdf)

![五案例預測與殘差](figures/five_case_so2_overview.png)

## 1. 平均偏移與剩餘誤差

定義 `error = prediction - truth`，負的平均誤差表示低估。逐例驗證以下恆等式：

`MSE = mean(error)² + mean((error - mean(error))²)`

「扣除平均誤差後 RMSE」是 centered RMSE，僅用來分辨平均水準與剩餘誤差。它使用測試真值計算平均誤差，並非可部署的校正方法，也不是新的無偏測試成績。下表百分比均以同 seed／案例的 Gain 為基準。

'''
    text += markdown(pd.DataFrame(display))
    text += '''
- **Seed42／R5-23**：Difficult 的平均低估由約 7.07×10⁻⁵ 擴大至 2.15×10⁻⁴。新增 MSE 的 99.3% 來自平均誤差平方增加；扣除平均偏移後，兩模型的 RMSE 接近。此例主要應檢查輸出水準，而非先假定需要更強的時序平滑。
- **Seed44／R5-21**：Difficult 的曲線相關係數與 centered RMSE 均較佳，整體水準卻更低。這支持「曲線變化可更接近，但平均偏移仍使總 RMSE 變差」的解釋。
- **Seed44／R5-22**：同樣以平均低估加重為主要退步項；扣除偏移後剩餘誤差較小。幅度比仍略大於 1，不能只靠偏移分解宣稱幅度完全正確。
- **Seed44／R5-25**：兩模型都明顯低估前段高峰，Difficult 的 P95−P5 幅度比由 0.380 降至 0.333，centered RMSE 也增加。此例包含峰值／變化幅度不足，單純加一個常數無法解決。
- **Seed44／R5-26**：平均低估非常大，Difficult 的變化幅度也較小。雖然 centered RMSE 下降，相關係數仍是負值；曲線變平也可能減少 centered RMSE，因此不能據此認定動態追蹤變好。

R5-21、R5-22、R5-26 的「平均誤差平方」新增貢獻分別超過總新增 MSE 的 100%，是因為另一項 centered MSE 反而減少，抵消了部分退步；兩項仍精確加總至總 MSE 變化。

## 2. 幅度與時序走勢

幅度比定義為 `P95−P5(prediction) / P95−P5(truth)`，1 表示兩者的時序變化範圍相近。這不是輸入擾動下的物理增益；相關係數也不等於物理方向正確。

時序方向一致率比較相鄰時點的 `Δprediction` 與 `Δtruth`，僅保留 `|Δtruth| >= max(1e-12, 0.001 × truth的P95−P5)` 的區間，預測不變計為不一致。這與 ANN-reference KCI 的定義、資料及物理意義不同。

'''
    text += markdown(pd.DataFrame(shapes))
    text += '''
## 3. 退步是否集中於快速變化區段

先以每個案例的真值計算 `|ΔSO₂|`，取第 75 百分位以上且大於零的點為快速區段。第一個預測點的變化量使用其前一個已觀測歷史點，因此所有 126 個評估點都有變化量。這五例各有 32 點被選為快速區段，占 25.4%。圖上以灰色帶標示。

「占新增 SSE」是該區段的 `sum(error_Difficult² - error_Gain²)` 除以整例同一總和；它描述退步分布，不是因果貢獻，也不是模型全部誤差的占比。

'''
    text += markdown(pd.DataFrame(rapid_rows))
    text += '''
R5-25 的快速區段只占 25.4% 的點，卻包含 43.8% 的新增 SSE；在 P70、P80 門檻下仍有相對集中的現象。其餘四例約為 21.3%～27.8%，不支持退步主要集中在最快變化點的說法。這不等於已排除所有延遲或動態誤差。

## 4. 對下一個實驗的建議

第一個待驗證假設應是「如何維持 SO₂ 輸出水準，同時保留目前的增益方向一致性」。先在訓練／validation 資料檢查平均殘差是否也有系統性低估，再決定一個受控修正；不要直接把這些測試案例的平均誤差加回預測，當成方法改善。

R5-25 與 R5-26 要另外檢查峰值幅度及動態形狀，不能只套用常數偏移修正。現有分析尚不足以判定應調整哪個 loss 權重，也不能證明偏移是梯度縮放造成。

## 個別案例圖

'''
    for seed,case in CASES:
        stem=f'seed{seed}_{case}_so2_diagnosis'
        text+=f'- Seed{seed}／{case}：[PNG](figures/{stem}.png) · [PDF](figures/{stem}.pdf)\n'
    text+='''
## 資料與驗證

- [逐案例摘要](case_diagnosis_summary.csv)
- [逐模型診斷](per_model_diagnostics.csv)
- [真值／預測／殘差對齊資料](aligned_time_series.csv)
- [快速與其他區段指標](rapid_and_other_metrics.csv)
- [P70／P75／P80 門檻敏感度](rapid_threshold_sensitivity.csv)
- [來源 SHA256 與驗證紀錄](validation_and_provenance.json)

真值前處理沿用原推論順序：10 列 rolling median、每 10 列取樣、移除缺值、SO₂ clip至1e-9後取log、依模型既有統計量標準化與還原。已對齊預測 CSV 的原始索引，並核對十筆正式 TS RMSE。圖中時間依設定的 10 分鐘間隔換算為 nominal hours；SO₂ 保留原始尺度，沒有自行假定 ppm 或其他單位。

既有 KCI 是在獨立穩態操作點上以正負擾動評估 ANN-reference 增益符號；這份報告沒有重新量測五個動態案例的 KCI，因此不以整體 KCI 推論這五條時序曲線的物理增益均正確。
'''
    (OUT/'diagnostic_report.md').write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    for path in ['src/utils.py', 'scripts/prediction/predict.py', 'scripts/training/train_pgin_stepwise_gain.py', 'scripts/training/train_pgin_stepwise_gain_difficult_group_gradnorm.py']:
        protect(ROOT / path)
    series, metrics, phases, sensitivity, cases = [], [], [], [], []
    cached = []
    for seed, case in CASES:
        gain_config = config_for(seed)
        interval = int(gain_config['data'].get('sampling_interval_min', gain_config['window']['sampling_interval_min']))
        assert gain_config['window'].get('use_median_downsampling', True)
        source = ROOT / gain_config['data']['path'] / f'Test_dataform_change_air2_R={case.removeprefix("R")}.csv'
        raw = pd.read_csv(protect(source))
        assert 'DateTime' not in raw.columns
        processed = raw.rolling(interval, min_periods=interval).median(numeric_only=True).iloc[interval-1::interval].reset_index(drop=True)
        processed.dropna(inplace=True)
        truth_full = processed[TARGET].clip(lower=1e-9)
        predictions = {}
        for model, difficult in [('Gain', False), ('Difficult-GradNorm', True)]:
            config = config_for(seed, difficult)
            experiment = config['exp_name']
            folder = ROOT / 'results' / experiment
            protect(ROOT / 'saved_models' / (experiment + '.pth'))
            mean = pd.read_csv(protect(folder / 'zscore_mean.csv'), index_col=0).squeeze('columns')
            std = pd.read_csv(protect(folder / 'zscore_std.csv'), index_col=0).squeeze('columns')
            stats_scale = float(std[TARGET]) if abs(float(std[TARGET])) >= 1e-6 else 1.0
            restored_full = np.exp((np.log(truth_full) - float(mean[TARGET])) / stats_scale * stats_scale + float(mean[TARGET]))
            prediction = pd.read_csv(protect(folder / case / 'prediction_results.csv'), index_col=0)[TARGET]
            assert prediction.index.is_unique and prediction.index.isin(processed.index).all()
            expected_indices = processed.index[gain_config['window']['train_window_mins'] // interval:]
            np.testing.assert_array_equal(prediction.index, expected_indices)
            truth = restored_full.loc[prediction.index].to_numpy()
            if predictions:
                np.testing.assert_array_equal(prediction.index, evaluation_indices)
                np.testing.assert_allclose(truth, observed, rtol=1e-14, atol=0)
            evaluation_indices = prediction.index
            observed = truth
            values = prediction.to_numpy()
            assert np.isfinite(values).all() and np.isfinite(truth).all()
            error = values - truth
            mse = float(np.mean(error ** 2))
            rmse = float(np.sqrt(mse))
            official = pd.read_csv(protect(folder / case / 'evaluation_metrics.csv')).set_index('Variable')
            np.testing.assert_allclose(rmse, official.loc[TARGET, 'RMSE'], rtol=1e-10, atol=1e-15)
            bias = float(error.mean())
            centered_mse = float(np.mean((error - bias) ** 2))
            np.testing.assert_allclose(mse, bias ** 2 + centered_mse, rtol=1e-14, atol=1e-24)
            truth_range = float(np.quantile(truth, .95) - np.quantile(truth, .05))
            assert truth_range > 0
            direction_tolerance = max(1e-12, .001 * truth_range)
            true_difference = np.diff(truth)
            pred_difference = np.diff(values)
            direction_mask = np.abs(true_difference) >= direction_tolerance
            direction_agreement = 100 * float(np.mean(np.sign(true_difference[direction_mask]) * pred_difference[direction_mask] > 0)) if direction_mask.any() else np.nan
            metrics.append({'seed': seed, 'case': case, 'model': model, 'points': len(values), 'rmse': rmse,
                            'mae': float(np.abs(error).mean()), 'bias': bias, 'bias_squared': bias**2,
                            'centered_mse': centered_mse, 'centered_rmse': np.sqrt(centered_mse),
                            'bias_share_of_mse_percent': 100 * bias**2/mse,
                            'trajectory_p90_range_ratio': float((np.quantile(values, .95) - np.quantile(values, .05))/truth_range),
                            'trajectory_std_ratio': float(np.std(values)/np.std(truth)), 'trajectory_correlation': safe_corr(truth, values),
                            'temporal_direction_agreement_percent': direction_agreement, 'direction_intervals': int(direction_mask.sum()),
                            'direction_tolerance': direction_tolerance})
            predictions[model] = values
        time = (evaluation_indices.to_numpy() - evaluation_indices[0]) * interval / 60
        # Use the preceding observed history value for the first evaluation-point change.
        absolute_change = truth_full.diff().abs().loc[evaluation_indices].to_numpy()
        assert np.isfinite(absolute_change).all()
        threshold = float(np.quantile(absolute_change, .75))
        rapid = (absolute_change >= threshold) & (absolute_change > 0)
        assert rapid.any() and (~rapid).any()
        frame = pd.DataFrame({'seed': seed, 'case': case, 'sample_index': evaluation_indices, 'nominal_time_hours': time,
                              'truth_so2': observed, 'gain_so2': predictions['Gain'], 'difficult_so2': predictions['Difficult-GradNorm'],
                              'absolute_truth_change_per_sample': absolute_change, 'rapid_change': rapid})
        frame['gain_error'] = frame.gain_so2 - frame.truth_so2
        frame['difficult_error'] = frame.difficult_so2 - frame.truth_so2
        series.append(frame)
        for model, error in [('Gain', frame.gain_error.to_numpy()), ('Difficult-GradNorm', frame.difficult_error.to_numpy())]:
            for phase, mask in [('rapid', rapid), ('other', ~rapid)]:
                phases.append({'seed': seed, 'case': case, 'model': model, 'phase': phase, 'points': int(mask.sum()),
                               'rmse': float(np.sqrt(np.mean(error[mask]**2))), 'bias': float(error[mask].mean()), 'sse': float(np.sum(error[mask]**2))})
        ge, de = frame.gain_error.to_numpy(), frame.difficult_error.to_numpy()
        extra_sse = de**2 - ge**2
        assert extra_sse.sum() > 0, (seed, case)
        for quantile in [.70, .75, .80]:
            cutoff = float(np.quantile(absolute_change, quantile))
            mask = (absolute_change >= cutoff) & (absolute_change > 0)
            sensitivity.append({'seed': seed, 'case': case, 'change_quantile': quantile, 'threshold': cutoff,
                                'rapid_points': int(mask.sum()), 'rapid_point_percent': 100*float(mask.mean()),
                                'gain_rapid_rmse': float(np.sqrt(np.mean(ge[mask]**2))),
                                'difficult_rapid_rmse': float(np.sqrt(np.mean(de[mask]**2))),
                                'rapid_share_of_additional_sse_percent': 100*float(extra_sse[mask].sum()/extra_sse.sum())})
        gm, dm = metrics[-2], metrics[-1]
        delta_bias = dm['bias_squared'] - gm['bias_squared']
        delta_centered = dm['centered_mse'] - gm['centered_mse']
        delta_mse = float(extra_sse.mean())
        np.testing.assert_allclose(delta_bias + delta_centered, delta_mse, rtol=1e-13, atol=1e-24)
        cases.append({'seed': seed, 'case': case, 'points': len(frame), 'gain_rmse': gm['rmse'], 'difficult_rmse': dm['rmse'],
                      'rmse_change_percent': 100*(dm['rmse']/gm['rmse']-1), 'gain_bias': gm['bias'], 'difficult_bias': dm['bias'],
                      'gain_centered_rmse': gm['centered_rmse'], 'difficult_centered_rmse': dm['centered_rmse'],
                      'bias_squared_change': delta_bias, 'centered_mse_change': delta_centered,
                      'bias_share_of_additional_mse_percent': 100*delta_bias/delta_mse,
                      'dominant_added_mse_component': 'bias_squared' if delta_bias > delta_centered else 'centered_residuals',
                      'gain_p90_range_ratio': gm['trajectory_p90_range_ratio'], 'difficult_p90_range_ratio': dm['trajectory_p90_range_ratio'],
                      'gain_correlation': gm['trajectory_correlation'], 'difficult_correlation': dm['trajectory_correlation'],
                      'gain_temporal_direction_percent': gm['temporal_direction_agreement_percent'],
                      'difficult_temporal_direction_percent': dm['temporal_direction_agreement_percent'],
                      'rapid_threshold_q75': threshold, 'rapid_point_percent': 100*float(rapid.mean()),
                      'rapid_share_of_additional_sse_percent': 100*float(extra_sse[rapid].sum()/extra_sse.sum())})
        cached.append((seed, case, time, observed, predictions, rapid, absolute_change, cases[-1], interval/60))
    metrics_df, cases_df = pd.DataFrame(metrics), pd.DataFrame(cases)
    pd.concat(series, ignore_index=True).to_csv(OUT/'aligned_time_series.csv', index=False)
    metrics_df.to_csv(OUT/'per_model_diagnostics.csv', index=False)
    cases_df.to_csv(OUT/'case_diagnosis_summary.csv', index=False)
    pd.DataFrame(phases).to_csv(OUT/'rapid_and_other_metrics.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'rapid_threshold_sensitivity.csv', index=False)
    print(cases_df.to_string(index=False), flush=True)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    overview, overview_axes = plt.subplots(5, 2, figsize=(12, 13.5), sharex=True)
    for row_index, (seed, case, time, truth, predictions, rapid, change, diagnostic, dt) in enumerate(cached):
        detailed, axes = plt.subplots(3, 1, figsize=(10.4, 8.5), sharex=True, gridspec_kw={'height_ratios': [2, 1.3, 1]})
        for axis in [axes[0], overview_axes[row_index, 0]]:
            shade(axis, time, rapid, dt)
            axis.plot(time, truth, color=COLORS['Truth'], linewidth=1.6, label='Truth')
            for model in predictions:
                axis.plot(time, predictions[model], color=COLORS[model], linestyle='--' if model=='Difficult-GradNorm' else '-', linewidth=1.25, label=model)
            axis.set_ylabel(r'SO$_2$ (original scale)')
            style(axis)
            axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        for axis in [axes[1], overview_axes[row_index, 1]]:
            shade(axis, time, rapid, dt)
            axis.axhline(0, color='#555555', linewidth=.8)
            for model, values in predictions.items():
                axis.plot(time, values-truth, color=COLORS[model], linestyle='--' if model=='Difficult-GradNorm' else '-', linewidth=1.15)
            axis.set_ylabel('Prediction - truth')
            style(axis)
            axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        shade(axes[2], time, rapid, dt)
        axes[2].plot(time, change, color='#666666', linewidth=1)
        axes[2].axhline(diagnostic['rapid_threshold_q75'], color='#999999', linestyle=':', linewidth=1, label='75th percentile')
        axes[2].set_ylabel(r'$|\Delta$ observed SO$_2|$')
        axes[2].legend(loc='upper right', frameon=False, fontsize=8)
        axes[2].set_xlabel('Nominal time since prediction start (h; 10-min samples)')
        style(axes[2])
        axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.20), frameon=False)
        detailed.suptitle(f'Seed {seed}, {case}: SO$_2$ prediction diagnosis', y=.985, fontsize=14, fontweight='bold')
        detailed.text(.5,.935,f"RMSE: Gain {diagnostic['gain_rmse']:.3e}; Difficult-GradNorm {diagnostic['difficult_rmse']:.3e} ({diagnostic['rmse_change_percent']:+.2f}%)",ha='center',fontsize=10)
        detailed.text(.5,.018,'Gray shading: observed changes in the top quartile within this case. Bias correction is diagnostic only.',ha='center',fontsize=8.5)
        detailed.subplots_adjust(top=.84,bottom=.10,left=.11,right=.98,hspace=.24)
        for ext in ['png','pdf']: detailed.savefig(figures/f'seed{seed}_{case}_so2_diagnosis.{ext}',dpi=300,bbox_inches='tight')
        plt.close(detailed)
        overview_axes[row_index,0].set_title(f"Seed {seed}, {case}   RMSE {diagnostic['rmse_change_percent']:+.2f}%",loc='left',fontweight='bold')
        overview_axes[row_index,1].set_title('Residuals',loc='left')
    for axis in overview_axes[-1]: axis.set_xlabel('Nominal time since prediction start (h)')
    handles,labels=overview_axes[0,0].get_legend_handles_labels()
    overview.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.968),frameon=False)
    overview.suptitle('SO$_2$ failure cases: predictions and residuals',fontsize=15,fontweight='bold',y=.995)
    overview.text(.5,.014,'Shading: largest 25% observed changes within each case. Each row uses its own SO$_2$ scale.',ha='center',fontsize=9)
    overview.subplots_adjust(top=.92,bottom=.055,left=.09,right=.985,hspace=.36,wspace=.25)
    for ext in ['png','pdf']: overview.savefig(figures/f'five_case_so2_overview.{ext}',dpi=300,bbox_inches='tight')
    plt.close(overview)
    write_report()
    for name,digest in protected.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    validation={'passed':True,'cases':[{'seed':s,'case':c} for s,c in CASES],'aligned_points':sum(len(x) for x in series),
                'official_rmse_values_verified':10,'mse_decomposition_verified':True,'rapid_threshold_quantile':.75,
                'sensitivity_quantiles':[.70,.75,.80],'direction_threshold':'0.1% of observed P95-P5 range, min 1e-12',
                'direction_metric_is_kci':False,'bias_correction_uses_test_truth':True,'model_training_performed':False,
                'protected_sha256':protected,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'visual_review':'pending'}
    (OUT/'validation_and_provenance.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print('PASS: 10 official RMSE values and bias/centered-error decompositions; original files preserved.',flush=True)


if __name__=='__main__':
    main()
