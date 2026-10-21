# Seed43 四模型預測與 KCI 評估

日期：2026-10-21。四個模型均使用使用者已訓練完成的 seed43 checkpoint，本次沒有重新訓練或修改模型／演算法。

## 評估範圍

- 動態資料：R5、R5-1、R5-2、R5-6 的 test split，以及完整 R5-19～R5-26，共 12 組資料、四模型、48 次推論。沒有包含 train／valid split 或額外的 R5-11～R5-14 案例。
- 沿用 `scripts/prediction/predict.py` 的 `run_prediction`、horizon 分析及 grouped plotting 函式，輸出 H1、H18 與連續時序 TS 指標。
- KCI 沿用 `scripts/evaluation/evaluate_gain_in_out_distribution.py` 的原始計算函式，In-range 完整 4385 個操作點、Soft-OOD 完整 2692 個操作點，沒有抽樣。每點包含四個 target/MV 配對、正負擾動與全部 18 個 post-step 比較。
- 四個模型的結構、時間窗口、動態訓練檔案及標準化統計已核對一致，各方法的穩態訓練資料來源沿用原程式。Gain＋SS 使用 0.01 的 all-48-output SS；Difficult-GradNorm 使用原 no-SS 方法。本次不包含品質 SS=0.001 試驗。

| 模型 | 訓練紀錄 epochs |
| --- | --- |
| No-Gain | 70 |
| Gain | 70 |
| Gain + SS=0.01 | 76 |
| Difficult-GradNorm | 71 |

## 整體 all18 KCI

| 模型 | In-range all18 KCI | Soft-OOD all18 KCI |
| --- | --- | --- |
| No-Gain | 90.9202% | 85.9838% |
| Gain | 91.6499% | 86.6564% |
| Gain + SS=0.01 | 92.7696% | 87.3369% |
| Difficult-GradNorm | 98.5884% | 95.7272% |

各模型的有效 coverage 相同：In-range 87.0468%，Soft-OOD 79.6155%。KCI 分母只包含有效 teacher gain 比較；零動態 gain 保留在有效分母並視為不一致。

## 外插 SO₂ 連續時序 RMSE

下列為原始濃度尺度的 RMSE。變化率 = `(Difficult / Gain - 1) × 100%`，負值代表降低。

| 案例 | No-Gain | Gain | Gain + SS=0.01 | Difficult-GradNorm | Difficult 相對 Gain 變化 |
| --- | --- | --- | --- | --- | --- |
| R5-19 | 0.000128582 | 0.000128407 | 0.000139259 | 0.00011998 | -6.56% |
| R5-20 | 0.0032195 | 0.00314226 | 0.00324779 | 0.00190899 | -39.25% |
| R5-21 | 0.00137261 | 0.00135373 | 0.00131392 | 0.00100185 | -25.99% |
| R5-22 | 0.000617345 | 0.000633833 | 0.000477971 | 0.00053342 | -15.84% |
| R5-23 | 0.000565888 | 0.000567836 | 0.000733693 | 0.000531956 | -6.32% |
| R5-24 | 0.00245115 | 0.00232167 | 0.00416857 | 0.001703 | -26.65% |
| R5-25 | 0.000360291 | 0.00036035 | 0.000311506 | 0.000358709 | -0.46% |
| R5-26 | 0.005967 | 0.00585974 | 0.00459304 | 0.00581885 | -0.70% |

此 seed43 結果中，Difficult-GradNorm 的 SO₂ TS RMSE 在八個外插案例均低於一般 Gain，各案例百分比變化的中位數為 -11.20%。這是案例百分比變化摘要，不是將八組樣本合併計算的 RMSE，也不是統計顯著性檢定。R5-25／R5-26 的相對差距較小。

## Difficult-GradNorm 相對一般 Gain 的改善案例數

| 資料 | 目標 | H1 改善案例數 | H18 改善案例數 | TS 改善案例數 |
| --- | --- | --- | --- | --- |
| 內部 test | B35_H2S | 2／4 | 3／4 | 3／4 |
| 內部 test | B35_SO2 | 1／4 | 2／4 | 2／4 |
| 外插 | B35_H2S | 7／8 | 7／8 | 7／8 |
| 外插 | B35_SO2 | 7／8 | 7／8 | 8／8 |

這些結果按目標、horizon 與資料分布分開報告；不能將外插 SO₂ 的優勢推廣成所有內部案例、所有目標或所有 horizon 均較好。

## Soft-OOD 各配對 KCI

| 模型 | air2 → H₂S | T2 → H₂S | air2 → SO₂ | T2 → SO₂ |
| --- | --- | --- | --- | --- |
| No-Gain | 76.6996% | 87.9276% | 95.1734% | 82.4753% |
| Gain | 76.6316% | 88.5023% | 95.3449% | 84.7117% |
| Gain + SS=0.01 | 70.6369% | 92.8651% | 94.7503% | 89.4846% |
| Difficult-GradNorm | 93.2996% | 96.1549% | 97.2333% | 96.0128% |

## air2 → H₂S 的 teacher 正號群組

此群組用來檢查整體 KCI 是否掩蓋少數符號表現。分母為該 teacher 符號的全部有效比較，包含零動態 gain。

| 模型 | In-range teacher 正號 KCI | Soft-OOD teacher 正號 KCI |
| --- | --- | --- |
| No-Gain | 19.6177% | 31.0606% |
| Gain | 19.7206% | 31.1736% |
| Gain + SS=0.01 | 1.8627% | 12.1677% |
| Difficult-GradNorm | 90.5187% | 89.2943% |

KCI 是相對 ANN teacher 的符號一致率；它與動態案例 RMSE 屬於不同測試，亦不等於物理模擬或閉迴路控制驗證。這裡是單一 seed43 的結果，不能取代跨種子的統計比較。

## 檔案與驗證

- 各模型／案例的預測 CSV、TS 評估、H1～H18 指標與圖表已歸回 `results/<experiment>/`，路徑見下方表格。模型與標準化統計仍使用原始實驗名稱。
- `four_model_rmse_values.csv`：288 筆 H1／H18／TS 的 H₂S／SO₂ RMSE 與 R²。
- [gain_evaluation_summary.csv](../../gain_eval_all18/four_model_comparison_seed43/gain_evaluation_summary.csv)：8 筆模型／分布的整體 KCI。
- [gain_evaluation_detail.csv](../../gain_eval_all18/four_model_comparison_seed43/gain_evaluation_detail.csv)：32 筆模型／分布／配對的 KCI、coverage、TP/TN/FP/FN/zero gain 與正負符號 KCI。
- [gain_vs_no_gain_comparison.csv](../../gain_eval_all18/four_model_comparison_seed43/gain_vs_no_gain_comparison.csv)：Gain 系列相對 No-Gain 的原評估器比較。
- `rmse_relative_comparison.csv`、`rmse_improvement_summary.csv`：相對 No-Gain／Gain 的逐案例變化與改善案例數。
- `prediction_row_counts.csv`：每案例推論點數；四模型時間索引已核對一致，品質預測皆為有限值。
- `existing_prediction_recheck.json`：原有 No-Gain／Gain 四組內部測試重新推論，RMSE、MAE、R² 與既有結果完全一致，RMSE 差異為零。
- `evaluation_provenance.json`：模型、設定、程式、資料及既有成果的 SHA256。保護清單中的全部原檔均核對未變。

為避免重複產生巨型逐步診斷 CSV，本次沒有另存 evaluator 內部逐點逐步診斷列；完整資料均有參與 KCI 計數。臨時執行程式、快取與日誌於核對後清除，正式評估結果保留。


## 整理後位置（2026-10-21）

| 模型 | 預測與指標 | KCI |
| --- | --- | --- |
| No-Gain | [模型結果](../../transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed43/) | [KCI](../../gain_eval_all18/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed43/) |
| Gain | [模型結果](../../transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed43/) | [KCI](../../gain_eval_all18/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed43/) |
| Gain + SS=0.01 | [模型結果](../../transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss001_seed43/) | [KCI](../../gain_eval_all18/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss001_seed43/) |
| Difficult-GradNorm | [模型結果](../../transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed43_difficult_group_gradnorm_no_ss/) | [KCI](../../gain_eval_all18/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed43_difficult_group_gradnorm_no_ss/) |

[四模型 KCI 總表](../../gain_eval_all18/four_model_comparison_seed43/) · [圖表總覽](../seed43_figures.md)

既有 No-Gain／Gain 的舊彙整結果完整保存在各模型的 `previous_predictions_20261021/`。40 個內容完全相同的內部測試重複檔案已去重，原檔仍在模型資料夾。
