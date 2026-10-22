# Difficult-GradNorm ratio=0.02：預測與 KCI

seed42、無 SS loss；直接載入訓練保存的 checkpoint，沒有更新模型。訓練 history 的 target/base gradient norm 比率已核對為 0.02。history 最後 epoch 為 113，此數字不代表最佳 checkpoint 的 epoch。

## 預測結果

已完成 25 個案例：配置原有的 22 個案例，加上指定的 R5-3／4／5 驗證案例。每個案例資料夾保存 TS 預測、48 個輸出的評估指標，以及兩個品質變數的 t+1 至 t+18 指標。

以下為 0.02 相對 0.01 的 TS RMSE。先計算每個案例的 RMSE，再在分區內等權平均，變化為 `100 × (新平均 / 舊平均 − 1)`；不是 pooled RMSE，也不是逐案例百分比的平均。負值代表改善。

| partition | target | change_of_mean_RMSE_percent | improved_cases | cases |
| --- | --- | --- | --- | --- |
| external_19_26 | B35_H2S | -5.07% | 8 | 8 |
| external_19_26 | B35_SO2 | +12.02% | 2 | 8 |
| external_validation | B35_H2S | -3.58% | 1 | 2 |
| external_validation | B35_SO2 | +27.75% | 1 | 2 |
| internal_test | B35_H2S | +11.67% | 0 | 4 |
| internal_test | B35_SO2 | +7.37% | 2 | 4 |
| internal_validation | B35_H2S | +10.87% | 0 | 3 |
| internal_validation | B35_SO2 | -30.01% | 3 | 3 |

完整四 ratio 的 H1／H18／TS 數值見 [four_ratio_rmse_values.csv](four_ratio_rmse_values.csv)，平均值見 [four_ratio_rmse_summary.csv](four_ratio_rmse_summary.csv)，配對變化見 [rmse_case_changes_vs_existing_ratios.csv](rmse_case_changes_vs_existing_ratios.csv)。

R5-3／4／5 沿用指定的 internal validation 分組；R5-5 包含超出 Air2／T2 core 的操作點。R5-13／14 為 external validation。原 internal test 為 R5、R5-1、R5-2、R5-6 的 test split；external R5-19–26 使用全檔。

## KCI

使用原有 `lhs_test_in_range.xlsx`（4,385 個可用操作點）與 `lhs_test_soft_ood_3.xlsx`（2,692 點），與先前三個 ratio 一致。本次未使用新抽樣 CSV `LHS_air2_t2_in_range_final_test_20261022.csv`。

| model | distribution | KCI_percent | coverage_percent |
| --- | --- | --- | --- |
| Difficult-GradNorm ratio=0.005 | in_range | 96.090% | 87.047% |
| Difficult-GradNorm ratio=0.005 | soft_ood | 94.399% | 79.616% |
| Difficult-GradNorm ratio=0.01 | in_range | 98.510% | 87.047% |
| Difficult-GradNorm ratio=0.01 | soft_ood | 96.694% | 79.616% |
| Difficult-GradNorm ratio=0.015 | in_range | 98.816% | 87.047% |
| Difficult-GradNorm ratio=0.015 | soft_ood | 96.657% | 79.616% |
| Difficult-GradNorm ratio=0.02 | in_range | 98.813% | 87.047% |
| Difficult-GradNorm ratio=0.02 | soft_ood | 96.969% | 79.616% |

KCI 為 ANN teacher 的方向一致性；每個操作點、四 gain pairs、正負擾動與 post-step 1–18 個別判斷，不先平均動態 gain。有效 teacher 門檻為絕對值 ≥ 1e-5，零動態 gain 計為不一致。正負 teacher 組數值見 [kci_by_teacher_sign.csv](../gain_eval_all18/difficult_gradnorm_ratio0p02_no_ss_seed42/kci_by_teacher_sign.csv)。

## 圖表與驗證

[七模型比較圖與繪圖資料](../visualization/gradnorm_ratio0p005_0p01_0p015_0p02_seed42/README.md)。新增 0.02 使用橘紅色 X 與實線，先前圖檔保留。

獨立重算 25 個案例的 1,200 筆 TS RMSE 與 100 筆 H1／H18 RMSE；逐筆重算 KCI 有效性、方向、confusion counts、coverage 及整體值，檢查 probe／方向／step／pair 無重複或缺漏。見 [evaluation_validation.json](evaluation_validation.json)、[prediction_metric_recheck.csv](prediction_metric_recheck.csv)。來源、checkpoint 與既有結果的 SHA-256 核對見 [evaluation_input_manifest.json](evaluation_input_manifest.json)。

驗證通過：1,019,088 筆 KCI 診斷重新聚合一致，287 份受保護檔案未改變。獨立 GPU 預測重算與原指標最大絕對差為 6.40e-10；H1／H18 使用 `rtol=1e-5, atol=1e-9` 容許 float32 重播差異，容許範圍與實際差值均已記錄。
