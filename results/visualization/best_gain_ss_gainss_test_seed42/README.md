# 三家族最佳模型與 No-Gain：Test 比較

依各家族 validation AvgRelRMSE 最低值固定選定 Gain=250、SS-only=2.5、Gain+SS=(2.5,2.5)，seed42；test 分數未參與選擇。保留 No-Gain 基準。

動態核心：R5／R5-1／R5-2／R5-6 的 test split；動態近外插：R5-19～26。穩態 KCI：lhs_test_in_range.xlsx（4385點）、lhs_test_soft_ood.xlsx（2692點，與先前 _3.xlsx 內容相同）。

## 五張表格與第六張內外插圖

- [01_overall_kci PNG](01_overall_kci.png)／[PDF](01_overall_kci.pdf)
- [02_pair_kci PNG](02_pair_kci.png)／[PDF](02_pair_kci.pdf)
- [03_internal_ts_rmse PNG](03_internal_ts_rmse.png)／[PDF](03_internal_ts_rmse.pdf)
- [04_external_so2_rmse PNG](04_external_so2_rmse.png)／[PDF](04_external_so2_rmse.pdf)
- [05_external_h2s_rmse PNG](05_external_h2s_rmse.png)／[PDF](05_external_h2s_rmse.pdf)
- [06_regional_kci_vs_avgrelrmse PNG](06_regional_kci_vs_avgrelrmse.png)／[PDF](06_regional_kci_vs_avgrelrmse.pdf)

## 資料與計算定義

Δ (%) = 100 × (RMSE_model / RMSE_NoGain − 1)，負值表示改善。No-Gain 原單位 RMSE 乘以1000，表頭標示10⁻³ kmol/kmol；百分比以未四捨五入的 RMSE 計算。

KCI 整體以四個增益配對的 correct／valid 合併，不是四個百分比的算術平均。零動態增益視為錯誤；有效 ANN gain 門檻1e-5，兩方向各0.5個訓練標準差，warmup35步後逐點判斷全部18步。

區域 AvgRelRMSE 先計算同案例、同目標、同H1/H18/TS的模型／No-Gain比值。核心24項、近外插48項分別取幾何平均；整體sqrt(I×E)，內外各50%。第六張圖呈現區域分數，No-Gain有實際KCI座標上的點及y=1參考線。

- [選模來源](selected_models.csv)／[來源與保護雜湊](run_protocol.json)
- [整體KCI](overall_kci.csv)／[配對KCI與計數](gain_pair_kci.csv)
- [核心H1/H18/TS表](internal_horizon_rmse_table.csv)／[外插SO2 H1/H18/TS表](external_so2_horizon_rmse_table.csv)／[外插H2S H1/H18/TS表](external_h2s_horizon_rmse_table.csv)
- [可複製編輯的合併儲存格表格（HTML）](grouped_rmse_tables.html)
- [原核心TS摘要](internal_ts_rmse_table.csv)／[原外插SO2 H18/TS摘要](external_so2_rmse_table.csv)／[原外插H2S H18/TS摘要](external_h2s_rmse_table.csv)
- [288項指標](test_metrics.csv)／[1728項horizon指標](horizon_metrics.csv)／[1824項RMSE重核](RMSE_recheck.csv)
- [與No-Gain逐項比較](paired_case_comparison.csv)／[區域繪圖座標](regional_plot_coordinates.csv)／[原始預測](predictions/)
- [整體AvgRelRMSE](avgrelrmse_overall.csv)／[資料案例](data_audit.csv)／[數值核對](evaluation_validation.json)／[圖表核對](plot_validation.json)

第3、4、5張已改為每個案例分列H1/H18/TS的表格，案例欄在三列中置中。綠色代表RMSE低於No-Gain，紅色代表高於No-Gain；每列最低RMSE（包含No-Gain）以粗體標示。第3張原檔名保留internal_ts以維持既有連結，內容現已包含兩個品質的全部三種時域。

本次只評估固定checkpoint。這些是既有test案例，不宣稱未曾檢視的新盲測；單一seed、相關horizon不是獨立重複實驗。KCI是ANN-reference方向一致性，不代表物理增益正確性或閉迴路性能。

完整重跑：`python -m scripts.prediction.compare_best_models_test`。只重畫：`python results/visualization/best_gain_ss_gainss_test_seed42/rebuild_figures.py`。
