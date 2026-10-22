# Difficult-GradNorm ratio=0.01：預測與 KCI 評估

本次評估 seed42、無 SS loss 的 `pgin_gain005_ratio0p01_seed42_difficult_group_gradnorm_no_ss`。確認訓練紀錄的 target ratio 為 **0.01**；比較基準是原本 target ratio=0.015 的 Difficult-GradNorm，並非一般 Gain。評估直接讀取訓練好的權重，沒有更新模型。

## RMSE 比較

以下先逐案例計算 `100 × (新 RMSE / 原版 RMSE − 1)`，再對案例等權平均；負值代表改善。Internal 使用四個 test split（R5、R5-1、R5-2、R5-6），external 使用 R5-19～R5-26。H1、H18 分別是第 1、18 步預測；TS 是原生 rolling time-series 評估。數字不是跨案例合併 RMSE，也不是多 seed 統計。

| 資料 | 品質 | 評估 | 平均 RMSE 變化 | 改善案例 |
| --- | --- | --- | --- | --- |
| Internal test | H2S | H1 | -9.23% | 4/4 |
| Internal test | H2S | H18 | -10.16% | 4/4 |
| Internal test | H2S | TS | -10.95% | 4/4 |
| Internal test | SO2 | H1 | -4.22% | 3/4 |
| Internal test | SO2 | H18 | +0.97% | 2/4 |
| Internal test | SO2 | TS | +3.86% | 2/4 |
| External | H2S | H1 | +2.76% | 4/8 |
| External | H2S | H18 | +2.99% | 3/8 |
| External | H2S | TS | +6.45% | 2/8 |
| External | SO2 | H1 | -8.99% | 5/8 |
| External | SO2 | H18 | -9.13% | 6/8 |
| External | SO2 | TS | -7.53% | 6/8 |

本次最符合原先目標的是 **external SO₂**：H1、H18、TS 平均分別降低 **8.99%、9.13%、7.53%**，其中 TS 有 6/8 案例改善；代價是 external H₂S TS 平均增加 **6.45%**。Internal H₂S 三種評估均改善；internal SO₂ 的 H1 改善，但 H18、TS 的案例平均略退步。因此，這一個 seed 的結果支持降低 target ratio 有助於部分 SO₂ 外部案例，尚不能認定兩個品質輸出全面改善。

External TS 逐案例變化：

| 案例 | H₂S TS | SO₂ TS |
| --- | --- | --- |
| R5-19 | +19.79% | +9.06% |
| R5-20 | -3.74% | -18.57% |
| R5-21 | -1.82% | -9.24% |
| R5-22 | +21.48% | -10.37% |
| R5-23 | +8.62% | +2.23% |
| R5-24 | +1.71% | -28.94% |
| R5-25 | +4.64% | -1.10% |
| R5-26 | +0.95% | -3.29% |

原始數值及完整 22 個資料集的 H1／H18／TS 比較見 [rmse_comparison_vs_ratio0p015.csv](rmse_comparison_vs_ratio0p015.csv)，摘要見 [rmse_comparison_summary.csv](rmse_comparison_summary.csv)。各案例資料夾保存 `prediction_results.csv`、48 個輸出的 `evaluation_metrics.csv` 及兩個品質輸出 × 18 步的 `horizon_metrics_t1_to_t18.csv`。

## KCI 比較

使用與原版相同的 held-out 探測點及 ANN teacher：in-range **4,385** 點，Soft-OOD **2,692** 點。正負擾動各取 18 個 post-step，四個 gain pairs 共 **1,019,088** 筆比較。Teacher gain 的絕對值至少 1e-5 才有效；KCI 以有效比較為分母，要求嚴格同號，動態 gain 為零視為不一致。

| 分布 | 原版 KCI | 本次 KCI | 變化（百分點） | Coverage |
| --- | --- | --- | --- | --- |
| in_range | 98.816% | 98.510% | -0.305 | 87.05% |
| soft_ood | 96.657% | 96.694% | +0.037 | 79.62% |

| 分布 | Gain pair | 原版 KCI | 本次 KCI | 變化（百分點） |
| --- | --- | --- | --- | --- |
| in_range | air2→H2S | 98.599% | 98.209% | -0.391 |
| in_range | T2→H2S | 97.704% | 97.477% | -0.227 |
| in_range | air2→SO2 | 99.760% | 99.680% | -0.080 |
| in_range | T2→SO2 | 99.190% | 98.591% | -0.599 |
| soft_ood | air2→H2S | 95.164% | 95.146% | -0.018 |
| soft_ood | T2→H2S | 96.506% | 96.650% | +0.144 |
| soft_ood | air2→SO2 | 97.954% | 98.054% | +0.100 |
| soft_ood | T2→SO2 | 96.930% | 96.808% | -0.122 |

整體 Soft-OOD KCI 大致維持；分組仍有取捨。Soft-OOD air2→H₂S 的 teacher 正組由 87.85% 升至 90.92%，負組由 97.05% 降至 96.23%；T2→SO₂ 正組由 98.38% 降至 97.51%，負組由 95.35% 升至 96.04%。各符號組的樣本數與 KCI 保存於 [kci_by_teacher_sign.csv](../gain_eval_all18/difficult_gradnorm_ratio0p01_no_ss_seed42/kci_by_teacher_sign.csv)。air2→SO₂ 沒有有效 teacher 負組，該組為 N/A，不能當成 100%。

KCI 表示對 ANN teacher 的方向一致性；本次沒有加入物理模擬或閉迴路控制驗證。Coverage 是 teacher 有效比較的比例，與 KCI 一起報告，不能把被排除的近零 gain 視為正確。

完整 KCI：[gain_evaluation_detail.csv](../gain_eval_all18/difficult_gradnorm_ratio0p01_no_ss_seed42/gain_evaluation_detail.csv)、[gain_evaluation_summary.csv](../gain_eval_all18/difficult_gradnorm_ratio0p01_no_ss_seed42/gain_evaluation_summary.csv)。

## 圖表與驗證

原生案例圖的縱軸可能裁切超出真值範圍的預測。另提供 [comparison_time_series/](comparison_time_series/)：internal test 四案例與 external 八案例，各有 H₂S、SO₂ 的真值／原版 ratio=0.015／本次 ratio=0.01 完整縱軸對照，附每個案例繪圖用 CSV；黑色實線為真值，藍色虛線為原版，橘色實線為本次。這些圖只呈現 TS。

已獨立重算 1,056 個 TS RMSE 與 88 個 H1／H18 RMSE，核對全量 1,019,088 筆 KCI、8 筆配對摘要、2 筆分布摘要及正負組分母。51 個受保護的原始檔案、訓練結果與權重 SHA-256 均未改變。數值檢查與繪圖範圍檢查通過，並抽查兩張完整範圍對照圖。見 [evaluation_validation.json](evaluation_validation.json)、[prediction_metric_recheck.csv](prediction_metric_recheck.csv) 及 [evaluation_input_manifest.json](evaluation_input_manifest.json)。

本次新增的評估結果保留；臨時核對腳本已清除。尚未新增其他 seed，因此不宣稱跨 seed 穩定性。
