# Difficult-GradNorm ratio=0.005：預測與 KCI

本次直接讀取 `pgin_gain005_ratio0p005_seed42_difficult_group_gradnorm_no_ss` 訓練好的 checkpoint。已核對群組紀錄中的 target/base gradient norm 比例為 **0.005**，訓練 history 記錄到第 **97** 個 epoch；這是 history 最後 epoch，不代表最佳 checkpoint 正好在該 epoch。維持 seed42、無 SS loss，評估沒有更新模型。

## 範圍與結果

完成配置中原有 **22** 個資料集，並補上 R5-3、R5-4、R5-5，共 **25** 個案例的 TS 預測及 H1～H18 指標。比較資料包含 0.005、0.01、0.015 三個 ratio。

新增 internal validation 的 R5-3～5：0.005 相較 0.01，H₂S 與 SO₂ 的 TS RMSE 三例均改善。但 external validation 的 R5-13／14，兩品質的案例平均 RMSE 都提高，SO₂ 的退步主要集中在 R5-13。這次結果不支持單純減小 ratio 就能同時改善外部預測與 gain 一致性。

以下以每個分區的案例 RMSE 先取算術平均，再計算 `100 × (平均 RMSE_0.005 / 平均 RMSE_0.01 − 1)`。負值代表 0.005 較好。這不是逐案例百分比變化的平均，也不是 pooled RMSE；「改善案例」則逐案例判定。

| Partition | Target | Cases | RMSE change vs 0.01 | Improved cases |
| --- | --- | --- | --- | --- |
| internal_validation | H2S | 3 | -3.13% | 3/3 |
| internal_validation | SO2 | 3 | -17.63% | 3/3 |
| external_validation | H2S | 2 | +38.21% | 0/2 |
| external_validation | SO2 | 2 | +153.40% | 1/2 |
| internal_test | H2S | 4 | -1.90% | 3/4 |
| internal_test | SO2 | 4 | -7.65% | 3/4 |
| external_19_26 | H2S | 8 | +11.19% | 1/8 |
| external_19_26 | SO2 | 8 | +17.22% | 2/8 |

完整三個 ratio 的 TS 案例平均 RMSE：

| Partition | Target | ratio=0.005 | ratio=0.01 | ratio=0.015 |
| --- | --- | --- | --- | --- |
| internal_validation | H2S | 0.000617324 | 0.000637266 | 0.000708753 |
| internal_validation | SO2 | 0.000232819 | 0.000282634 | 0.00019905 |
| external_validation | H2S | 0.000675726 | 0.000488904 | 0.000456301 |
| external_validation | SO2 | 0.000576739 | 0.000227597 | 0.000254064 |
| internal_test | H2S | 0.000694534 | 0.000707993 | 0.000797961 |
| internal_test | SO2 | 0.000274221 | 0.000296942 | 0.000314001 |
| external_19_26 | H2S | 0.00228646 | 0.00205629 | 0.00199435 |
| external_19_26 | SO2 | 0.00149216 | 0.00127301 | 0.00144118 |

原始數值：[three_ratio_rmse_values.csv](three_ratio_rmse_values.csv)，450 筆（3 ratio × 25 案例 × 2 品質 × 3 評估）。完整 H1／H18／TS 的分區比較見 [rmse_comparison_summary.csv](rmse_comparison_summary.csv)；逐案例比較見 [rmse_comparison_vs_existing_ratios.csv](rmse_comparison_vs_existing_ratios.csv)。兩個 CSV 同時保留「平均 RMSE 的相對變化」與「逐案例百分比變化的平均」所需資訊，兩種彙整不能混用。

各案例資料夾保存 48 個輸出的 `prediction_results.csv`、`evaluation_metrics.csv`，以及兩品質 × 18 步的 `horizon_metrics_t1_to_t18.csv`。H1／H18 以真實歷史逐窗口初始化；TS 則用起始歷史後持續預測回填。

## KCI 與正負組

全量 held-out in-range **4,385** 點、Soft-OOD **2,692** 點，每點四個 gain pairs、正負擾動各 18 步，共 **1,019,088** 筆比較。與 0.01 使用相同 XLSX 來源雜湊，並核對三個 ratio 的有效分母、teacher 正負比例及 coverage 完全一致。

| Distribution | ratio=0.005 KCI | ratio=0.01 KCI | ratio=0.015 KCI |
| --- | --- | --- | --- |
| in_range | 96.090% | 98.510% | 98.816% |
| soft_ood | 94.399% | 96.694% | 96.657% |

Coverage：in-range **87.05%**、Soft-OOD **79.62%**。Teacher gain 絕對值至少 1e-5 才有效；KCI 為有效比較中的嚴格同號比例，動態 gain 為零視為不一致。

**整體 KCI 會掩蓋少數符號組的退步。** air2→H₂S 的 teacher 正組，0.005 的 in-range KCI 只有 **47.03%**，Soft-OOD 為 **70.04%**；0.01 對應為 **91.85%**、**90.92%**。

| Distribution | Valid comparisons | ratio=0.005 | ratio=0.01 | ratio=0.015 |
| --- | --- | --- | --- | --- |
| in_range | 18468 | 47.03% | 91.85% | 92.11% |
| soft_ood | 15048 | 70.04% | 90.92% | 87.85% |

其餘配對與正負組見 [three_ratio_pair_comparison.csv](../gain_eval_all18/difficult_gradnorm_ratio0p005_no_ss_seed42/three_ratio_pair_comparison.csv)、[three_ratio_kci_by_teacher_sign.csv](../gain_eval_all18/difficult_gradnorm_ratio0p005_no_ss_seed42/three_ratio_kci_by_teacher_sign.csv)。air2→SO₂ 沒有有效 teacher 負組，該組 KCI 為 N/A，不能當成 100%。

KCI 反映對 ANN teacher 的方向一致性，與動態 RMSE 使用不同資料；本次未新增物理模擬或控制實驗。這些比較僅涵蓋 seed42。

## 圖表與檔案

已將 0.005 加入 No-Gain、Gain、Gain＋SS=0.01、GradNorm 0.015、GradNorm 0.01 的比較，六模型配色沿用原圖，**0.005 為綠色倒三角／點劃線**。

全部圖表：[README.md](../visualization/gradnorm_ratio0p005_0p01_0p015_seed42/README.md)。包含 internal test、external R5-19～26、指定 internal／external validation 的 RMSE，以及 all18 KCI，均提供 PNG、PDF、繪圖 CSV。原有圖表保留。

R5-3～5 依指定列為 internal validation，其中 R5-5 部分點超出既定 air2／T2 核心範圍；R5-13／14 依指定列為 external validation，它們已有先前評估紀錄。本次維持分組，不修改原有 train／valid／test 切分。

KCI 原始檔位於 [difficult_gradnorm_ratio0p005_no_ss_seed42](../gain_eval_all18/difficult_gradnorm_ratio0p005_no_ss_seed42/README.md)。

## 核對

- 獨立重算 **1,200** 個 TS RMSE 及 **100** 個 H1／H18 RMSE，共 1,300 筆；Horizon 另外使用 encoder／decoder 逐步回填重新推論。
- 從逐探測點 CSV 重算 **1,019,088** 筆 KCI 的有效性、嚴格同號、TP／TN／FP／FN、零動態 gain、正負組及 coverage，並核對 8 筆配對、2 筆分布摘要。
- 原始資料、配置、權重、訓練結果、來源程式與讀取的既有比較資料均以 SHA-256 確認未修改。臨時核對腳本已清除，保留正式預測、報告與核對結果。
- 五張比較圖的資料範圍與圖面檢查通過。

見 [evaluation_validation.json](evaluation_validation.json)、[prediction_metric_recheck.csv](prediction_metric_recheck.csv)、[evaluation_input_manifest.json](evaluation_input_manifest.json) 及 [evaluation_output_hashes.json](evaluation_output_hashes.json)。
