# all18 Gain／KCI 評估結果

每個模型的資料夾使用對應 YAML 檔名（不含 `.yaml`），沿用 seed42 命名方式。各資料夾的 `gain_evaluation_detail.csv` 是兩分布 × 四配對的 8 筆 KCI；`gain_evaluation_summary.csv` 是兩分布的整體 KCI 與 coverage。

| 模型 | Seed42 | Seed43 |
| --- | --- | --- |
| No-Gain | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42/) | [seed43](transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed43/) |
| Gain | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42/) | [seed43](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed43/) |
| Gain + SS=0.01 | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss001_seed42/) | [seed43](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss001_seed43/) |
| Difficult-GradNorm | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42_difficult_group_gradnorm_no_ss/) | [seed43](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed43_difficult_group_gradnorm_no_ss/) |

## Seed43 四模型總表

- [四模型 KCI 明細](four_model_comparison_seed43/gain_evaluation_detail.csv)
- [四模型整體 KCI](four_model_comparison_seed43/gain_evaluation_summary.csv)
- [各 Gain 方法相對 No-Gain](four_model_comparison_seed43/gain_vs_no_gain_comparison.csv)
- [完整 RMSE／KCI 評估報告](../visualization/four_model_rmse_visualization_seed43/evaluation_report.md)

Seed43 使用完整 4,385 個 in-range 與 2,692 個 Soft-OOD 操作點。逐操作點的分組圖資料保存在 [probe_accuracy_seed43.csv.gz](../visualization/gain_pair_difficulty_seed43/probe_accuracy_seed43.csv.gz)；這是每操作點聚合的 36 次比較，與 seed42 的逐步 `gain_probe_diagnostics.csv` 格式不同。

原 seed42 檔案保留。動態預測與訓練 history 位於各自的 `results/<experiment>/`，不放在本目錄。
