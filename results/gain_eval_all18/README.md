# all18 Gain／KCI 評估結果

每個模型的資料夾使用對應 YAML 檔名（不含 `.yaml`），沿用 seed42 命名方式。各資料夾的 `gain_evaluation_detail.csv` 是兩分布 × 四配對的 8 筆 KCI；`gain_evaluation_summary.csv` 是兩分布的整體 KCI 與 coverage。

## 目前保留的 seed42 結果

| 模型 | 評估資料夾 |
| --- | --- |
| No-Gain | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42/) |
| Gain | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42/) |
| Gain + SS=0.01 | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss001_seed42/) |
| Difficult-GradNorm | [seed42](transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42_difficult_group_gradnorm_no_ss/) |

動態預測及訓練 history 位於各自的 `results/<experiment>/`，不放在本目錄。

## Seed43／44 本機封存

2026-10-21 已將 seed43／44 的配置、模型權重、訓練與預測結果、KCI 評估及圖表複製到桌面的 `seed43_44`，逐檔核對 SHA-256 後，移除原專案中的專屬檔案。

封存沿用專案目錄結構，KCI 資料位於 `seed43_44/results/gain_eval_all18/`。封存根目錄的 `EXPORT_README.md` 提供內容說明，`export_manifest.json` 保存逐檔雜湊清單。桌面副本不在此 Git repository 內。
