# Gain + SS：兩個權重同步變動

每個設定使用相同數值的 `gain_loss_weight` 和 `ss_loss_weight`：

| Gain weight | SS weight | 檔名中的標記 |
| --- | --- | --- |
| 0.05 | 0.05 | `gain005_ss005` |
| 2.5 | 2.5 | `gain2p5_ss2p5` |
| 25 | 25 | `gain25_ss25` |
| 250 | 250 | `gain250_ss250` |
| 2500 | 2500 | `gain2500_ss2500` |

沿用原 Gain+SS 的資料、模型、seed=42、訓練與驗證切分，以及 `ss_target_scope: all_48_outputs`。SS loss 比較全部 48 個輸出，Gain loss 沿用原本 ANN teacher 的方向限制。這裡列的是 loss 係數；實際兩種 loss 的大小仍依訓練而異。

2026-11-04 已移除舊 Gain+SS 專用 checkpoint、訓練結果及 KCI 評估批次。2026-11-05 已完成這五組新設定的訓練，包括原本同名的 0.05／0.05。

從專案根目錄依序重跑五組：

```powershell
Get-ChildItem configs/pgin/gain_ss -Filter *.yaml | Sort-Object Name | ForEach-Object {
    .\.venv\Scripts\python.exe -m scripts.training.train_pgin_stepwise_gain_ss_loss --config $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $($_.Name)" }
}
```

結果寫至 `results/pgin/gain_ss/<exp_name>/`；權重仍寫至 `saved_models/<exp_name>.pth`。即時訓練圖位於各實驗的 `runtime_visualizations/`，避免多個實驗共用同一張圖。可用 `training.pgin_runtime_plot: false` 關閉即時圖。

訓練後以相同 YAML 執行 `scripts.prediction.predict` 或 `scripts.prediction.predict_step_change`，會從該實驗的新分類位置讀取標準化統計量。舊的混合模型歷史圖保留原有數值，不能視為這次五組新訓練的結果。

五組 validation 分析見 [比較報告](../../../results/visualization/gain_ss_all_weights_casewise_region50_validation_seed42/README.md)。使用相同 frozen checkpoint 重跑完整動態 RMSE 與穩態 KCI（不訓練）：

```powershell
.\.venv\Scripts\python.exe -m scripts.prediction.validate_gain_ss_hyperparameters
```

內插 R5-3／4／5、外插 R5-27／28／29；AvgRelRMSE 內外各 50%。KCI 另使用 `lhs_valid_in_range.xlsx`、`lhs_valid_soft_ood.xlsx` 的全部有效穩態探測點。
