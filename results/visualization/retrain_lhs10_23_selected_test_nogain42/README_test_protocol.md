# Validation 選模與三 seed test 比較

本次以已保存的 validation 結果選出四個模型家族各一組超參數，再分別評估 seed 42、43、44；另外評估 No-Gain 的三個 seed，共 15 個已訓練模型。選模在這次 test 計算前完成，未依本次 test 結果挑選。

| 家族 | 選用超參數 | seed |
|---|---|---|
| No-Gain | 固定參考及其他 seed | 42、43、44 |
| Gain | 25 | 42、43、44 |
| SS-only | 2.5 | 42、43、44 |
| Gain + SS | 25 / 25 | 42、43、44 |
| Difficult-GradNorm | ρ = 0.015 | 42、43、44 |

## Validation 選模規則

每個家族有五組候選設定。對每個 seed，以 `sqrt(內插 AvgRelRMSE × 外插 AvgRelRMSE)` 綜合內外插，再計算三個 seed 的算術平均作為誤差分數；KCI 則先對內插及外插各給 50% 權重，再取三個 seed 的算術平均。這裡的 AvgRelRMSE 以 validation 的 No-Gain seed 42 為共同分母。

在**各家族內**，分別對五組候選的誤差與 KCI 做 min-max 正規化，以誤差越低、KCI 越高為理想點，選取到理想點的歐氏距離最小者；兩個正規化指標的權重相同。若距離相同，依序選誤差較低及 alias 字母順序。這是明確的折衷規則，並不表示選出的設定同時擁有該家族最低誤差與最高 KCI，距離也不能拿來跨家族比較。所有候選的分數與來源雜湊見 `validation_selection_candidates.csv` 與 `validation_selection_plan.json`。

## Test 範圍與指標

動態 test 為內插 R5、R5-1、R5-2、R5-6 的 test split，以及外插 R5-19～R5-26 完整軌跡。各模型的 H1、H18、TS RMSE 都除以相同 case、輸出與預測位置的 **test No-Gain seed 42 RMSE**。內插及外插分別對 24 項與 48 項比例取幾何平均；各 seed 的內外插綜合誤差再取 `sqrt(內插 × 外插)`。No-Gain 43、44 因使用相同 seed 42 分母，數值不必為 1。

KCI 使用獨立穩態探點檔案 `lhs_test_in_range.xlsx`、`lhs_test_soft_ood.xlsx`，可用探點分別為 4,385、2,692。比較 ANN 局部增益與動態模型在 ±0.5 個動態標準差擾動後 18 步反應的方向；ANN 局部增益絕對值須至少為 1e-5，動態增益為零計為不一致。KCI 是相對 ANN 參考的方向一致率，不能單憑它證明 Aspen 真實增益方向或閉迴路效能。

這些 test 案例與穩態探點先前已使用過，不能稱為全新的盲測。動態預測採用已記錄的未來 decoder 輸入，因此是條件式預測。三個 seed 共用資料與 ANN；三 seed 的變異描述動態模型訓練變異，而非不同 ANN 或獨立 test 資料的變異。

## 檔案與重現

`selected_models.json` 紀錄凍結的選模與 checkpoint；`run_protocol.json` 紀錄來源雜湊、案例與計算設定；`evaluation_validation.json` 記錄覆核結果。`model_data/` 下保存各模型預測陣列、穩態增益陣列、指標與完成雜湊。`regional_coordinates.csv` 是圖上各點，`relative_rmse_components.csv` 是配對後的 RMSE 項目。

重現選模的程式是 `scripts/analysis/select_retrained_validation_compromise.py`；重算 test 的程式是 `scripts/prediction/compare_retrained_mean_tradeoff_test.py`。下列命令會用已保存的選模計畫，寫入獨立的 `_staging` 目錄；若暫存目錄已有符合協定的完成結果，執行器會核對雜湊後沿用。

```powershell
.\.venv\Scripts\python.exe -u -m scripts.prediction.compare_retrained_mean_tradeoff_test --plan results/visualization/retrain_lhs10_23_selected_test_nogain42/validation_selection_plan.json
```

重算與覆核完成後再將輸出納入正式結果；上述指令本身不會覆蓋正式結果。
