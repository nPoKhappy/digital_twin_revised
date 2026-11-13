# Seed 42：21 個模型的 validation 比較

本次使用 21 個新訓練的 seed 42 模型，每個模型分別載入自己的 checkpoint 與 scaler；不更新模型權重。兩張子圖各有 21 點，共 **42 個區域座標**，不是 42 個模型。

- 圖：[PNG](regional_kci_vs_avgrelrmse.png)、[PDF](regional_kci_vs_avgrelrmse.pdf)。圖中文字有部分重疊，精確數值和模型名稱請查閱 [regional_plot_coordinates.csv](regional_plot_coordinates.csv)。此原圖兩個子圖共用 y 軸、各自決定 x 軸範圍。
- 左圖：動態 validation 案例 R5-3、R5-4、R5-5；KCI 使用另一份穩態資料 `data/Claus_steady_state/lhs_valid_in_range.xlsx`。
- 右圖：動態 validation 案例 R5-27、R5-28、R5-29；KCI 使用另一份穩態資料 `data/Claus_steady_state/lhs_valid_soft_ood.xlsx`。
- Y 軸：每個案例、輸出與評估方式各自計算模型 RMSE / 相同條件的 seed 42 No-Gain RMSE；每區對 3 案例 × 2 輸出 × H1/H18/TS，共 18 個比率取幾何平均。No-Gain 為 1。逐項結果在 [relative_rmse_components.csv](relative_rmse_components.csv)。
- X 軸：ANN-reference micro KCI。以 ANN 梯度絕對值至少 1e-5 的有效資料，比對動態模型有限步階增益的正負號；合併 18 個 step、正負兩個擾動方向和 2 輸出 × 2 MV。動態增益為零算不一致。各模型使用自己的 scaler 決定 0.5 標準差物理擾動。各輸出與 MV 的計數在 [kci_components.csv](kci_components.csv)。

注意：兩個軸使用不同的動態與穩態 validation 資料。R5-5 分在左圖，但其 1,440 筆原始時間資料中有 33.19% 落在既有 air2/T2 核心範圍外。KCI 是相對 ANN 的增益方向一致性，不能單獨證明 Aspen 真實增益方向或閉迴路表現。這是既有 validation 案例，不是新的盲測；單一 seed 42 也不能代表 63 個模型或估計 seed 間變異。

## 檔案與檢查

`model_data/<alias>/` 保存六個動態預測 NPZ、兩個穩態 gain 矩陣 NPZ、逐模型指標及計數；`complete.json` 保存快取雜湊。每個 RMSE 都以預測值重算，KCI 計數則以 NumPy 獨立重算。各 checkpoint 和 scaler 已與完成的 seed 42 批次紀錄核對，新 ANN teacher 與 scaler 也與該紀錄一致。`run_protocol.json` 保存受保護來源的雜湊，`evaluation_validation.json` 記錄完成檢查。

在專案根目錄重現：

```powershell
.\.venv\Scripts\python.exe -m scripts.prediction.validate_retrained_21_seed42
```

既有快取會先檢查雜湊；來源與原紀錄不同時會停止，而非混用舊結果。
