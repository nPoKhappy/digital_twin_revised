# 五個 Gain 超參數 validation 比較

權重：0.05、2.5、25、250、2500；seed42。五個 Gain 與五個 SS-only 使用相同分析、檔案結構及產圖程式。

**內插：R5-3、R5-4、R5-5；外插：R5-27、R5-28、R5-29。這兩組資料夾都是 validation，比三模型 test 使用不同案例。**

## 內插／外插比較圖

[內外插雙面板 PNG](regional_kci_vs_avgrelrmse.png)／[PDF](regional_kci_vs_avgrelrmse.pdf)／[灰階](regional_kci_vs_avgrelrmse_grayscale.png)／[10 個繪圖座標](regional_plot_coordinates.csv)

X 是對應穩態 validation 分布的 ANN-reference micro KCI；Y 是該區域內 18 項同案例 RMSE 比值的幾何平均。No-Gain=1；同一權重在兩組圖使用相同顏色與標記，各面板座標範圍依資料調整。

## 一致的計算定義

先以同案例、同目標、同 H1/H18/TS 的模型 RMSE 除以 No-Gain RMSE。內外各 18 項分別取幾何平均 I、E；整體 AvgRelRMSE=sqrt(I×E)，內外各 50%。沒有先平均原始 RMSE 再計算比值。

整體 KCI 合併兩個穩態 validation 分布的正確數／有效比較數；區域圖使用各自的計數。KCI 並不採內外各 50%，且不是從 R5-27/28/29 動態案例計算。

## 整體與區域分數

| 權重 | 內部 AvgRelRMSE | 外部 AvgRelRMSE | 整體 AvgRelRMSE | ANN-reference KCI (%) |
|---:|---:|---:|---:|---:|
| 250 | 0.954419897 | 0.953090166 | 0.953754800 | 98.941025 |
| 2.5 | 1.009738334 | 0.957612148 | 0.983329901 | 94.836404 |
| 0.05 | 0.998561304 | 0.991157142 | 0.994852335 | 88.665333 |
| 25 | 1.088158448 | 1.117902741 | 1.102930329 | 97.547127 |
| 2500 | 1.176007087 | 1.157372435 | 1.166652556 | 99.026582 |

## H1／H18／TS：內外各 50%

| 權重 | H1 | H18 | TS |
|---:|---:|---:|---:|
| 0.05 | 0.993893506 | 0.996225115 | 0.994439877 |
| 2.5 | 0.984748317 | 0.984777795 | 0.980469846 |
| 25 | 1.273508350 | 1.057412219 | 0.996318322 |
| 250 | 1.047247736 | 0.941668929 | 0.879756612 |
| 2500 | 1.439820277 | 1.096863675 | 1.005457390 |

## 共用圖表與檔案

- [整體 KCI × AvgRelRMSE](overall_kci_vs_avgrelrmse.png)／[PDF](overall_kci_vs_avgrelrmse.pdf)／[座標](overall_plot_coordinates.csv)
- [內外插雙面板](regional_kci_vs_avgrelrmse.png)／[PDF](regional_kci_vs_avgrelrmse.pdf)／[座標](regional_plot_coordinates.csv)
- [H1/H18/TS × 內外插六面板](horizon_kci_vs_avgrelrmse.png)／[PDF](horizon_kci_vs_avgrelrmse.pdf)
- [原單位 RMSE 四面板，含 No-Gain](validation_rmse_summary.png)／[PDF](validation_rmse_summary.pdf)／[CSV](rmse_summary_with_no_gain.csv)
- [整體分數](avgrelrmse_overall.csv)／[區域分數](avgrelrmse_by_region.csv)／[H1/H18/TS 分數](avgrelrmse_by_evaluation.csv)
- [區域 × H1/H18/TS](avgrelrmse_by_region_evaluation.csv)／[180 個逐案例比值與原始 RMSE](relative_rmse_components.csv)
- [216 項指標，含 No-Gain](validation_metrics.csv)／[1296 項 H1～H18 指標](horizon_metrics.csv)／[1368 項 RMSE 重算](RMSE_recheck.csv)
- [資料來源](data_audit.csv)／[內插重現核對](internal_metrics_recheck.csv)／[預測陣列](predictions)
- [KCI 計數](kci_pair_counts.csv)／[計算協定](metric_protocol.json)／[評估核對](evaluation_validation.json)／[圖表核對](plot_validation.json)

目標權重欄位為 `gain_weight`；共用繪圖座標統一使用 `loss_weight`。Gain 與 SS-only 是不同訓練目標，因此數值與排名可以不同。

## 重畫與重跑

僅重畫（不重跑模型）：

```powershell
.\.venv\Scripts\python.exe results/visualization/gain005_gain2p5_gain25_gain250_gain2500_casewise_region50_validation_seed42/rebuild_figures.py
```

完整重跑兩組 validation：

```powershell
.\.venv\Scripts\python.exe scripts/prediction/validate_hyperparameters.py
```

H1/H18 每個起點重新給 18 點真實歷史，再遞迴至第 1／18 步；TS 僅最初 18 點使用真實歷史，其後持續回填預測。皆使用已知未來 decoder inputs。R5-5 沿用內部分類，部分操作點超出 core。

KCI 沿用已核對來源的穩態 validation 結果。單一 seed42、相關軌跡不是獨立重複實驗；未新增誤差棒。ANN-reference KCI 不代表物理正確性或閉迴路控制驗證。
