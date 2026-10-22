# 指定驗證案例：五模型比較

本次依指定使用 R5-3／R5-4／R5-5 作 internal validation、R5-13／R5-14 作 external validation，比較 No-Gain、Gain、Gain＋SS=0.01，以及 Difficult-GradNorm ratio=0.015／0.01。沒有重新訓練、挑選新 checkpoint 或修改權重。R5-13／14 原本已有評估紀錄，本次沿用預測並重新核對；不將它們描述成全新未見資料。

## 主要觀察

兩個 GradNorm 版本間，**SO₂ 的 TS RMSE 在 R5-3、R5-4、R5-5、R5-14 都是 0.015 較低，R5-13 則是 0.01 較低**。H₂S 相反，R5-3、R5-4、R5-5、R5-13 是 0.01 較低，R5-14 是 0.015 較低。因此，這批驗證資料呈現品質輸出的取捨，不能宣稱 0.01 全面優於 0.015。

GradNorm 逐案例 TS 比較如下，變化為 `100 × (RMSE_0.01 / RMSE_0.015 − 1)`，負值代表 0.01 較好：

| Case | Target | ratio=0.015 TS RMSE | ratio=0.01 TS RMSE | Change |
| --- | --- | --- | --- | --- |
| R5-3 | H2S | 0.00065114 | 0.000524264 | -19.49% |
| R5-3 | SO2 | 0.00014469 | 0.000230583 | +59.36% |
| R5-4 | H2S | 0.000492429 | 0.000471759 | -4.20% |
| R5-4 | SO2 | 0.000295995 | 0.000387164 | +30.80% |
| R5-5 | H2S | 0.000982689 | 0.000915776 | -6.81% |
| R5-5 | SO2 | 0.000156464 | 0.000230156 | +47.10% |
| R5-13 | H2S | 0.000156339 | 9.98039e-05 | -36.16% |
| R5-13 | SO2 | 0.000377595 | 0.000254265 | -32.66% |
| R5-14 | H2S | 0.000756263 | 0.000878004 | +16.10% |
| R5-14 | SO2 | 0.000130533 | 0.000200928 | +53.93% |

完整 H1／H18／TS 比較見 [ratio0p01_vs_ratio0p015.csv](ratio0p01_vs_ratio0p015.csv)。

## 分區摘要

以下先對每一分區的案例 RMSE 取算術平均，再比較兩個平均值，與執行前 [evaluation_protocol.json](evaluation_protocol.json) 記錄的案例彙整方式一致。H₂S、SO₂ 分開計算，不將原始濃度誤差直接相加。這個表不是 pooled RMSE，也不是跨 seed 統計。

| Partition | Target | Evaluation | ratio=0.015 mean RMSE | ratio=0.01 mean RMSE | Change of mean |
| --- | --- | --- | --- | --- | --- |
| internal_validation | H2S | H1 | 0.00051128 | 0.000469451 | -8.18% |
| internal_validation | H2S | H18 | 0.000721843 | 0.000643582 | -10.84% |
| internal_validation | H2S | TS | 0.000708753 | 0.000637266 | -10.09% |
| internal_validation | SO2 | H1 | 0.000139718 | 0.000167653 | +19.99% |
| internal_validation | SO2 | H18 | 0.000208174 | 0.000263369 | +26.51% |
| internal_validation | SO2 | TS | 0.00019905 | 0.000282634 | +41.99% |
| external_validation | H2S | H1 | 0.00038017 | 0.000425422 | +11.90% |
| external_validation | H2S | H18 | 0.000441856 | 0.000497002 | +12.48% |
| external_validation | H2S | TS | 0.000456301 | 0.000488904 | +7.15% |
| external_validation | SO2 | H1 | 0.000158052 | 0.000201786 | +27.67% |
| external_validation | SO2 | H18 | 0.000346364 | 0.000281227 | -18.81% |
| external_validation | SO2 | TS | 0.000254064 | 0.000227597 | -10.42% |

「平均 RMSE 的相對變化」與「逐案例百分比變化的平均」不同。尤其 external 只有兩例，誤差尺度不同時，兩種平均甚至可能得到相反方向；應同時看逐案例結果，不用彙整方式挑選有利結論。[ratio_change_summary.csv](ratio_change_summary.csv) 保存的是後者，供核對；[ratio_mean_rmse_summary.csv](ratio_mean_rmse_summary.csv) 保存上表所用的前者。

五模型 TS 的案例平均 RMSE：

| Partition | Model | H2S mean TS RMSE | SO2 mean TS RMSE |
| --- | --- | --- | --- |
| external_validation | Difficult-GradNorm ratio=0.01 | 0.000488904 | 0.000227597 |
| external_validation | Difficult-GradNorm ratio=0.015 | 0.000456301 | 0.000254064 |
| external_validation | Gain | 0.000689136 | 0.000587012 |
| external_validation | Gain + SS=0.01 | 0.000720079 | 0.000990765 |
| external_validation | No-Gain | 0.000702192 | 0.000612266 |
| internal_validation | Difficult-GradNorm ratio=0.01 | 0.000637266 | 0.000282634 |
| internal_validation | Difficult-GradNorm ratio=0.015 | 0.000708753 | 0.00019905 |
| internal_validation | Gain | 0.000665186 | 0.000271952 |
| internal_validation | Gain + SS=0.01 | 0.000718752 | 0.000216464 |
| internal_validation | No-Gain | 0.000666144 | 0.000273089 |

## 資料與程序

- 每個案例原始資料 1,440 列，先沿用 10 列中位數下採樣，得到 144 列；H₂S／SO₂ 取 log 後，使用各模型已保存的訓練標準化參數。
- TS 使用前 18 步真實歷史初始化，之後預測回填，共 126 個評估點。H1／H18 使用 109 個真實歷史起點，每個起點往前預測 18 步。
- Internal 三個 Excel 原檔保持不變；新增預測位於 `predictions/no_gain/`、`gain/`、`gain_ss001/`、`dg_ratio0p015/`、`dg_ratio0p01/`。短名稱用來避開 Windows 路徑長度限制，實際來源模型記錄在 `run_configs/`。
- 新分組依指定，未更動既有 test 的配置與結果。R5-5 有部分時段超出既定 air2／T2 核心範圍，圖中以星號標註；R5-13／14 各自超出 air2／T2 核心上限。
- 63 個模型輸入欄位四捨五入至 8 位小數後，這五個案例未找到與現行四個訓練 CSV 完全相同的整列資料。這是數值重複檢查，不是對所有歷史使用情形的保證。範圍與列數見 [data_audit.csv](data_audit.csv)。

## 圖表與驗證

圖表索引：[README.md](../visualization/validation_r5_3_4_5_r5_13_14_seed42/README.md)。四張圖各提供 PNG 與 PDF，包含五模型的 RMSE 折線圖及真值／預測時序疊圖。

已重算五模型 × 五案例 × 48 個輸出的 **1,200 個 TS RMSE**，以及 **100 個 H1／H18 RMSE**，全部通過。Horizon 另外使用 encoder／decoder 逐步回填重新推論，與原生預測函式的指標比較。所有繪圖資料均在縱軸範圍內，四張圖已逐張檢視。來源程式、配置、權重、標準化參數、資料及讀取的既有預測檔案以 SHA-256 核對未變更。

- [five_model_validation_metrics.csv](five_model_validation_metrics.csv)：150 筆主要比較指標。
- [validation_summary.csv](validation_summary.csv)：各分區、模型、品質及評估方式的案例平均。
- [metric_recheck.csv](metric_recheck.csv)：1,300 筆數值核對。
- [horizon_predictions_for_recheck.npz](horizon_predictions_for_recheck.npz)：逐窗口的品質預測與真值，供重算。
- [evaluation_validation.json](evaluation_validation.json)：核對狀態及受保護來源雜湊。

本次結果只涵蓋 seed42，沒有新增 seed。未自動替換主要模型或原有配置。
