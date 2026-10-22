# Seed42 五模型驗證圖

依指定將 R5-3、R5-4、R5-5 歸為 internal validation，R5-13、R5-14 歸為 external validation。五個模型為 No-Gain、Gain、Gain＋SS=0.01、Difficult-GradNorm ratio=0.015、Difficult-GradNorm ratio=0.01。

| 圖表 | PNG | PDF |
| --- | --- | --- |
| Internal：H₂S／SO₂ 的 H1、H18、TS RMSE | [查看](five_model_rmse_internal_validation.png) | [PDF](five_model_rmse_internal_validation.pdf) |
| External：H₂S／SO₂ 的 H1、H18、TS RMSE | [查看](five_model_rmse_external_validation.png) | [PDF](five_model_rmse_external_validation.pdf) |
| Internal：真值與五模型 TS 預測疊圖 | [查看](five_model_TS_internal_validation.png) | [PDF](five_model_TS_internal_validation.pdf) |
| External：真值與五模型 TS 預測疊圖 | [查看](five_model_TS_external_validation.png) | [PDF](five_model_TS_external_validation.pdf) |

桃紅菱形／點線為 ratio=0.015；紫色粗十字／虛線為 ratio=0.01。真值以黑色實線表示。各 RMSE 面板有各自的縱軸範圍，請在同一面板內比較；TS 疊圖縱軸包含全部模型預測。沒有 error bar。

Internal 的三個案例皆新產生預測；external 兩個案例沿用既有預測，並重新核對。兩個 GradNorm 版本皆無 SS loss，五模型皆為 seed42。本次沒有訓練或更新權重。

R5-5 依指定放在 internal，但只有 66.81% 原始時間點同時位於 air2=[140,300]、T2=[140,240] 的既定核心範圍內；這個分組名稱不表示整條軌跡均為嚴格內插。R5-13 的 air2 超過上限；R5-14 的 T2 超過上限。

完整數值：[five_model_validation_metrics.csv](five_model_validation_metrics.csv)，共 150 筆（5 模型 × 5 案例 × 2 品質 × 3 評估）。

比較解讀及驗證紀錄：[findings.md](../../validation_r5_3_4_5_r5_13_14_seed42_20261022/findings.md)。原有 internal／external test 圖表未修改。
