# Seed42 六模型與三個 GradNorm ratio

保留既有 No-Gain、Gain、Gain＋SS=0.01、Difficult-GradNorm ratio=0.015／0.01，加入新訓練的 **ratio=0.005**。0.005 為綠色倒三角／點劃線，0.01 為紫色粗十字／虛線，0.015 為桃紅菱形／點線。其餘模型維持原配色。

| 圖表 | PNG | PDF |
| --- | --- | --- |
| Internal test，四個 test split | [查看](six_model_raw_rmse_internal_test.png) | [PDF](six_model_raw_rmse_internal_test.pdf) |
| External R5-19～R5-26 | [查看](six_model_raw_rmse_external_all_data.png) | [PDF](six_model_raw_rmse_external_all_data.pdf) |
| 指定 internal validation：R5-3、4、5 | [查看](six_model_rmse_internal_validation.png) | [PDF](six_model_rmse_internal_validation.pdf) |
| 指定 external validation：R5-13、14 | [查看](six_model_rmse_external_validation.png) | [PDF](six_model_rmse_external_validation.pdf) |
| In-range／Soft-OOD all18 KCI | [查看](six_model_all18_kci.png) | [PDF](six_model_all18_kci.pdf) |

RMSE 圖保留每個案例的 H1、H18、TS 原始數值，不跨案例平均；各面板縱軸範圍不同。External 圖較寬，可放大 PNG 或開啟向量 PDF。沒有 error bar，所有結果皆為 seed42。

指定 internal validation 的 R5-5 部分點超出既定 air2／T2 核心範圍，圖中以星號註記。KCI 是 ANN teacher 方向一致性，與動態 RMSE 使用不同的評估資料。

繪圖資料：[six_model_test_rmse_values.csv](six_model_test_rmse_values.csv)（432 筆）、[six_model_validation_rmse_values.csv](six_model_validation_rmse_values.csv)（180 筆）、[six_model_all18_kci_values.csv](six_model_all18_kci_values.csv)（48 筆）。

完整分析、正負 teacher 分組與驗證紀錄：[evaluation_report.md](../../pgin_gain005_ratio0p005_seed42_difficult_group_gradnorm_no_ss/evaluation_report.md)。原有五模型圖保留在原資料夾。
