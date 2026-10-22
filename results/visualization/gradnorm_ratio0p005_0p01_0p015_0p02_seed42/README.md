# 七模型比較：seed42

No-Gain、Gain、Gain + SS=0.01 與 Difficult-GradNorm ratio=0.005／0.01／0.015／0.02。0.02 為橘紅色 X 與實線。沿用現有案例與圖表樣式，舊圖保留；每張圖提供 PNG／PDF。單 seed 的逐案例數值不加 error bar。

- [seven_model_raw_rmse_internal_test](./seven_model_raw_rmse_internal_test.png)（[PDF](./seven_model_raw_rmse_internal_test.pdf)）
- [seven_model_raw_rmse_external_all_data](./seven_model_raw_rmse_external_all_data.png)（[PDF](./seven_model_raw_rmse_external_all_data.pdf)）
- [seven_model_rmse_internal_validation](./seven_model_rmse_internal_validation.png)（[PDF](./seven_model_rmse_internal_validation.pdf)）
- [seven_model_rmse_external_validation](./seven_model_rmse_external_validation.png)（[PDF](./seven_model_rmse_external_validation.pdf)）
- [seven_model_all18_kci](./seven_model_all18_kci.png)（[PDF](./seven_model_all18_kci.pdf)）

`seven_model_test_rmse_values.csv`、`seven_model_validation_rmse_values.csv` 保存繪圖數值；KCI 配對與整體數值亦分別保存 CSV。H1／H18 為滾動起點的預測步指標，TS 為連續滾動預測指標。不同 panel 的 y 軸範圍不同。

[評估報告](../../pgin_gain005_ratio0p02_seed42_difficult_group_gradnorm_no_ss/evaluation_report.md)。本次僅比較 seed42；KCI 代表 ANN-reference 同號一致性。
