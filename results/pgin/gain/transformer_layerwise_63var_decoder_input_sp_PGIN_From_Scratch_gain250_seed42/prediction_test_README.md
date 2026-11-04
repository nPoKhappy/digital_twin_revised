# Gain=250：test-only 預測

內部 R5／R5-1／R5-2／R5-6 test split，外部 R5-19～26，共 12 組。48 個輸出、576 筆 TS RMSE 獨立重算、72 筆 H1／H18／TS 指標核對通過。

[五模型完整比較](../visualization/no_gain_gain005_gain2p5_gain25_gain250_test_seed42/README.md) · [Gain250 指標](prediction_test_H1_H18_TS_summary.csv) · [執行紀錄](prediction_test.log)

各案例子目錄包含原始單位預測 CSV、指標與濃度圖；grouped_time_series 及 grouped_horizon_parity 包含各案例 TS、H1／H18 圖。
