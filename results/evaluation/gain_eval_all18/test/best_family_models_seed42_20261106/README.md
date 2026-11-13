# Validation 選定的三家族最佳模型與 No-Gain：Test KCI

2026-11-06 固定選用 Gain=250、SS-only=2.5、Gain+SS=(2.5,2.5)，另含 No-Gain。選擇依據為各家族 validation AvgRelRMSE 最低值，沒有依 test 結果挑選權重。

使用 `lhs_test_in_range.xlsx`（4385點）與 `lhs_test_soft_ood.xlsx`（2692點）。後者為先前 `_3.xlsx` 的內容改名。ANN teacher、模型與標準化統計在動態 RMSE、穩態 KCI 間一致。

KCI 沿用既有 native evaluator：warmup35步、正負各0.5個訓練標準差的有限差分、18個post-step時刻的嚴格方向一致性、teacher gain有效門檻1e-5。零動態增益不算正確。

每個模型／分布包含增益陣列 NPZ、配對計數 CSV、逐點診斷 CSV 與核對 JSON；NumPy 及 CSV 兩種重算均通過。逐點 CSV 依既有規則保留本機、不加入 Git。No-Gain 八個分布／配對計數與原有 test KCI 完全一致。

[配對明細](gain_evaluation_detail.csv)／[區域摘要](gain_evaluation_summary.csv)／[五張表與內外插圖](../../../../visualization/best_gain_ss_gainss_test_seed42/README.md)

本批僅使用 test 穩態資料；KCI 為 ANN-reference 方向一致性，不是物理增益正確率或閉迴路驗證。
