# 三模型 test 比較：No-Gain、Gain=250、SS-only=2.5

內部：R5、R5-1、R5-2、R5-6 的 test split。外部：R5-19～R5-26 完整 CSV。這些是既有 test 案例；未混入 R5-3／4／5 或 R5-27／28／29 驗證案例。

三個固定 seed42 checkpoint。只推論，無訓練更新。SS-only 的 Gain=0、SS=2.5、SS target scope=all_48_outputs。

## ????

SS-only=2.5 ??? AvgRelRMSE ???0.803887????? Gain=250?0.874056??No-Gain=1?????????????????????? RMSE?

SS-only=2.5 ??? 0.908805?Gain=250 ? 1.075606?SS-only ??????????????????? 0.711081 ? 0.710273??????????????? seed ???????

???? 24 ???????H1/H18/TS ????SS-only ? 15 ??? No-Gain?Gain ? 8 ???? 48 ?????? 33 ??38 ?????????????????????????????

??????? TS RMSE ????????H2S/SO2??? SS-only=2.5 ???????????? H1 ? AvgRelRMSE ?? 1.087342????????????

R5-26 ???????SS-only ? TS R? ? H2S=-2.444?SO2=-1.843????????????????????????????

## AvgRelRMSE

同案例、同目標、同 H1/H18/TS 的 RMSE 先除以 No-Gain。內部 24 項、外部 48 項各自取幾何平均，再計算 sqrt(I×E)，內外各占 50%。No-Gain=1，越低越好。

| 模型 | 內部 | 外部 | 整體（50%／50%） |
|---|---:|---:|---:|
| SS-only=2.5 | 0.908805344 | 0.711080686 | 0.803886763 |
| Gain=250 | 1.075605954 | 0.710272942 | 0.874055951 |
| No-Gain | 1.000000000 | 1.000000000 | 1.000000000 |

## H1／H18／TS：兩區各 50%

| 模型 | H1 | H18 | TS |
|---|---:|---:|---:|
| No-Gain | 1.000000000 | 1.000000000 | 1.000000000 |
| Gain=250 | 0.927397432 | 0.845106082 | 0.852001944 |
| SS-only=2.5 | 0.863423397 | 0.795826227 | 0.756036168 |

## 原單位平均案例 RMSE（表內乘以 10⁻³ kmol/kmol）

各區內的案例 RMSE 等權算術平均，與逐案例比值的幾何平均不同。

| 區域 | 目標 | 評估 | No-Gain | Gain=250 | SS-only=2.5 |
|---|---|---|---:|---:|---:|
| external | B35_H2S | H1 | 2.143262 | 1.677002 | 1.016764 |
| external | B35_H2S | H18 | 2.608890 | 1.899062 | 1.525038 |
| external | B35_H2S | TS | 2.659948 | 1.961261 | 1.481772 |
| external | B35_SO2 | H1 | 1.251624 | 1.029812 | 0.519000 |
| external | B35_SO2 | H18 | 1.830868 | 1.074197 | 1.277589 |
| external | B35_SO2 | TS | 1.893220 | 1.120520 | 0.885394 |
| internal | B35_H2S | H1 | 0.489096 | 0.545241 | 0.578453 |
| internal | B35_H2S | H18 | 0.678685 | 0.725113 | 0.673684 |
| internal | B35_H2S | TS | 0.718991 | 0.756346 | 0.690883 |
| internal | B35_SO2 | H1 | 0.193144 | 0.211397 | 0.193668 |
| internal | B35_SO2 | H18 | 0.298433 | 0.317067 | 0.212510 |
| internal | B35_SO2 | TS | 0.289453 | 0.299962 | 0.220703 |

## 圖與數據

- [AvgRelRMSE 圖](test_avgrelrmse_comparison.png)／[PDF](test_avgrelrmse_comparison.pdf)
- [原始 RMSE 摘要](test_rmse_summary.png)／[H1～H18](test_horizon_rmse.png)
- [內部 TS](time_series_internal.png)／[外部 19～22](time_series_external_19_22.png)／[外部 23～26](time_series_external_23_26.png)
- [整體分數](avgrelrmse_overall.csv)／[H1/H18/TS](avgrelrmse_by_evaluation.csv)／[原始平均 RMSE](test_summary.csv)
- [144 項與 No-Gain 配對比較](paired_case_comparison.csv)／[72 項 SS-only 對 Gain 比較](ss2p5_vs_gain250.csv)／[改善項數](comparison_counts.csv)
- [216 項 test 指標](test_metrics.csv)／[1296 項 horizon 指標](horizon_metrics.csv)／[1368 項 RMSE 重算](RMSE_recheck.csv)
- [資料案例](data_audit.csv)／[計算定義](evaluation_protocol.json)／[來源與檢查](evaluation_validation.json)

TS 僅最初 18 點使用真實歷史，其後回填預測；H1/H18 每個滑動起點重新給真實歷史，再遞迴預測第 1／18 步。皆使用已知未來 decoder inputs。

相關 H1/H18/TS 項目不是獨立重複實驗，單一 seed 不代表統計顯著性。本次未重算 KCI，亦未使用 test 結果重新選超參數；不宣稱為未曾檢視的新盲測。

重跑：`.\.venv\Scripts\python.exe scripts/prediction/compare_three_model_test.py`；僅重畫加上 `--plot-only`。
