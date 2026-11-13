# Gain+SS 五個同步權重：穩態 validation KCI

2026-11-05 重新評估五個 seed42 checkpoint；Gain 與 SS 權重相同，SS 目標為全部 48 個輸出。

| 分布 | 穩態來源 | 探測點數 |
| --- | --- | ---: |
| 內插 | `data/Claus_steady_state/lhs_valid_in_range.xlsx` | 4404 |
| 外插 | `data/Claus_steady_state/lhs_valid_soft_ood.xlsx` | 2585 |

沿用 Gain／SS-only 的 ANN teacher、標準化統計、有效 teacher gain 門檻 1e-5，以及正負各 0.5 個訓練標準差的有限差分。先 warmup 35 步，再對全部 18 個 post-step 時刻逐點判斷增益方向；動態增益為零列入分母但不算正確。
使用既有 evaluator 的數值函式；以 NumPy 從保存增益陣列重算混淆計數，並串流重讀逐點 CSV 交叉核對。五組有效分母與既有 Gain validation 完全相同。

| Gain = SS | 整體 micro KCI (%) |
| ---: | ---: |
| 0.05 | 92.474863 |
| 2.5 | 94.707833 |
| 25 | 97.029306 |
| 250 | 98.463036 |
| 2500 | 94.150887 |

各模型／分布子資料夾包含：

- `gain_evaluation_detail.csv`：2 個目標 × 2 個 MV 的計數與分數。
- `gain_arrays.npz`：ANN teacher gain、兩方向 × 18步 × probes × 2 targets × 2 MVs 的動態增益，以及 ANN 五個物理輸入。
- `gain_probe_diagnostics.csv`：逐點明細，完整保留在本機，依既有規則不加入 Git。
- `evaluation_validation.json`：兩種重算核對、列數與產物 SHA-256。

[五組配對 KCI](gain_ss_pair_kci_comparison.csv)／[區域摘要](gain_evaluation_summary.csv)／[完整動態 RMSE 與圖表](../../../../visualization/gain_ss_all_weights_casewise_region50_validation_seed42/README.md)

整體 KCI 為兩分布 correct／valid 合併的 micro 比率，不採內外 50%；內外各 50% 僅適用 AvgRelRMSE。沒有使用 test 穩態資料或訓練 KCI history。KCI 是 ANN-reference 方向一致性，不是物理正確性或閉迴路性能。
