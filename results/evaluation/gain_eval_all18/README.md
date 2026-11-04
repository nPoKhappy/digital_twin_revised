# 穩態 KCI（all18）結果索引

依實際使用的穩態 probe 資料分類；本次只整理位置與路徑，未重跑模型或更動數值。

## 分類與資料來源

| 分類 | 核心範圍 in-range | Soft-OOD 外插 | 篩選後操作點數 |
|---|---|---|---|
| `validation/` | `data/Claus_steady_state/lhs_valid_in_range.xlsx` | `data/Claus_steady_state/lhs_valid_soft_ood.xlsx` | 4404 / 2585 |
| `test/` | `data/Claus_steady_state/lhs_test_in_range.xlsx` | `data/Claus_steady_state/lhs_test_soft_ood_3.xlsx` | 4385 / 2692 |
| `logs/` | 跨模型執行日誌 | — | — |

操作點是一組完整 Aspen 穩態狀態；這些數量是 KCI probe 數，與動態 RMSE 的 R5 時序案例數不同。所有現有結果均為 seed42。

## 驗證資料 KCI

| 模型／用途 | 結果 |
|---|---|
| Gain=0.05,2.5,25,250 | [validation/gain_weights_seed42_20261104](validation/gain_weights_seed42_20261104) |
| Gain=2500 | [validation/gain2500_seed42_20261104](validation/gain2500_seed42_20261104) |
| SS-only=0.05,2.5,25,250,2500 | [validation/ss_only_seed42_20261104](validation/ss_only_seed42_20261104) |

前四個 Gain 權重與 Gain=2500 的補跑批次保留各自原始產物；動態比較圖已整合五個權重。SS-only 僅保留 0.05、2.5、25、250、2500。

- [五個 Gain：動態 validation RMSE／KCI 圖](../../visualization/gain005_gain2p5_gain25_gain250_gain2500_casewise_region50_validation_seed42/README.md)
- [五個 SS-only：動態 validation RMSE／KCI 圖](../../visualization/ss_only_all_weights_casewise_region50_validation_seed42/README.md)

## 測試資料 KCI

| 模型／用途 | 結果 |
|---|---|
| No-Gain | [test/no_gain_seed42](test/no_gain_seed42) |
| Gain=0.05 | [test/gain0p05_seed42](test/gain0p05_seed42) |
| Difficult-GradNorm rho=0.005 | [test/difficult_gradnorm/rho0p005_seed42](test/difficult_gradnorm/rho0p005_seed42) |
| Difficult-GradNorm rho=0.01 | [test/difficult_gradnorm/rho0p01_seed42](test/difficult_gradnorm/rho0p01_seed42) |
| Difficult-GradNorm rho=0.015 | [test/difficult_gradnorm/rho0p015_seed42](test/difficult_gradnorm/rho0p015_seed42) |
| Difficult-GradNorm rho=0.02 | [test/difficult_gradnorm/rho0p02_seed42](test/difficult_gradnorm/rho0p02_seed42) |


## 檔案用途

- `gain_evaluation_detail.csv`：分布 × 輸出 × MV 的配對 KCI、有效分母及混淆計數。
- `gain_evaluation_summary.csv`：每個分布的 micro KCI 與 coverage。
- `gain_probe_diagnostics.csv`：逐探測點／擾動方向／配對／步長診斷，檔案較大，全部保留。
- `evaluation_validation.json`：原始評估與來源核對紀錄。
- [catalog.csv](catalog.csv)：9 個結果資料夾的分類、模型、路徑與原資料夾對照。

執行日誌與 `run_command.json` 保留當時命令；歷史命令可能含整理前路徑。搬移對照與逐檔 SHA-256 核對保存在 [整理紀錄](../../../logs/gain_eval_classification_20261104.json)。

KCI 是 ANN-reference 方向一致性，不代表 Aspen 物理 Gain 正確率或閉迴路控制性能。Seed43／44 的既有桌面封存位置維持不變。

舊 Gain+SS 的 test KCI 已清除，待新的五組配對權重重跑後再加入。
