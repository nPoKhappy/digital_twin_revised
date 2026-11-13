# Results 分類

訓練產物對應 `configs/` 的層級；共用評估與跨模型比較另外集中存放。

```text
results/
├─ pgin/
│  ├─ no_gain/<exp_name>/
│  ├─ gain/<exp_name>/
│  ├─ gain_ss/<exp_name>/
│  ├─ ss_only/<exp_name>/
│  └─ difficult_gradnorm/<exp_name>/
├─ steady_state/Tabular_MLP_New/
├─ evaluation/gain_eval_all18/     # test／validation KCI 批次與 catalog.csv
├─ analysis/                      # 資料與 ANN teacher 等專題分析
├─ visualization/                 # 跨模型比較、圖表、數據與重畫腳本
└─ sampling/                      # LHS 操作條件檢查
```

每個動態模型 YAML 的 `output.results_dir` 指定分類父目錄，程式再附加 `exp_name`，例如 `configs/pgin/gain/` → `results/pgin/gain/<exp_name>/`。其中包含訓練圖、標準化統計與逐案例預測；checkpoint 保留在 `saved_models/`。ANN teacher 的既有專用入口使用 `steady_state/Tabular_MLP_New/`；`configs/steady_state/tabular_mlp_claus.yaml` 是另一份歷史設定，並非該入口的輸入。

[experiment_catalog.csv](experiment_catalog.csv) 列出設定與對應輸出位置；Gain+SS 五組已於 2026-11-05 核對 checkpoint 與統計量並更新狀態，其餘保留 2026-11-04 整理快照。訓練與評估指令見 [Gain+SS 說明](../configs/pgin/gain_ss/README.md)。

2026-11-04 將原本根層的實驗資料夾移入上述分類；保留模型的數值產物未重新計算。舊 Gain+SS 專用結果與 checkpoint 已刪除；跨模型的歷史比較保留原數值，包含舊 Gain+SS 的圖表不代表新五組的結果。歷史執行日誌中的命令保留當時路徑。

主要現行比較：

- [五個 Gain validation](visualization/gain005_gain2p5_gain25_gain250_gain2500_casewise_region50_validation_seed42/README.md)
- [五個 SS-only validation](visualization/ss_only_all_weights_casewise_region50_validation_seed42/README.md)
- [五個 Gain+SS validation](visualization/gain_ss_all_weights_casewise_region50_validation_seed42/README.md)／[結果解讀](visualization/gain_ss_all_weights_casewise_region50_validation_seed42/analysis_summary.md)
- [No-Gain、Gain=250、SS-only=2.5 test](visualization/three_model_no_gain_gain250_ss2p5_test_seed42/README.md)
- [三家族最佳模型與 No-Gain：五張 test 表格及內外插比較圖](visualization/best_gain_ss_gainss_test_seed42/README.md)
- [KCI 評估索引](evaluation/gain_eval_all18/README.md)
