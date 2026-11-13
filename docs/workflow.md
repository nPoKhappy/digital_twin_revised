# 工作流程與入口

所有命令從專案根目錄執行，使用 `python -m scripts.<分類>.<檔名>`。原有 CLI 參數與計算邏輯保留；路徑及跨程式匯入配合新位置更新。

## 研究流程

1. `scripts/sampling/` 生成 LHS 擾動條件；需要的完整穩態資料由 Aspen 模擬取得。
2. `scripts/preprocessing/` 按既有規則切分動態資料。
3. `scripts/training/train_tabular_v2.py` 訓練穩態 ANN teacher。
4. `scripts/training/` 中的動態／PGIN 入口依各自方法訓練。
5. `scripts/prediction/` 使用設定、權重及對應標準化統計進行推論。
6. `scripts/evaluation/` 評估 ANN-reference KCI，`scripts/analysis/` 與既有 `scripts/plotting/` 保留；現行比較見 README 的「分析與繪圖」，可用各自 rebuild_figures.py 重畫。

既有權重可以直接用於推論與評估，不需要因為搬移程式重新訓練。採樣 CSV 只是操作條件，不能直接代替 Gain 評估所需的完整 Aspen 穩態匯出資料。

## 訓練入口

| 模組 | 原有用途 |
| --- | --- |
| `scripts.training.train` | 一般動態模型訓練 |
| `scripts.training.train_tabular_v2` | 穩態 ANN teacher 訓練 |
| `scripts.training.train_pgin_from_scratch` | 原有 from-scratch PGIN 入口 |
| `scripts.training.train_pgin_autoregressive_gain` | 自回歸 Gain 訓練 |
| `scripts.training.train_pgin_stepwise_gain` | 逐步 Gain 訓練 |
| `scripts.training.train_pgin_stepwise_gain_ss_loss` | 逐步 Gain 與 SS value loss 訓練 |
| `scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm` | Difficult-GradNorm；保留原有 CLI override 行為 |

Difficult-GradNorm 的 CLI 會指定八份 baseline 穩態來源、`DIFFICULT_PAIRS`，並依 `--no-ss` 覆寫 SS 權重。不要只因 YAML 檔案可以讀取，就把其他實驗一律改用這個入口。

## 其他常用命令

```powershell
python -m scripts.training.train_tabular_v2
python -m scripts.prediction.predict_tabular_v2
python -m scripts.prediction.predict_step_change --config configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42.yaml
```

## 資料與產物位置

- 資料輸入保留在 `data/`。動態結果分類至 `results/pgin/<family>/<exp_name>/`；權重保留在 `saved_models/`，舊 Gain+SS 專用模型與結果已清除。
- 原 `LHS/LHS_air2_t2_*_perturbation.csv` 搬至 `data/sampling/`，內容保留。
- 原 `LHS/LHS_Air2_T2_*_Check_Plots/` 搬至 `results/sampling/`，包含既有 Hard-OOD 圖；本次沒有新增 Hard-OOD 生成方法。
- `results/pgin/<family>/<exp_name>/zscore_mean.csv`、`zscore_std.csv` 是模型推論依賴；`results/steady_state/Tabular_MLP_New/` 的 ANN 統計量也是訓練與評估依賴。
- 歷史日誌、備份與既有 provenance 內容保留原始紀錄，其中出現的舊路徑代表當時位置。
- 舊 `gain/`、`LHS/`、`Claus_steady_state/` 與 `scripts/` 根層的 Python 快取原樣保存在 `logs/legacy_python_cache_20261019/`；包含部分已無來源檔的歷史快取，沒有刪除。
- 若其他專案或 Aspen 工作流程使用原 `LHS/` CSV 路徑，需將其輸入位置改為 `data/sampling/`。

## 搬移前已存在的限制

- `predict_step_change.py` 與 `split_dataset_partitions.py` 的原預設 `configs/transformer_layerwise_57var.yaml` 在搬移前就不存在；請用 `--config` 明確指定目前的設定檔。
- `configs/steady_state/tabular_mlp_claus.yaml` 保留原設定，內含目前不存在的 `data/my_own_data/` 路徑；它不是 `train_tabular_v2` 的設定入口。
- `scripts/maintenance/annotate_difficult_gradnorm.py` 是針對舊版 1,488 行來源的歷史一次性工具，保留原有行數保護；不要對目前版本直接重跑。
- `scripts/maintenance/check_ann_step_data_20261013.py` 執行時會寫入既定的 ANN 驗證結果資料夾，沿用原有行為。

## 驗證方式

搬移採用目前工作區內容作為基準，包含尚未提交的 Difficult-GradNorm 修改。以獨立副本執行搬移前後的數值與檔案輸出比對，保留既有訓練設定和實驗產物。驗證副本、測試 CSV、圖表、暫存設定及驗證工具會在完成後刪除。


2026-11-04：原有 scripts 程式已還原。現行 validation 比較由 `scripts/prediction/validate_hyperparameters.py` 產生，三模型 test 比較由 `scripts/prediction/compare_three_model_test.py` 產生。兩組 validation 各自提供 `rebuild_figures.py`，使用保存的 CSV 重畫。

## Results 分類與 Gain+SS 重跑

所有 YAML 已加入 `output.results_dir`，例如 `./results/pgin/gain_ss`。訓練、動態推論及 KCI 評估透過 `src/experiment_paths.py` 取得同一份標準化統計位置。完整位置見 [Results 總覽](../results/README.md)。ANN teacher 專用入口仍使用 `results/steady_state/Tabular_MLP_New/`。

新五組 Gain+SS 的兩個權重分別同為 0.05、2.5、25、250、2500，沿用 seed42 及全部 48 個 SS 輸出目標。依 [重跑指令](../configs/pgin/gain_ss/README.md) 訓練；本次整理沒有啟動訓練。保留的混合模型歷史圖含有舊 Gain+SS 結果，不能當作這五組的新結果。舊的特定 Gain+SS 消融繪圖入口仍描述原來實驗，需要相符的歷史產物才能重跑。

2026-11-05 五組 Gain+SS 已完成訓練。使用 `python -m scripts.prediction.validate_gain_ss_hyperparameters` 重新評估固定 checkpoint；內插 R5-3／4／5、外插 R5-27／28／29，AvgRelRMSE 內外各 50%。穩態 KCI 使用同一批 validation XLSX，獨立於動態 RMSE 案例。輸出見 [Gain+SS validation](../results/visualization/gain_ss_all_weights_casewise_region50_validation_seed42/README.md)。此評估入口提供 `--phase dynamic|kci|report`，完整預設為 `all`；相同輸入可使用已保存的預測或完成批次，輸入雜湊改變則停止。
