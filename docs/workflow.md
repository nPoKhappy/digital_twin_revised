# 工作流程與入口

所有命令從專案根目錄執行，使用 `python -m scripts.<分類>.<檔名>`。原有 CLI 參數與計算邏輯保留；路徑及跨程式匯入配合新位置更新。

## 研究流程

1. `scripts/sampling/` 生成 LHS 擾動條件；需要的完整穩態資料由 Aspen 模擬取得。
2. `scripts/preprocessing/` 按既有規則切分動態資料。
3. `scripts/training/train_tabular_v2.py` 訓練穩態 ANN teacher。
4. `scripts/training/` 中的動態／PGIN 入口依各自方法訓練。
5. `scripts/prediction/` 使用設定、權重及對應標準化統計進行推論。
6. `scripts/evaluation/` 評估 ANN-reference Gain／KCI；`scripts/analysis/` 與 `scripts/plotting/` 分析結果。

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
python -m scripts.evaluation.diagnose_ann_gain_pair --config configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42.yaml
python -m scripts.analysis.analyze_gain_pair_difficulty --help
python -m scripts.plotting.plot_gain_vs_no_gain_comparison --help
python -m scripts.plotting.plot_kci_pair_comparison
python -m scripts.plotting.plot_ss_weight_horizon_comparison --help
```

## 資料與產物位置

- `data/Claus_dynamic/`、`data/Claus_steady_state/`、既有 `saved_models/` 與實驗 `results/` 維持原有內容及名稱。
- 原 `LHS/LHS_air2_t2_*_perturbation.csv` 搬至 `data/sampling/`，內容保留。
- 原 `LHS/LHS_Air2_T2_*_Check_Plots/` 搬至 `results/sampling/`，包含既有 Hard-OOD 圖；本次沒有新增 Hard-OOD 生成方法。
- `results/<exp_name>/zscore_mean.csv`、`zscore_std.csv` 是模型推論依賴；`results/Tabular_MLP_New/` 的 ANN 統計量也是訓練與評估依賴。
- 歷史日誌、備份與既有 provenance 內容保留原始紀錄，其中出現的舊路徑代表當時位置。
- 舊 `gain/`、`LHS/`、`Claus_steady_state/` 與 `scripts/` 根層的 Python 快取原樣保存在 `logs/legacy_python_cache_20261019/`；包含部分已無來源檔的歷史快取，沒有刪除。
- 若其他專案或 Aspen 工作流程使用原 `LHS/` CSV 路徑，需將其輸入位置改為 `data/sampling/`。

## 搬移前已存在的限制

- `plot_data11_concentration_surfaces.py` 匯入 `show_lhs_air2_tr2_surface`；目前專案缺少此模組。本次只搬位置，不補寫曲面運算。
- `plot_plant_simulated_distribution.py` 依賴 `plant_simulated_data/`、`Claus_plant_data/` 與 `*_converted.csv`；目前 checkout 缺少這些資料。
- `predict_step_change.py` 與 `split_dataset_partitions.py` 的原預設 `configs/transformer_layerwise_57var.yaml` 在搬移前就不存在；請用 `--config` 明確指定目前的設定檔。
- `configs/steady_state/tabular_mlp_claus.yaml` 保留原設定，內含目前不存在的 `data/my_own_data/` 路徑；它不是 `train_tabular_v2` 的設定入口。
- `scripts/maintenance/annotate_difficult_gradnorm.py` 是針對舊版 1,488 行來源的歷史一次性工具，保留原有行數保護；不要對目前版本直接重跑。
- `scripts/maintenance/check_ann_step_data_20261013.py` 執行時會寫入既定的 ANN 驗證結果資料夾，沿用原有行為。

## 驗證方式

搬移採用目前工作區內容作為基準，包含尚未提交的 Difficult-GradNorm 修改。以獨立副本執行搬移前後的數值與檔案輸出比對，保留既有訓練設定和實驗產物。驗證副本、測試 CSV、圖表、暫存設定及驗證工具會在完成後刪除。
