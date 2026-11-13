# Modified Claus Digital Twin

改良型克勞斯製程的動態預測與物理引導神經網路研究專案。以穩態 ANN 作為 Gain teacher，訓練 Transformer 動態模型，比較 No-Gain、Gain、Gain + SS value loss 與 Difficult-GradNorm 等方法，並分析預測誤差及 ANN-reference Gain／KCI。

本專案包含 LHS 操作條件生成、資料切分、ANN／Transformer 訓練、推論、Gain 評估與結果繪圖。LHS 生成器輸出操作條件；完整製程響應與穩態輸出仍需由 Aspen 模擬取得。

## 導覽

- [環境與執行方式](#環境與執行方式)
- [專案目錄](#專案目錄)
- [資料與模型需求](#資料與模型需求)
- [設定檔與實驗命名](#設定檔與實驗命名)
- [訓練](#訓練)
- [推論](#推論)
- [Gain 與 KCI 評估](#gain-與-kci-評估)
- [採樣與資料切分](#採樣與資料切分)
- [分析與繪圖](#分析與繪圖)
- [輸出與重現實驗](#輸出與重現實驗)
- [既有限制與常見問題](#既有限制與常見問題)

## 環境與執行方式

環境依 [pyproject.toml](pyproject.toml)、[uv.lock](uv.lock) 與 [.python-version](.python-version) 管理。目前 Python 版本檔指定 `3.11.13`，主要依賴包含 PyTorch `2.6.0`、NumPy、pandas、scikit-learn、matplotlib、SciPy、PyYAML 與 openpyxl；PyTorch 套件來源設定為 CUDA 12.6 索引。

Windows PowerShell：

```powershell
Set-Location 'C:\Users\Administrator\Desktop\digital_twin_revised'
uv sync
.\.venv\Scripts\Activate.ps1
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

以下命令都從專案根目錄執行，使用 `python -m scripts.<分類>.<檔名>`。若沒有啟用虛擬環境，將 `python` 換成 `.\.venv\Scripts\python.exe`。即使部分程式會依自身位置尋找根目錄，資料與設定仍有相對於工作目錄的路徑，請維持在專案根目錄執行。

目前 Difficult-GradNorm 訓練入口會在 CUDA 可用且 YAML 指定 `device: cuda` 時使用 GPU，否則使用 CPU。Gain 評估器可透過 `--device auto|cpu|cuda` 明確指定；要求 `cuda` 但不可用時會報錯。

## 專案目錄

```text
digital_twin_revised/
├─ src/                         # 共用功能
│  ├─ models/                   # Transformer、MLP 與模型工廠
│  ├─ dataset.py                # 動態時序 Dataset
│  ├─ dataset_tabular.py        # 表格資料 Dataset
│  ├─ engine.py                 # 共用訓練步驟
│  ├─ utils.py                  # 讀檔、轉換、標準化與指標
│  └─ variable_selection.py     # 模型變數與欄位契約
├─ scripts/                     # 執行入口
│  ├─ training/                # ANN、動態模型與 PGIN 訓練
│  ├─ prediction/              # 動態、階躍與 ANN 推論
│  ├─ sampling/                # LHS 操作條件生成
│  ├─ preprocessing/           # 動態資料切分
│  └─ maintenance/             # 歷史維護與 ANN 階躍檢查工具
├─ configs/
│  ├─ steady_state/            # 既有穩態 YAML
│  └─ pgin/
│     ├─ no_gain/
│     ├─ gain/
│     ├─ gain_ss/
│     ├─ ss_only/
│     └─ difficult_gradnorm/
├─ data/
│  ├─ Claus_dynamic/           # 動態原始資料、切分檔與階躍案例
│  ├─ Claus_steady_state/      # Aspen 穩態訓練與評估資料
│  └─ sampling/                # LHS 操作條件 CSV
├─ saved_models/               # 模型權重
├─ results/                    # 分類總覽見 results/README.md
│  ├─ pgin/                    # 與 configs/pgin/ 相同的五種模型分類
│  ├─ steady_state/            # ANN teacher 統計量與產物
│  ├─ evaluation/              # KCI 評估批次
│  ├─ analysis/                # 專題分析
│  ├─ visualization/           # 跨模型比較與圖表
│  └─ sampling/                # LHS 採樣檢查圖
├─ logs/                       # 日誌與歷史備份
├─ docs/
│  ├─ workflow.md              # 入口、資料流與搬移說明
│  └─ experiment_index.md      # 設定、權重與統計量對照
├─ pyproject.toml
├─ uv.lock
└─ README.md
```

## 資料與模型需求

| 工作 | 主要輸入 | 必須一起保留的檔案 |
| --- | --- | --- |
| 動態／PGIN 訓練 | YAML `data.path`、`training_files` 指定的 CSV | Gain／SS 方法需要對應穩態資料與 ANN teacher |
| 穩態 ANN 訓練 | `data/Claus_steady_state/lhs_generated_dynamic_ss_data*.xlsx` | `train_tabular_v2` 使用內建設定與原始 Aspen 欄名對照 |
| 動態模型推論 | YAML `inference_files` 指定的案例、相符模型權重 | `results/pgin/<family>/<exp_name>/zscore_mean.csv`、`zscore_std.csv` |
| ANN 推論 | `results/steady_state/Tabular_MLP_New/tabular_test_dataset.csv` | ANN 權重與自己的標準化統計 |
| Gain／KCI 評估 | held-out 完整 Aspen 穩態 XLSX／CSV | 各動態模型的 YAML、權重、統計量，以及 ANN teacher |
| LHS 採樣 | 生成器內的 `SOURCE_CSV` | 原始欄位及固定操作條件來源 |

固定 ANN teacher 的檔案組合：

```text
saved_models/Tabular_MLP_5in_2out_QV.pth
results/steady_state/Tabular_MLP_New/zscore_mean.csv
results/steady_state/Tabular_MLP_New/zscore_std.csv
```

ANN teacher 的五個輸入為 `air2_SP`、`HEATER2_output_T_SP`、`acidgas_Fm`、`acidgas_P`、`acidgas_T`，輸出為 `B35_H2S`、`B35_SO2`。動態模型與 ANN 使用各自的標準化統計，不能互換。

目前 PGIN 入口會依既有流程載入 ANN teacher 與穩態資料，包括使用 Gain 權重 0 的 No-Gain 設定；單純把 loss 權重設為 0，不會自動移除這些檔案依賴。

Difficult-GradNorm 命令列入口固定載入八份訓練穩態來源，編號為 `10、11、15、16、17、18、19、20`，檔名格式為 `data/Claus_steady_state/lhs_generated_dynamic_ss_data_<編號>.xlsx`。其讀檔器依原有 Aspen 匯出格式處理欄名、單位列與完成狀態；新增資料時請維持同樣格式。

`results/` 同時包含推論必需的標準化統計，不只是可刪除的圖表目錄。需要清理某次實驗時，先按該次 `exp_name` 確認依賴。

## 設定檔與實驗命名

完整檔案對照見 [實驗索引](docs/experiment_index.md)。目前各設定分類如下：

| 分類 | Gain 權重 | SS 權重 | 用途 |
| --- | --- | --- | --- |
| `configs/pgin/no_gain/` | 0 | 0 | 動態預測基準 |
| `configs/pgin/gain/` | 0.05 / 2.5 / 25 / 250 / 2500 | 0 | Gain 引導 |
| `configs/pgin/gain_ss/` | 0.05 / 2.5 / 25 / 250 / 2500 | 與 Gain 相同 | 兩個權重同步變動；五組已訓練，[指令與驗證分析](configs/pgin/gain_ss/README.md) |
| `configs/pgin/ss_only/` | 0 | 0.05 / 2.5 / 25 / 250 / 2500 | 僅 SS value loss 比較 |
| `configs/pgin/difficult_gradnorm/` | 0.05 | 0 | 指定配對與 teacher 符號群組的梯度縮放 |

後續命令共用以下四個 seed42 設定變數，請先在同一個 PowerShell 工作階段執行：

```powershell
$noGainConfig = 'configs/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42.yaml'
$gainConfig = 'configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42.yaml'
$gainSSConfig = 'configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss005_seed42.yaml'
$difficultConfig = 'configs/pgin/difficult_gradnorm/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42_difficult_group_gradnorm_no_ss.yaml'
```

設定檔的 `exp_name` 連動 `saved_models/<exp_name>.pth` 與 `results/pgin/<family>/<exp_name>/`。重新訓練同一名稱會寫入原路徑；要保留既有實驗，先複製 YAML 並設定新的 `exp_name`。不要只改 YAML 檔名。

## 訓練

### 入口選擇

| 模組 | 用途 |
| --- | --- |
| `scripts.training.train` | 一般動態模型訓練 |
| `scripts.training.train_tabular_v2` | 穩態 ANN teacher 訓練，使用內建設定 |
| `scripts.training.train_pgin_from_scratch` | 原有 from-scratch PGIN 入口 |
| `scripts.training.train_pgin_autoregressive_gain` | 自回歸 Gain 訓練 |
| `scripts.training.train_pgin_stepwise_gain` | 逐步 Gain／No-Gain 訓練 |
| `scripts.training.train_pgin_stepwise_gain_ss_loss` | 逐步 Gain 與 SS value loss 訓練 |
| `scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm` | Difficult-GradNorm 與可選 SS value loss |

入口之間的 CLI override 與訓練行為不同，請依實驗方法選擇，不能只更換 YAML 就假設所有入口等價。

### 穩態 ANN teacher

```powershell
python -m scripts.training.train_tabular_v2
```

此入口不讀取 `configs/steady_state/tabular_mlp_claus.yaml`。它會從內建穩態來源訓練，輸出 ANN 權重、測試資料與標準化統計至固定位置。若已有可用 teacher，可以直接進行後續動態模型訓練或評估；重跑會更新既有 teacher 產物。

### No-Gain、Gain 與 Gain + SS

```powershell
python -m scripts.training.train_pgin_stepwise_gain --config $noGainConfig
python -m scripts.training.train_pgin_stepwise_gain --config $gainConfig
python -m scripts.training.train_pgin_stepwise_gain_ss_loss --config $gainSSConfig
```

以上是各方法的執行入口範例，不代表重建了既有產物的完整歷史執行命令。實驗比較需確認資料來源、seed、訓練設定與權重版本相符。

### Difficult-GradNorm，無 SS value loss

```powershell
python -m scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm --config $difficultConfig --no-ss --target-ratio 0.015 --max-scale 50
```

目前正式設定與 CLI 的重點：

| 項目 | 目前設定／行為 |
| --- | --- |
| seed | 42 |
| 動態 batch size／穩態 batch size | 64／64 |
| 最大 epochs／每輪動態 batches 上限 | 1000／10；仍受可用 batches 與 early stopping 限制 |
| learning rate | 0.0005 |
| Gain／SS 權重 | 0.05／0；`--no-ss` 明確覆寫 SS 權重為 0 |
| 歷史窗口 | 180 分鐘，以 10 分鐘間隔取樣，共 18 步 |
| `window.prediction_length` | 18 步 |
| 回歸訓練長度 | `stepwise_regression_steps` 未指定時為 36 步 |
| Gain rollout | YAML 的 35 步 warmup，加 18 步擾動後輸出，共 53 步 |
| CLI 群組參數 | `target-ratio=0.015`、`max-scale=50` |

歷史窗口、回歸訓練長度、Gain warmup 與擾動後評估長度是不同參數，不要把它們全部視為 `prediction_length`。

CLI 固定啟用的 Difficult 配對為 `H2S ← Air2`、`H2S ← T2` 與 `SO2 ← T2`；`SO2 ← Air2` 仍保留原有 Gain loss 的處理。`--target-ratio` 與 `--max-scale` 是 CLI 參數，不能任意新增同名 YAML 欄位並假設程式會讀取。

若 `exp_name` 尚未以 `_difficult_group_gradnorm` 或 `_difficult_group_gradnorm_no_ss` 結尾，CLI 會依 `--no-ss` 自動附加對應後綴；`--experiment-suffix` 可明確指定後綴。需要建立新實驗時，建議保留 `_seed42` 等 seed 標記及方法後綴，以便後續評估辨識。

這支程式會在 `training_files` 載入的每段資料上再次按 `test_data_split`、`valid_data_split` 建立時間切分；即使來源檔名包含 `_train_split`，仍會走這段內部切分。它不在訓練時建立 test loader。

### 既有 checkpoint 的梯度診斷

```powershell
python -m scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm --config $difficultConfig --no-ss --gradient-diagnostics-only --diagnostic-batches 1 --diagnostic-output-dir results/analysis/gradient_diagnostics_difficult_seed42
```

此模式載入 checkpoint 並量測梯度，不執行權重更新；需要的 checkpoint 預設依實驗名稱尋找，也可用 `--diagnostic-checkpoint` 明確指定。輸出包含 `gradient_component_detail.csv`、`gradient_cosine_detail.csv`、對應 summary，以及實際產生時的群組縮放紀錄。

## 推論

動態模型依 YAML 的 `inference_files` 執行：

```powershell
python -m scripts.prediction.predict --config $gainConfig
python -m scripts.prediction.predict --config $difficultConfig
```

必須使用與 checkpoint 一致的模型結構、變數順序及標準化統計。每個案例會依原有流程輸出預測 CSV、誤差指標、時序圖與 horizon 分析；部分彙整圖放在同一實驗結果資料夾。

階躍案例推論與 ANN 獨立測試：

```powershell
python -m scripts.prediction.predict_step_change --config $gainConfig --warmup-steps 0
python -m scripts.prediction.predict_tabular_v2
```

階躍程式目前讀取 `data/Claus_dynamic/step_change/in_training_distribution/`，產物位於 `results/pgin/<family>/<exp_name>/step_change_predictions/`。ANN 推論讀取已保留的 `tabular_test_dataset.csv`，產出 `tabular_predictions.csv`、`tabular_metrics.csv` 與 H2S／SO2 parity 圖。

## Gain 與 KCI 評估

2026-11-04 清理後，五組 Gain 與五組 SS-only 的動態 validation 可用 `scripts/prediction/validate_hyperparameters.py` 完整重跑並覆蓋原比較資料夾。內插為 R5-3／4／5，外插為 R5-27／28／29；AvgRelRMSE 採內外各 50%。KCI 沿用已核對來源雜湊的穩態 validation 結果，沒有用這六個動態案例重算。

KCI 為 ANN-reference 方向一致性，不能解讀為物理正確性或閉迴路驗證。

## 採樣與資料切分

### LHS 操作條件生成

```powershell
python -m scripts.sampling.generate_lhs_air2_t2_in_range
python -m scripts.sampling.generate_lhs_air2_t2_soft_ood
```

| 生成器 | 操作區域 | CSV 輸出 | 檢查圖 |
| --- | --- | --- | --- |
| `generate_lhs_air2_t2_in_range` | Air2 `140–300`、T2 `140–240`，其他輸入依生成器設定 | `data/sampling/LHS_air2_t2_in_range_perturbation.csv` | `results/sampling/LHS_Air2_T2_Local_Check_Plots/` |
| `generate_lhs_air2_t2_soft_ood` | 核心以外、生成包絡以內的分層採樣 | `data/sampling/LHS_air2_t2_soft_ood_perturbation.csv` | `results/sampling/LHS_Air2_T2_Soft_OOD_Check_Plots/` |

目前兩支生成器各輸出 5,000 筆操作條件；Soft-OOD 分為單變數 OOD 4,000 筆與 Air2、T2 同時 OOD 1,000 筆。CSV 包含 baseline 與擾動案例及其 metadata。重新執行會覆寫同名 CSV／圖表，生成器目前未固定 LHS 隨機種子，每次採樣不保證相同。

現有 `Hard_OOD_Check_Plots` 是保留的歷史圖表；目前 `scripts/sampling/` 沒有對應的 Hard-OOD 生成入口。

### 動態資料切分

```powershell
python -m scripts.preprocessing.split_dynamic_data
python -m scripts.preprocessing.split_dataset_partitions --config $gainConfig
```

兩支工具用途不同，依需求選擇，不需要依序全部執行：

- `split_dynamic_data` 讀取動態目錄中尚未含 `_split` 的 `Test_dataform_change_air2_R=*.csv`，按時間切為約 80%／10%／10%，寫出 train／valid／test 檔。
- `split_dataset_partitions` 讀取 YAML `training_files`，依 `test_data_split`、`valid_data_split` 寫出 valid／test 檔，不另寫 train。若輸入檔名已含 `_train_split`，新檔名會再附加 `_valid_split`／`_test_split`。

這些命令會在資料目錄寫入檔案。既有分割檔已齊全時，可直接使用，不必為執行推論或評估重新切分。

## 分析與繪圖

- [五個 Gain validation](results/visualization/gain005_gain2p5_gain25_gain250_gain2500_casewise_region50_validation_seed42/README.md)
- [五個 SS-only validation](results/visualization/ss_only_all_weights_casewise_region50_validation_seed42/README.md)
- [五個 Gain+SS validation](results/visualization/gain_ss_all_weights_casewise_region50_validation_seed42/README.md)
- [No-Gain、Gain=250、SS-only=2.5 的 test 比較](results/visualization/three_model_no_gain_gain250_ss2p5_test_seed42/README.md)
- [三家族最佳模型與 No-Gain：test KCI、誤差表及內外插圖](results/visualization/best_gain_ss_gainss_test_seed42/README.md)

三組超參數比較的權重均為 0.05、2.5、25、250、2500；逐案例比值採內外區域各 50%。各資料夾提供獨立重畫腳本；11/3 以前的 visualization 結果保留。

## 輸出與重現實驗

2026-11-04：訓練與推論產物依 YAML `output.results_dir` 寫入分類父目錄，再附加 `exp_name`。對應分類見 [Results 總覽](results/README.md)。Gain+SS 改為五組相同的 Gain／SS 權重；舊專用模型與結果已清除，請依 [Gain+SS 重跑指令](configs/pgin/gain_ss/README.md) 訓練。

以 Difficult-GradNorm 為例，主要產物如下，部分檔案依是否實際產生該項紀錄而輸出：

```text
saved_models/<exp_name>.pth
results/pgin/<family>/<exp_name>/
├─ zscore_mean.csv
├─ zscore_std.csv
├─ training_loss_curve.png
├─ kci_history.csv
├─ kci_consistency_curve.png
├─ ss_loss_history.csv
├─ gain_group_gradnorm_history.csv
├─ gain_pair_kci_history.csv
└─ final_gain_pair_kci_bar.png
```

最佳 checkpoint 依驗證 MSE 選擇；沒有可用驗證 loader 時依程式的訓練 loss fallback。權重檔存的是 `state_dict`，不包含完整 optimizer 與訓練狀態，不能當作完整的中斷續訓快照。

重現或交接實驗時，請一起保留：使用的程式入口與 CLI 參數、原始 YAML、實際 `exp_name`、資料版本與切分方式、seed、動態權重與標準化統計、ANN teacher 的權重與統計，以及獨立評估 CSV。訓練期間的 KCI history 與 held-out 評估結果用途不同，應分開記錄。

若只是檢查流程，請複製設定、縮短 epochs／steps／batch size 並另設實驗名稱；這類結果只用於確認流程，不作為正式模型比較。訓練中的即時圖可用 `training.pgin_runtime_plot: false` 關閉；這不會關閉訓練結束後的實驗結果輸出。

## 既有限制與常見問題

| 狀況 | 說明與處理方式 |
| --- | --- |
| 找不到 `src`、資料或 YAML | 確認工作目錄是專案根目錄，並使用目前 `scripts/` 與 `configs/pgin/` 的路徑 |
| 找不到動態 checkpoint／統計量 | 依 YAML `exp_name` 核對權重與兩份統計；新設定名稱不會自動取得舊模型產物 |
| checkpoint 的 tensor shape 不符 | 確認模型結構、`variables_num`、欄位契約與訓練時一致 |
| Gain 評估出現重複比較鍵 | 同 seed 的多個 Gain 變體分次與 No-Gain 比較 |
| 階躍推論／切分工具找不到預設 YAML | 原預設 `configs/transformer_layerwise_57var.yaml` 目前不存在，請明確傳入 `--config` |
| `plot_data11_concentration_surfaces` 缺少模組 | 目前缺少 `show_lhs_air2_tr2_surface` 來源模組；此入口尚無法獨立執行 |
| 工廠分布圖缺資料 | `plot_plant_simulated_distribution` 需要目前 checkout 缺少的 `plant_simulated_data/`、`Claus_plant_data/` 與 `*_converted.csv` |
| `configs/steady_state/tabular_mlp_claus.yaml` 的路徑不存在 | 該檔保留原有 `data/my_own_data/` 設定，不是 `train_tabular_v2` 的設定入口 |

`scripts/maintenance/annotate_difficult_gradnorm.py` 是針對舊版 1,488 行來源的歷史一次性工具，保留行數保護，不應對目前版本直接重跑。`check_ann_step_data_20261013.py` 執行時會寫入既定的 ANN 驗證結果目錄，請先確認原有產物是否需要保留。

目錄整理保留了歷史 provenance、日誌與備份，其中的舊路徑代表當時位置。舊程式快取保存在 `logs/legacy_python_cache_20261019/`；跨專案流程若仍引用原 `LHS/` CSV 路徑，需改為 `data/sampling/`。

其他入口與搬移對照見 [工作流程](docs/workflow.md)；設定、權重、統計量與程式位置見 [實驗索引](docs/experiment_index.md)。
