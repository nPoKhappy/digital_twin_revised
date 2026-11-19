# Difficult-GradNorm 新增比例

這批保留已完成的 ρ = 0.015 舊模型，新增 ρ = 0.15、1.5、15、150，各自訓練 seed 42、43、44。每個新設定使用與先前三 seed 相同的 ANN、動態訓練資料、train-only scaler、gain 權重及困難配對；穩態值損失為 0。比例必須由 `config_manifest_dg_expanded.json` 的 `--target-ratio` 傳入，不能只執行 YAML 而省略此 CLI 參數。

為讓較大的目標比例有較多調整空間，四個新比例均使用 `--max-scale 10000000`。這是梯度群組倍率上限，不保證實際梯度比例會達到目標；是否飽和必須檢查訓練輸出的 `gain_group_gradnorm_history.csv`。已完成的 ρ = 0.015 模型使用上限 50；與新增設定比較時，同時改變了 ρ 和倍率上限，不能直接將差異只歸因於 ρ。

兩個 terminal 可同時執行以下兩個群組；每組各自依序訓練 seed 42、43、44，每個 seed 執行該組兩個比例：

```powershell
# Terminal 1: ρ = 0.15、1.5
.\.venv\Scripts\python.exe -u -m scripts.training.run_expanded_difficult_gradnorm --group low

# Terminal 2: ρ = 15、150
.\.venv\Scripts\python.exe -u -m scripts.training.run_expanded_difficult_gradnorm --group high
```

各自加上 `--check-only` 可先檢查三個 seed，不會啟動訓練。每個 seed 有分開的 `_dg_expanded_low_batch_run.json` 與 `_dg_expanded_high_batch_run.json`；原子鎖會拒絕同一個模型被其他群組或舊單 seed 指令重複執行。遇到已存在的批次紀錄、模型產物或鎖時會停止，不會覆蓋，請先檢查是否仍有程序執行以及批次的 `status`、`active` 和 `completed`。如果程序被強制終止，殘留鎖也必須在確認無程序運行後才處理。

也可沿用先前的單 seed 執行方式，每次依序執行全部四種比例：

```powershell
.\.venv\Scripts\python.exe -u -m scripts.training.run_expanded_difficult_gradnorm --seed 42
.\.venv\Scripts\python.exe -u -m scripts.training.run_expanded_difficult_gradnorm --seed 43
.\.venv\Scripts\python.exe -u -m scripts.training.run_expanded_difficult_gradnorm --seed 44
```

結果輸出於 `results/pgin/retrain_lhs10_23/seed<seed>/<exp_name>/`，模型輸出於 `saved_models/<exp_name>.pth`。執行器會核對原始 seed 批次記錄的 ANN 和訓練來源 SHA、設定及 CUDA，並拒絕覆蓋現有新批次紀錄或模型結果。

舊 `config_manifest.json` 和 `_batch_run.json` 保存過去實驗記錄，不含本批新設定；請使用新 manifest 與新 runner。由於使用者要求移除其他舊 ρ 設定與產物，原舊 manifest 仍指向已移除的歷史檔案，不代表這些結果仍可重現。

截至本次整理，四個新比例各自的 seed 42、43、44 checkpoint 與結果均已產生，並已納入擴充 DG validation。seed 44 的部分原始群組批次曾中斷，後續將受影響的模型從頭訓練；原批次 JSON 的 running 狀態保留為當時紀錄，不能單憑它判定目前仍在執行或所有續跑均已完成。請參照實際 checkpoint、結果與驗證報告核對。
