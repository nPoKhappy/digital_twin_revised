# LHS 10–23 ANN 重訓後的動態模型設定

每個 seed 有 21 份設定：No-Gain 一份、Gain 五份、SS-only 五份、Gain+SS 五份、Difficult-GradNorm 五份。前三組權重沿用原設定的 0.05、2.5、25、250、2500；Gain+SS 採相等權重。Difficult-GradNorm 固定 Gain 權重 0.05、SS 權重 0，並分別以 CLI 指定目標比例 ρ = 0.0025、0.005、0.01、0.015、0.02，最大縮放 50。0.0025 是新探索值；舊結果在新 ANN 下不能保證有相同的梯度飽和趨勢。三個 seed 皆共用同一個重新訓練的 ANN teacher。

這些 YAML 沿用既有動態訓練與模型設定，另設定 `training.scaler_fit_scope: train_only`，使動態標準化統計只使用實際可建立訓練視窗的訓練資料。動態訓練的穩態來源保持原頂層 wildcard：`data/Claus_steady_state/lhs_generated_dynamic_ss_data*.xlsx`；這與 ANN 自身的訓練／驗證／測試切分不是同一個邊界。ANN 自身保留的 test 集不能據此稱為整個動態系統未見過的資料。

先完成 ANN 重訓，確認新的 checkpoint、scaler、`training_result.json` 與 `training_protocol.json` 已安裝在現有 teacher 預設路徑。批次會核對完成標記、checkpoint／scaler 與報告中的雜湊，以及 11 份穩態來源與 ANN 訓練紀錄的一致性；第一個正式批次建立跨 seed 的共用來源紀錄，後續 seed 若使用不同 teacher 或資料會停止。從專案根目錄執行全部 seed42 動態訓練：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42
```

若只想檢查設定、來源和 CUDA 而不開始訓練，可在上述命令加上 `-CheckOnly`。批次若標記 `failed`，可依 [Seed 42 逐項續跑指令](individual_commands_seed42.md) 明確選擇尚無產物的模型；程式會比對原批次所記錄的來源雜湊，拒絕覆寫已有結果。

seed43、44 的設定已準備，可在需要時分別把 `-Seed 42` 改成 `-Seed 43` 或 `-Seed 44`。批次依序執行 21 個模型、失敗即停止；開始前檢查設定雜湊及輸出碰撞，執行期間檢查 teacher、動態資料與頂層穩態來源的雜湊，並將進度記錄於 `results/pgin/retrain_lhs10_23/seed<seed>/_batch_run.json`。若批次中斷或部分完成，請保留該紀錄及輸出，先檢查再安排後續執行。

新 checkpoint 名稱為 `saved_models/<exp_name>.pth`（精確名稱列於 manifest）；統計量和其他訓練產物在 `results/pgin/retrain_lhs10_23/seed<seed>/<exp_name>/`。不得將它們與舊 teacher、舊 scaler 或舊 KCI 數值混用；必須使用新 checkpoint 對應的動態 scaler 重新推論與評估。

這份設定和指令的建立本身不代表任何動態模型已開始或完成訓練。`config_manifest.json` 保存設定來源及雜湊。
