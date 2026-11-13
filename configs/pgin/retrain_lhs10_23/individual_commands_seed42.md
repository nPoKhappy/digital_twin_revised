# Seed 42：批次失敗後逐項續跑

僅在原批次已停止，且 `_batch_run.json` 的 status 為 `failed` 後使用。每次只選擇尚未執行、checkpoint 與結果資料夾都不存在的模型；不要同時啟動兩個訓練。

程式會核對原批次記錄的 teacher、scaler、動態與穩態資料雜湊及 config manifest，並為每個單項留下獨立紀錄。已完成或已有部分產物的模型會拒絕覆寫；若有部分產物，請先保留並檢查。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias no_gain
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain005
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain2500
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain250
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain25
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain2p5
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias ss005_only
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias ss2500_only
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias ss250_only
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias ss25_only
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias ss2p5_only
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain005_ss005
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain2500_ss2500
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain250_ss250
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain25_ss25
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias gain2p5_ss2p5
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias dg_ratio0p0025
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias dg_ratio0p005
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias dg_ratio0p01
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias dg_ratio0p015
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias dg_ratio0p02
```

各行應分別執行，依實際剩餘模型選擇。SS-only 與 Gain+SS 使用 `train_pgin_stepwise_gain_ss_loss`；No-Gain 與 Gain 使用 `train_pgin_stepwise_gain`；Difficult-GradNorm 使用自己的 trainer，並由 manifest 明確提供 ρ、max-scale、no-SS 與 YAML 的 11 份來源選項。
