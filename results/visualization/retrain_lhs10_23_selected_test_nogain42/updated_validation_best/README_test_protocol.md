# 95% 上限 DG 重訓後：validation 選模與三 seed test

本資料夾的 DG 使用以歷史訓練梯度第 95 百分位估算的倍率上限重新訓練後的 checkpoint。先在 validation 上依既定規則選模：每個 seed 的誤差是內外插 AvgRelRMSE 的幾何平均，KCI 為內外插算術平均；對三 seed 各自取平均後，在每個家族內正規化誤差及 KCI 差距，選距離「最低誤差、最高 KCI」最近的候選。候選和計算結果見 `validation_selection_candidates.csv` 和 `validation_selection_plan.json`。

選中 No-Gain、Gain 25、SS-only 2.5、Gain+SS 25/25 和 Difficult-GradNorm ρ=0.15，每組都有 seed 42、43、44。原有四個非 DG 家族共 12 個 test 預測在驗證來源及各檔雜湊後沿用；只有 3 個新 DG 模型重新推論。所有模型使用相同 ANN teacher，沒有再訓練或更新參數。

動態 test 包含 R5、R5-1、R5-2、R5-6 的 test split，以及 R5-19 至 R5-26。各 case、輸出與 H1/H18/TS 的 RMSE，除以對應的 No-Gain seed 42 RMSE；內、外插分別對 24 和 48 個比值取幾何平均。KCI 使用另外的穩態探點：in-range 4,385、soft-OOD 2,692；計算兩個擾動方向及 18 個預測步的 ANN-reference 方向一致率。

`independent_npz_audit.json` 由已保存的預測 NPZ 獨立重算所有 1,080 個 RMSE、6,480 個 horizon 與 120 個 KCI 配對。數據、設定、checkpoint 與 run record 的來源雜湊保存於 `run_protocol.json`；validation 圖表版面修正後，已核對所有 20 個候選與數值不變並重綁 provenance。

這些 test 案例及探點先前已被使用，不是新的盲測；動態預測使用已記錄的未來 decoder 輸入。KCI 衡量模型與 ANN 參考方向的一致率，不能單憑它證明 Aspen 真實增益方向或閉迴路表現。
