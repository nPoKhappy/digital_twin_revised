# 依更新後 validation 選模的三 seed test

使用更新後 validation 資料，對四個模型家族各選一組超參數：Gain 25、SS-only 2.5、Gain+SS 25/25、Difficult-GradNorm ρ=1.5；No-Gain 作為基準。五組各有 seed 42、43、44，共 15 個模型。選擇規則與所有候選值見 `validation_selection_plan.json` 和 `validation_selection_candidates.csv`。

選模只使用 validation：每 seed 的誤差為 `sqrt(內插 AvgRelRMSE × 外插 AvgRelRMSE)`，KCI 為內、外插各 50%；各指標再對三 seed 算術平均。各家族內對候選的誤差和 KCI 差距分別做 min-max 正規化，選至低誤差、高 KCI 理想點等權歐氏距離最小的超參數。此結果取決於候選集合，不能比較不同家族的距離。

12 個 No-Gain、Gain、SS-only、Gain+SS 測試結果逐檔驗證並沿用；新選中的 DG 1.5 三個 seed 重新推論並計算 test KCI。每個沿用模型的 `complete.json` 記錄原始來源、原 marker 和 protocol SHA，以及新 protocol 綁定；其預測數據與原輸出逐檔相同。沒有重新訓練模型。

動態 test 為 R5、R5-1、R5-2、R5-6 的 test split 和 R5-19～R5-26；每個 case、輸出及 H1/H18/TS 以相同項目的 No-Gain seed 42 RMSE 為分母。內、外插各對 24、48 個比值取幾何平均。KCI 由 in-range 4,385 和 soft-OOD 2,692 個有效穩態 test 探點、正負方向與 18 個預測步對 ANN 參考增益方向的比較得到，ANN 增益絕對值須至少 1e-5。KCI 不是 Aspen 真實增益的直接驗證。

先前 validation 歷史備份中五個原始來源檔已不存在，先前 test protocol 原本引用的三個 validation 摘要也已由新版取代；具體 SHA 和目前狀態記錄於 `run_protocol.json`。現有 validation 數值、仍存在的 NPZ 與其他來源已檢查，但無法宣稱已刪除的五個來源仍可追溯。

這些 test 案例和探點已使用過，並非全新的盲測；動態預測使用已記錄的未來 decoder 輸入，屬條件式預測。三 seed 共用資料及 ANN，描述的是動態模型訓練變化。
