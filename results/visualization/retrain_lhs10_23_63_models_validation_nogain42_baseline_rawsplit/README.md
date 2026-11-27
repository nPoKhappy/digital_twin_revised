# Raw-split validation

63 個重新訓練模型使用相同 validation 案例與 ANN-reference 穩態探點重新推論；AvgRelRMSE 使用新 No-Gain seed 42 的相同案例、輸出、時域 RMSE 作基準，每區域對18個比值取幾何平均。KCI 和動態 RMSE 使用不同資料；KCI 衡量相對 ANN 的方向一致，並非直接證明 Aspen 真實方向。R5-5 約 33.19% 原始時間點超出核心範圍。

動態預測使用已記錄的未來 decoder 操作變數；H1/H18 的視窗使用真實過去歷史，TS 從最初18點真實歷史遞迴至結束。此驗證不證明閉迴路表現。
