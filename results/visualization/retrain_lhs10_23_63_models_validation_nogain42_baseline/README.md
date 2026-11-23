# DG 95% 上限重訓後的 validation

63 個模型包含 48 個沿用的非 DG 模型與 15 個重新驗證的 DG 模型。各區域以 No-Gain seed 42 為共同基準，對 18 個同案例、輸出與時域的 RMSE 比值取幾何平均。KCI 是分別在穩態 in-range 與 soft-OOD 探點上計算的 ANN 參考方向一致率，與動態 RMSE 資料不同。R5-5 約 33.19% 的原始時間點超出核心範圍。

48 個沿用模型的輸出已對照舊報告雜湊並保留快照；先前記錄的 5 個歷史來源已不存在，因此不宣稱全部上游來源均可核對。新選模須依此 validation 執行；原 test 在重跑前仍是歷史結果。interactive_dom_validation.json 僅檢查新版 HTML 內嵌資料與 CSV 一致，未宣稱實際瀏覽器點擊測試。
