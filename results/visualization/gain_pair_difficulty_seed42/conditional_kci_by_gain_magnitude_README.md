# 三個訓練 seed 的 ANN 增益大小分組 KCI

圖上各點為 seed 42、43、44 的 KCI 算術平均，誤差棒為三個 seed 的樣本標準差（mean ± 1 SD，n=3，ddof=1），不是 bootstrap 信賴區間。三個動態模型共用相同 ANN 與相同 test 穩態探點；標準差描述動態訓練 seed 的變異。新版 validation 選模：{'gain': 'gain25', 'ss_only': 'ss2p5_only', 'gain_ss': 'gain25_ss25', 'difficult_gradnorm': 'dg_ratio0p15'}，另含 No-Gain。

每個區域、輸出與 MV 配對分別先篩選 |ANN 增益| ≥ 1e-5 的有效操作點，再依 |ANN 增益| 的 25%、50%、75% 分位數分為 Q1–Q4。沿用舊圖方法，以已儲存 teacher scaler 的 σ_MV / σ_target 標準化增益絕對值，使用 searchsorted side=right；若與邊界相等，分配至較高分組。界線對所有模型與 seed 相同；原始增益絕對值與正的標準化倍率在同一配對中排序相同。每個操作點合併正、負擾動及各 18 預測步共 36 次比較，動態增益為零計為方向不一致；每 seed 對各組全部有效點計算 KCI。逐 seed 數值見 `conditional_kci_by_gain_magnitude_per_seed.csv`，均值和 SD 見 `conditional_kci_by_gain_magnitude.csv`，分位數及各組點數見 `gain_magnitude_quartile_edges.csv`。

mean ± SD 可延伸至 0% 以下或 100% 以上，但實際 KCI 都在 0–100%。test 穩態探點先前已使用，並非全新盲測。KCI 是對 ANN 參考方向的一致率，不能單憑它驗證 Aspen 真實增益方向。前次 validation 歷史備份五個來源已刪除，選模與 test protocol 記錄此追溯限制；仍存的來源及預測陣列已核對。

重建：`.\.venv\Scripts\python.exe -m scripts.analysis.plot_updated_test_teacher_magnitude_three_seeds`，預設輸出 staging 資料夾供覆核。
