# 擴充 DG 權重後的驗證圖（共同 No-Gain seed 42 基準）

這份結果共 63 個動態模型、126 個區域座標。沿用歷史結果中 48 個 No-Gain／Gain／SS-only／Gain+SS 模型及 3 個 DG ρ = 0.015 模型；另外 12 個 DG 模型使用新 checkpoint 重新驗證。僅重新整理已完成的驗證輸出，未在本步驟重新訓練模型。

DG 的 ρ = 0.015 訓練時 max-scale = 50；ρ = 0.15、1.5、15、150 使用 max-scale = 10,000,000，因此不同 ρ 並非只有比例參數不同，不能將差異單獨歸因於 ρ。此設定保留於 `model_parameter_metadata.csv` 和 provenance 中。

橫軸為 ANN-reference KCI（穩態 in-range／soft-OOD probes），縱軸為對應動態案例各 18 項相對 No-Gain seed 42 RMSE 的幾何平均；內插 R5-3/4/5、外插 R5-27/28/29，兩軸使用不同資料。R5-5 約 33.19% 原始時間點超出核心範圍。KCI 為 ANN 增益方向一致性，不是 Aspen 物理增益或閉迴路驗證。

兩種 4×2 圖及對應分頁各使用一致的全圖 X、Y 範圍。linear 是線性尺度；expanded_y 在 1.35 以下展開，沒有省略數值，但距離比例於 1.35 改變，需在圖說說明。No-Gain 參考點跨家族重複顯示；真實座標不作抖動，部分點可能重疊。

歷史版本固定保留於 `retrain_lhs10_23_63_models_validation_nogain42_baseline_previous_before_dg_expansion`；新 DG 原始驗證保留於 `retrain_lhs10_23_expanded_dg_validation`。此前選定的 test 評估仍是歷史結果，並未因本報告重選或重跑。舊 test 原本指向 `retrain_lhs10_23_63_models_validation_nogain42_baseline/` 的歷史基準檔，現應對應到相同檔名的上述備份目錄；`historical_path_mapping_verification.json` 逐項記錄對應路徑並核對原始 SHA，舊 test 檔案沒有修改。

重建到新空資料夾：`.\.venv\Scripts\python.exe -m scripts.analysis.build_expanded_dg_validation_report --output-dir results/visualization/retrain_lhs10_23_63_models_validation_nogain42_expanded_staging`。
