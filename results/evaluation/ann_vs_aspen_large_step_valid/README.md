# ANN 與 Aspen 穩態 step Gain 方向比較

以 Case 編號連結原始輸入與配對表，逐欄核對 17 個輸入。僅兩端均為 Run Completed 且輸出有效的配對納入。

主要結果是 Aspen 與 ANN 的 (擾動後濃度－基準濃度)/帶號 ΔMV 之嚴格正負號比較，未排除近零 Gain；`gain_comparisons.csv` 保留逐組實際數值及零值分類。`mismatches.csv` 也包含零值，請以分類欄辨別。

`local_gradient_summary.csv` 額外比較 ANN 基準點局部導數與 Aspen 指定步長的有限差；`local_gradient_kci_threshold_summary.csv` 套用 ANN teacher |gain| >= 1e-5，是 KCI teacher 條件下的補充結果，不是 Transformer KCI。

`coverage_by_stratum.csv` 記錄以基準點為分區的完成數與比例；`pair_status.csv` 保留所有失敗組。若有跨區資訊，`pair_status_by_region_transition.csv` 與 `summary_by_region_transition.csv` 分別記錄兩端分區、配對狀態與方向一致率。

`aspen_gain_threshold_sensitivity.csv` 的 Aspen 增益門檻僅供敏感度比較，不宣稱是模擬器的可靠性門檻。成功狀態不保證獨立檢查過收斂，且不驗證動態過程或閉迴路。
