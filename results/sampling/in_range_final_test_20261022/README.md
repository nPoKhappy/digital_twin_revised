# 新增 in-range 最終測試操作條件

[操作條件 CSV](../../../data/sampling/LHS_air2_t2_in_range_final_test_20261022.csv) 由 `scripts/sampling/generate_lhs_air2_t2_in_range.py` 的隔離副本產生。原程式、舊抽樣資料與既有穩態資料均保留；僅調整執行副本的輸入／輸出路徑、圖檔位置及兩個隨機種子，抽樣、擾動、裁切與列數整理邏輯不變。

## 資料組成

- 500 個 LHS 基準操作點，加上 4,500 列局部擾動，共 5,000 列。
- 17 欄物理輸入與 6 欄追蹤資訊：`case_id`、`base_id`、`case_type`、`perturbed_mv`、`direction`、`delta`。
- 原生流程先建立每個基準點的 10 列正負擾動，再抽取 4,500 列擾動，因此部分基準點的擾動組不完整。
- 這不是 5,000 個獨立基準點。同一 `base_id` 屬於同一操作點家族；若日後切分資料，應整組保留，避免逐列隨機切分。
- LHS seed：`3325265741`；擾動列抽取 seed：`1434680491`。

| 變數 | Aspen tag | 抽樣範圍（含邊界） | 局部擾動 |
| --- | --- | --- | --- |
| Air2 SP | B33.SPo.SPo | 140–300 | ±2.5 |
| T2 SP | B20.SPo.SPo | 140–240 | ±2.5 |
| 酸氣流量 | B34.SPo.SPo | 120.5–160.5 | ±0.5 |
| 酸氣溫度 | ACIDGAS.T.T | 82.6–84.6 | ±0.2 |
| 酸氣壓力 | ACIDGAS.P.P | 1.66–1.69 | ±0.005 |

其餘 12 個輸入沿用原程式，由 `data/Claus_dynamic/step_change/in_training_distribution/air2_190_t2_155_t2_change_-5.csv` 的中位數設定。擾動超出範圍時沿用原生邊界裁切；其中 38 列落在 Air2／T2 邊界。若評估器採用 `140 < Air2 < 300`、`140 < T2 < 240` 的嚴格範圍，模擬前可用操作條件為 4,962 列，實際可評估數仍須依模擬完成狀態與變數有效性判定。

## 檢查與產物

[generation_validation.json](generation_validation.json) 記錄檢查結果：5 個變數均通過 500 個 LHS 分層與種子重現；所有數值有限、範圍正確、case ID 唯一；擾動僅改變一個輸入，delta 與方向一致；固定值等於來源中位數。

將五個變動輸入四捨五入至小數點後 8 位後，新 CSV 內無重複條件，與 15 份既有來源亦無重複條件。逐檔結果見 [overlap_checks.csv](overlap_checks.csv)，範圍與邊界列數見 [range_checks.csv](range_checks.csv)。這是操作條件重複檢查，不代表已完成物理模擬。受保護的 16 份原始程式／資料 SHA-256 均未改變。

[generation_manifest.json](generation_manifest.json) 保存種子、範圍、擾動量、固定值、套件版本及雜湊；[generation.log](generation.log) 保存執行紀錄；[LHS_Air2_T2_Local_Check_Plots](LHS_Air2_T2_Local_Check_Plots) 保存原程式產生的 17 張基準點分布圖。Air2 與 T2 圖已人工檢視。

## 下一步

目前僅產生 **Aspen 操作條件輸入 CSV**，尚未執行 Aspen 穩態模擬，也未完成 KCI 評估。CSV 不含模型所需的完整 63 個變數；須先在 Aspen 執行、保存完成狀態並匯出完整穩態資料，再接續既有 KCI 評估流程。

若這批資料作為最終 KCI test，先固定 ratio 與模型選擇，再評估這批資料。
