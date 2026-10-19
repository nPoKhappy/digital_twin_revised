# 既有 step-change 資料與 ANN teacher 的方向核對

分析日期：2026-10-13。

## 結論

指定資料夾包含 8 組名目操作條件、16 條 step-change 軌跡。每條軌跡分別比較 H2S 與 SO2，共 32 個案例／輸出比較。擾動前操作點的 ANN 局部偏導數符號，與記錄資料的長時間有限步階反應方向一致：32/32。套用目前訓練的 teacher 絕對 gain 門檻 1e-5 後，有效項為 29/32，其中 29/29 同號。

此結果支持指定案例中的方向先驗，不證明 ANN 導數大小普遍準確，也不證明每個暫態時間步都必須與穩態導數同號。

## 方法

- 輸入：`data/Claus_dynamic/step_change/in_training_distribution/*.csv`，共 16 檔。
- ANN：`saved_models/Tabular_MLP_5in_2out_QV.pth`，使用 `results/Tabular_MLP_New` 的既有標準化統計值及評估程式的 teacher 載入／求導函式。ANN 採 eval 模式，沒有更新參數。
- 各檔在原始零起算列索引 400 發生主要 MV step change；15 檔有 1440 列，1 檔有 1400 列。
- 擾動前：最後 100 列 pre-step 平均，即列索引 300–399。
- 擾動後：檔案最後 100 列平均。
- 記錄有限步階斜率：`(post_mean_output - pre_mean_output) / (post_mean_MV - pre_mean_MV)`，負擾動也保留分母符號。
- ANN 局部偏導數：在 pre-step 五輸入平均值求 physical-output 對 physical-MV 的偏導數。
- 另計算 ANN 固定其他輸入的有限步階斜率，以區分局部導數與有限擾動的差異。
- 原始品質濃度未取 log；比較雙方的物理輸出尺度一致。
- 原始檔未修改；來源與 ANN artifacts 的 SHA256 存於 `method_and_provenance.json`。

## 配對結果

| 配對 | 記錄正／負方向案例 | ANN 局部符號一致 | 局部偏導數 vs. 記錄步階斜率的絕對相對誤差中位數 | 有效門檻內一致／有效 |
|---|---:|---:|---:|---:|
| air2 → H2S | 1／7 | 8/8 | 16.52% | 8/8 |
| air2 → SO2 | 8／0 | 8/8 | 8.09% | 8/8 |
| T2 → H2S | 4／4 | 8/8 | 21.62% | 8/8 |
| T2 → SO2 | 4／4 | 8/8 | 23.31% | 5/5 |

這些相對誤差比較的是局部偏導數與有限步階斜率，不是對精確局部真值的導數誤差。最大值為 158.89%，出現在 air2=240、T2=170、T2 +10 的 SO2；該點 ANN gain 低於目前訓練門檻。

特別案例：air2=270、T2=155、air2 -5 的 H2S，記錄有限步階 gain 為 +1.00157e-5，ANN 局部偏導數為 +1.17062e-5。這是 air2 → H2S 的正號案例，因此 8/8 並非全部只猜負號的結果。

## 穩健性與資料品質

- 使用 50、100、200 列作為 pre/post 平均窗口，全部 96 個窗口／案例／輸出組合的記錄方向相同，且均與 100 列 baseline 的 ANN 符號相同。
- 以尾端 100 列的前半與後半平均差，除以完整 pre/post 輸出變化，最大為 4.35%。這是尾端漂移描述，並非獨立求解器穩態收斂證明。
- 四個 air2 -5 案例的非目標 T2_SP 在前後平均值間分別變動約 +0.0232、+0.1228、-0.0349、+0.2019 °C；部分案例另有小幅酸氣流量變動。不能將記錄步階斜率稱為其他輸入嚴格固定下的精確偏導數。
- air2 → SO2 的八個參考方向均為正，此配對沒有測到負號區域。
- 實際 air2 擾動為 +10 或 -5，即約 +0.225 或 -0.112 倍 Transformer 訓練標準差；T2 為 +10 或 -5，即約 +0.467 或 -0.233 倍。這不是四種配對的完整 ±0.5σ 驗證。
- 目前 teacher 訓練原始碼讀取 LHS Excel 而非這些 step CSV，但 checkpoint 未附完整訓練來源 manifest，未證明操作點與 teacher 訓練資料完全無重疊。

## 暫態方向檢查

以 step change 後每 10 個原始列取一次品質中位數，共 18 個 block；各 block 相對 pre-step 平均值計算方向。這是記錄資料的描述性檢查，不能等同原有 Transformer-vs-ANN KCI，也不能把 block 當成獨立案例。

- 全部同號 block 為 564/576。
- T2 → H2S、T2 → SO2 均為 144/144。
- air2 → H2S 為 134/144；air2 → SO2 為 142/144。
- air2=270、T2=155、air2 -5 的 H2S，在 post-step block 2–5 出現與 ANN 穩態參考方向相反的差異。該檔也有非目標 T2 設定值漂移，因此不能僅由此斷定存在固有逆向反應。
- 少數很早期反向差異沒有超過擾動前標準差的五倍；`effect_exceeds_5_pre_std` 僅是描述性篩查，並非正式統計檢定或求解器誤差界。
- 原始列的物理時間間隔未另行驗證，因此圖與 CSV 使用列索引／block，而不直接標為分鐘。

## 可支持的論文敘述

既有 in-range step-change 案例提供 ANN 方向先驗的案例層級佐證。在八組名目操作條件、十六條軌跡的三十二個案例／輸出比較中，ANN 在擾動前操作點的局部敏感度符號，均與記錄資料的長時間有限步階反應方向一致；此方向判斷對三種前後平均窗口保持穩定。此結果不代表導數幅值完全準確、所有操作區域的方向均已驗證，或所有暫態時間步都應與穩態方向相同。

## 輸出

- `gain_comparison_detail.csv`：32 項方向、幅值、輸出值誤差及漂移明細。
- `gain_comparison_summary.csv`：四個配對的彙總。
- `case_input_quality.csv`：16 檔輸入變動與 step 位置。
- `window_sensitivity.csv`：96 項窗口敏感度比較。
- `first18_blocks_direction.csv`：576 項描述性暫態比較。
- `gain_comparison.png`：記錄步階斜率與 ANN 局部／有限步階斜率比較。
- `recorded_responses_all_cases.png`：所有原始品質軌跡與 pre/post 均值、ANN 輸出。

重現指令（從專案根目錄）：`python -X utf8 logs/check_ann_step_data_20261013.py`。分析使用既有 checkpoint，不重訓模型，也不啟動 Aspen。
