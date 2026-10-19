from pathlib import Path
import ast
import io
import tokenize
from datetime import datetime

path = Path('scripts/training/train_pgin_stepwise_gain_difficult_group_gradnorm.py')
original_bytes = path.read_bytes()
original = original_bytes.decode('utf-8')
lines = original.splitlines(keepends=True)
assert len(lines) == 1488, 'Source changed; recheck annotation anchors before editing.'

# Keys are original line numbers. Comments inherit that line's indentation.
notes = {
    3: '''閱讀指南：這支程式同時處理三件事。
1. 用動態歷史資料訓練 Transformer 的逐步預測（MSE）。
2. 可選：讓固定操作點的預測維持在 Excel 穩態值（SS loss）。
3. 用凍結 ANN 的局部 gain 正負號，約束 Transformer 的擾動反應方向。
   Difficult-GradNorm 再調整指定配對、指定 teacher 符號群組的梯度強度。
閱讀順序：資料讀取 → main 設定 → 每批次 MSE/SS/Gain → 群組縮放 → 更新模型。
常用縮寫：MV＝操作變數；QV＝品質輸出；SS＝穩態；AR＝把預測回填歷史。
張量符號：B＝本批穩態點數；W＝歷史長度；T＝用於 Gain 的時間步數；
Q＝品質輸出數；M＝受擾動的 MV 數；z＝標準化空間；p＝反標準化後的值。
注意：H2S/SO2 在 Transformer 端先取 log，因此名稱帶 p 不一定已還原濃度。
以下只新增說明，保留原本演算法、預設值、輸出與命令列介面。

套件分工：os/sys/glob 處理路徑；argparse 讀命令列；yaml 讀實驗設定。
random/NumPy/PyTorch 管理隨機與數值；pandas 處理表格；tqdm/plt 顯示進度與圖。''',
    13: '''指定 CUDA 的 attention 計算路徑，避免不同加速實作造成結果差異。
這三行是全域後端設定，並非凍結模型，也不代表所有 GPU 運算都已保證可重現。''',
    24: '''將專案根目錄加進 import 搜尋路徑，讓直接執行 scripts/training/ 下的腳本也能匯入 src。''',
    31: '''從命令列執行時，main 會收到這八份既定 LHS 訓練來源作為 override。
此處是訓練資料；與 evaluate_gain_in_out_distribution.py 的 held-out 測試來源分開。''',
    35: '''每個 tuple 的順序是（輸出 target，操作變數 MV）。
這三組是命令列入口指定要做群組梯度縮放的配對；SO2 ← air2 仍計算 Gain loss，
但會走 preserved_gain_loss，維持原本的全體平均權重。''',
    45: '''把 Aspen 匯出的 tag 改成動態資料與模型共用的欄位名稱。
字典左邊是原始名稱，右邊是程式名稱；只改欄名，不在這裡轉換數值或單位。''',
    68: '''ANN teacher 的網路結構須與已儲存的 checkpoint 相符；本程式不訓練它。
nn.Module 提供參數管理、train/eval 模式與 state_dict 載入功能。''',
    71: '''__init__ 建立層與參數；super(...) 先初始化 PyTorch 的父類別。''',
    73: '''Sequential 依列出的順序傳遞資料：輸入 → 64 → 32 → 輸出。
Linear 做線性映射；GELU 提供非線性；BatchNorm/Dropout 的行為會隨 train/eval 改變。
載入 teacher 後使用 eval()，所以 Dropout 停用，BatchNorm 使用已存的統計值。''',
    87: '''呼叫 mlp_model(x) 時會進入 forward：x 為 [B, input_dim]，回傳 [B, output_dim]。''',
    90: '''穩態讀檔入口：接受單一路徑、glob pattern，或多個來源組成的 list/tuple。
回傳清理過的 DataFrame；每一列代表一個穩態操作點，並非相鄰時刻。''',
    93: '''isinstance 判斷是否已是多來源容器；單一字串先包成 list 才能統一迭代。
glob 展開萬用字元；sorted 固定各 pattern 的檔案順序。''',
    100: '''dict.fromkeys 去除重複路徑並保留首次出現的順序，避免同一份資料重複載入。''',
    109: '''Excel 第三列是欄名（header=2 為從零起算），下一列是單位列，故再略過一列。''',
    116: '''如果有 Aspen 執行狀態欄，只保留成功完成的模擬列。''',
    119: '''只替換實際存在的欄名；重新命名後若有重複欄，保留第一個。
後續 required_cols 檢查確保 Transformer 歷史、MV 與 ANN 輸入都齊全。''',
    129: '''選取需要的欄位並轉成數值；無法轉換的內容變 NaN，之後統一刪除該列。''',
    131: '''這裡只替 air 做單位換算；air2_SP 不經過這項換算。''',
    136: '''合併來源並重建連續列索引；dropna 刪除任一必要欄位缺值的操作點。''',
    142: '''H2S/SO2 與動態訓練一致，先裁到至少 1e-6，再取自然對數。
因此 df_raw 的品質欄位實際上是 log 濃度；其他欄位仍是原單位。''',
    146: '''列出實際讀到的檔案、有效列數與丟棄列數，方便追查訓練資料來源。''',
    154: '''將抽到的穩態操作點轉成 Encoder 所需的 W 步歷史，以及該點的操作條件。
回傳 historical_dfs（B 份 [W, 欄位數] 表格）與 ss1_p_df（B 列起始穩態點）。
de_mv/all_cols/std_all/target_mvs 目前保留在介面中，函式本體未使用它們。''',
    157: '''優先使用外部指定 indices；否則隨機抽樣（np.random.choice 預設可重複）。
正式 main 使用 next_steady_state_indices 的排列索引，而非這裡的隨機抽樣分支。''',
    168: '''repeat_rows_as_history=True：把同一個穩態點重複 W 次，模擬固定操作的歷史。
False 分支取連續 W 列；對 LHS 表格而言這只是列切片，不保證是真實時間歷史。''',
    176: '''每段歷史的最後一列作為 baseline 操作點；串接後列順序與 historical_dfs 相同。''',
    179: '''依 cols 的順序計算 z=(值-mean)/std，再轉成指定裝置上的 float32 張量。
mean/std 必須來自使用這個張量的模型；teacher 與 Transformer 使用不同統計值。''',
    187: '''梯度診斷使用「參數清單的索引」分組，不複製參數或改變模型結構。''',
    190: '''all 包含所有參數；其他群組依模型參數名稱前綴區分 Encoder/Decoder/輸出層。
enumerate 同時提供索引與（名稱，參數）；這裡的 _ 表示不需要使用參數本身。''',
    202: '''計算指定參數群的梯度 L2 norm：sqrt(sum(g**2))；None 表示該參數未參與此 loss。
detach() 切斷診斷運算的計算圖，float(...cpu()) 將結果變成一般 Python 數字。
因此回傳值用於紀錄或計算倍率，不會對「梯度 norm 本身」再求導。''',
    213: '''梯度 cosine＝內積 /（兩邊的 L2 norm 相乘）。
接近 +1 表示更新方向相近；負值表示相衝；零梯度或無可比較參數時回傳 NaN。
這是 loss 對參數的方向比較，不是 KCI 的輸入／輸出 gain 符號比較。''',
    237: '''診斷輸出：保留逐批次明細，再按 loss 分量與參數群彙總平均值。
此函式只整理已量到的梯度，不執行反向傳播或 optimizer 更新。''',
    250: '''只挑數值欄位計算平均，避免把分量名稱等文字欄位混入統計。''',
    255: '''groupby(...).mean() 的每筆輸入是診斷紀錄，因此是紀錄平均，並非按有效點數加權。''',
    272: '''訓練主程式。CLI 在檔案最下方組好參數，再呼叫這個函式。
*_override 的優先權高於 YAML；experiment_suffix 用來區分輸出實驗名稱。
gradient_diagnostics_only=True 會載入 checkpoint、量測梯度並提前返回，不更新權重。''',
    286: '''yaml.safe_load 將設定轉為巢狀 dict；config['training'].get(key, default) 在缺值時用預設。''',
    290: '''同時設定 Python、NumPy、PyTorch 的種子，並指定 cuDNN 的確定性／搜尋行為。''',
    299: '''exp 決定 checkpoint 與 results 子目錄；只有 use_pretrained=True 才載入預訓練權重。
微調模式也可另外指定 finetune_exp_name 與較小的 learning rate。''',
    310: '''設定要求 cuda 且機器有可用 GPU 才用 GPU；否則回退 CPU。''',
    319: '''variable_selection 決定資料契約：de_mv 是 Decoder 的操作變數，y_sv 是預測輸出，
en_mv_and_sv 是 Encoder 歷史欄位；列順序會直接決定張量各維對應的變數。
63-variable 模式下，歷史新列的排列須與後面 cat([MV, prediction]) 一致。''',
    322: '''整理建立穩態歷史所需的所有動態欄位，去重時保留既有順序。''',
    330: '''把實際欄位數寫回 config，get_model 才能建立相符的輸入與輸出層大小。''',
    335: '''teacher 的輸入順序固定為 air2、T2、酸氣流量、壓力、溫度；輸出固定 H2S、SO2。''',
    346: '''YAML 可選要約束的品質與 MV；只保留 teacher 與 Transformer 都有的變數。''',
    354: '''把變數名稱轉成 Decoder 欄位索引，避免假設 MV 一定在某個固定位置。''',
    355: '''相容舊的 H2S-only 開關；有明確 gain_group_gradnorm_pairs 時優先使用明確清單。''',
    359: '''清單非空就啟用群組梯度縮放；h2s_group_* 是沿用的參數名稱，實際可涵蓋 SO2。
此處驗證每組 target/MV 都在本次 Gain loss 的資料契約中。''',
    373: '''interval 的單位是分鐘；W 會用「歷史分鐘數 / 取樣間隔」換算成步數。''',
    377: '''逐檔讀取動態資料，保留各檔邊界，後面不會跨檔拼接出時間窗口。''',
    382: '''中位數降採樣：每 interval 列算一次完整窗口中位數，再每 interval 列取一筆。
關閉 use_median 時改成直接抽取每隔 interval 列的資料。''',
    389: '''先把品質變數取 log，再與其他欄位一起做 z-score，順序須與推論一致。''',
    398: '''這裡以 training_files 載入的全部資料估計 mean/std；時間切分在後面才進行。
每份原檔仍另存於 all_dfs_log，合併表僅用來計算共用的標準化統計值。''',
    401: '''一般訓練會存下 Transformer 的 mean/std 供推論使用；診斷模式跳過這份輸出。''',
    409: '''teacher 使用自己的訓練統計值，不能直接套用 Transformer 的 mean_all/std_all。''',
    425: '''若動態統計值缺少 teacher 必需欄位，才用 teacher 統計值補入，不覆蓋既有欄位。''',
    430: '''逐份動態資料套用同一組 Transformer z-score，仍維持原本的檔案分界。''',
    435: '''三種時間長度分開：回歸預設 36 步、SS 使用 H_out、Gain 使用 warmup+H_out。
W 是輸入歷史長度，與「接下來要預測幾步」不是同一個參數。''',
    447: '''每份檔案按時間先留末段 test，再從前段留出 valid；本程式不建立 test_loader。''',
    452: '''Dataset 每個樣本回傳（過去 Encoder 歷史，未來 MV，未來輸出真值）。
DataLoader 組批後分別是 [B,W,F_en]、[B,regression_steps,F_mv]、[B,regression_steps,F_y]。''',
    460: '''ConcatDataset 合併各檔的樣本集合；shuffle 打亂窗口抽取順序，不打亂窗口內時間。''',
    473: '''LHS 穩態資料提供兩種監督：Excel 值供 SS loss，操作條件供 ANN 算 gain 符號。''',
    480: '''來源優先順序：函式 override → YAML steady_state_source_path → 預設 glob。
本檔 CLI 固定傳入上方八份 BASELINE_STEADY_STATE_SOURCES。''',
    490: '''依設定建立 Transformer；微調或診斷模式才會另外載入既有 state_dict。''',
    500: '''診斷 checkpoint 優先用命令列指定路徑；否則用本次 exp 對應的已儲存權重。''',
    510: '''teacher 參數不訓練，但稍後仍須對「teacher 輸入」求導，所以不能整段用 no_grad。''',
    512: '''teacher 輸出以它的 mean/std 還原成濃度；Transformer 品質輸出則還原到 log 濃度。
兩者 gain 數值尺度不同，本程式只取 teacher gain 的符號作為方向參考。''',
    517: '''先找共用 teacher checkpoint；不存在時才使用以實驗名稱命名的備援檔。''',
    522: '''eval() 控制 BatchNorm/Dropout；requires_grad_(False) 才是凍結 teacher 參數。
這兩個操作不會阻止稍後 requires_grad=True 的輸入取得梯度。''',
    525: '''steps_per_epoch 限制每個 epoch 只跑多少動態 batch，並非一定跑完整個 train_loader。''',
    532: '''唯一負責更新參數的 optimizer；它只拿到 Transformer 參數，沒有 teacher 或群組倍率。''',
    534: '''Gain 時間軸：先維持 baseline MV，再從 step_change_step 起維持擾動後的 MV。
目前對應 YAML 是 35 步 warmup + 18 步輸出，共 53 步；參數仍以實際設定為準。
設定中的 step 從 1 起算，Python 索引從 0 起算，故後面需要減 1。''',
    549: '''runtime 圖只在正常訓練開啟；診斷模式不產生這張訓練中即時圖。''',
    554: '''記錄最佳驗證 loss 與連續未改善次數，供儲存模型與 early stopping 使用。''',
    558: '''把輸出欄位的 mean/std 放到相同裝置；接近零的 std 換成 1，避免除零。''',
    569: '''SS loss 可指定輸出欄位清單；否則按 scope 使用全部 y_sv 或只使用品質欄位。''',
    587: '''建立不同欄位集合之間的索引映射：SS 對 Transformer，監測品質對 teacher。''',
    590: '''此處的 rollout 是預測回填 Encoder 歷史的自回歸運算，不是實際 Aspen/MPC 閉迴路控制。
SS 的值參考來自 Excel；ANN 在此主要負責提供 Gain 方向。''',
    591: '''SS 權重優先採用 override；--no-ss 會傳入 0，YAML 也可直接設定 ss_loss_weight=0。
權重為 0 時不貢獻總 loss，但下方仍會計算 SS rollout 與原始 SS loss。''',
    596: '''Gain 開關與權重分開：關閉 enable_gain_loss 時有效權重為 0，仍可能只監測 KCI。''',
    600: '''群組縮放需要正的 Gain 權重；target_ratio 與 max_scale 也先檢查合法範圍。''',
    606: '''只要訓練 Gain、監測 KCI 或設定 runtime 圖任一需要，就會計算 Gain rollout。''',
    607: '''teacher gain 太接近零時，正負號不穩定，因此低於此門檻的比較不納入 loss/KCI。''',
    608: '''Gain tail 是選來計算 loss/KCI 的時間段，起點至少是 step change 開始的那一步。
stepwise_gain_tail_start_step 可覆寫；沒有時使用 dynamic_gain_tail_start_step 與 step change 的較大值。
這是訓練端的可設定 tail；獨立評估腳本則固定取最後 18 步，兩者不要混為同一段。''',
    612: '''實際擾動量 Δu＝finite_diff_delta_std × 該 MV 的 Transformer 訓練標準差。
這裡讀取訓練 YAML，與評估腳本內獨立設定的 sigma 不會自動連動。''',
    618: '''將 tail 起點轉成合法的零起算索引；切片 end 不包含該索引，所以用 H_gain。''',
    620: '''穩態點用隨機排列依序取批；cursor 會跨 epoch 保留，走完整份資料才重新洗牌。''',
    623: '''此巢狀函式共用 main 的 ss_perm/ss_cursor，避免每批重新抽樣而遺漏某些穩態點。''',
    626: '''nonlocal 表示更新外層函式的狀態；末批只回傳剩餘點，不補抽成完整 batch。''',
    636: '''列出本次真正生效的權重、時間軸與配對；遇到設定疑問時先核對這份摘要。''',
    654: '''history_* 儲存跨 epoch 的結果；diagnostic_* 儲存診斷模式各 batch 的梯度明細。''',
    662: '''保留固定的參數清單順序，讓 autograd.grad 回傳的梯度能與診斷群組索引對齊。''',
    665: '''群組縮放只以 output_dense 的權重／bias 梯度 norm 作為量尺。
實際 total_loss.backward() 仍會更新整個 Transformer，不是只訓練輸出層。''',
    671: '''外部 helper 每步預測、計算 MSE，再把 detach 後的預測回填歷史；最後平均所有步的 loss。''',
    676: '''每個 epoch 重設統計計數；模型重新切回 train 模式，讓一般回歸使用訓練行為。''',
    684: '''配對統計形狀 [Q,M]；correct/total 計數，另外記錄 teacher 符號與動態正號比例。''',
    688: '''以正／負無限大初始化本 epoch 的 teacher 輸出極值，再由每批結果逐次更新。''',
    693: '''enumerate 提供批次索引；手動 break 確保只執行 step_limit 批。''',
    700: '''清除上一批累積在參數 .grad 的梯度；稍後只對本批的 total_loss 做一次更新。''',
    703: '''MSE 分支使用真實動態 batch，與接下來抽取的穩態 batch 不是同一組資料。''',
    708: '''穩態 batch 大小獨立於動態 batch；排列最後一批可能較短，所以使用 B_actual。''',
    715: '''暫時把標準化 helper 的 stdout 導到記憶體，避免每個穩態點都印出訊息。''',
    720: '''stack 把 B 份 [W,F_en] 歷史組成 [B,W,F_en]；同一點的 W 列目前都是相同穩態值。''',
    724: '''Decoder baseline 操作值 [B,F_mv] 保留原單位；做擾動時先改原單位，再轉回 z。''',
    726: '''同一穩態點改用 teacher 的 scaler 轉成 [B,5]；啟用輸入梯度以取得局部 gain。''',
    730: '''teacher 輸出先反標準化為濃度，再用 detach 的副本做 min/max 與非正值監測。
ann_monitor_clamped 是 <=1e-6 的計數；這幾行並未真的裁切 y_mlp_p_ss1。''',
    738: '''SS 的監督真值是 Excel 操作點在 Transformer z 空間中的值，並非 teacher 預測值。''',
    740: '''SS 分支固定所有 MV，從重複穩態歷史預測 H_ss 步，檢查模型是否維持固定點。''',
    746: '''unsqueeze(1) 增加時間維：[B,F_mv] → [B,1,F_mv]，Decoder 每次只預測一步。''',
    750: '''每次丟掉最舊歷史，加入 [MV, 當步預測]；detach 只切斷跨步歷史回填的梯度。
ss_pred_z_steps 仍保存未 detach 的當步輸出，因此每步 SS loss 都能訓練模型。''',
    761: '''沿 dim=1 疊成 [B,H_ss,SS輸出數]；expand_as 讓同一穩態真值對應每個時間步。''',
    765: '''Gain 沒有計算時先使用零 loss；記住目前模式，計算完 Gain 後再恢復。''',
    768: '''eval() 關閉 Dropout 等訓練隨機性，讓 baseline/plus/minus 的差異來自輸入擾動。
eval() 不等於 no_grad()；Gain 分支仍建立 Transformer 參數的反向傳播圖。''',
    770: '''teacher Jacobian 形狀 [B,Q,M]，每個元素是該操作點某品質對某 MV 的局部偏導。''',
    774: '''grad_outputs 只把當前品質輸出設為 1，其他設 0，以選出要微分的輸出欄位。''',
    777: '''autograd.grad 直接回傳對輸入的梯度，不寫進模型參數的 .grad。
create_graph=False：teacher 導數作固定參考；retain_graph=True：後續品質還要共用此圖。''',
    786: '''目前求到的是 d(濃度)/d(MV_z)；除以 teacher 的 MV std，換成 d(濃度)/d(MV原單位)。''',
    791: '''torch.sign 將 gain 轉為 -1/0/+1；後面只要求方向一致，不要求 gain 大小相等。''',
    793: '''對應 Transformer 的品質欄位索引；log_target_idx 只用於顯示時還原濃度。''',
    798: '''內部 helper 使用外層的 de_mv、mean_all/std_all，將 [B,F_mv] 原單位操作值轉回 z。''',
    809: '''一次完整 Gain rollout：baseline 與擾動案例都從相同初始歷史出發。
step_de_p_tensor=None 表示完全不擾動；否則只在指定 step change 之後改用擾動值。
回傳 [H_gain,B,Q] 的反標準化品質軌跡（H2S/SO2 仍在 log 空間），另附 MV z 與 debug。''',
    816: '''每次呼叫都 clone 初始歷史，避免 plus/minus 接續 baseline 的最終狀態。''',
    821: '''擾動前使用 baseline MV；擾動開始後維持 step MV，不是在每一步累加一次 Δu。''',
    829: '''反 z-score，但尚未 exp：本程式的 Gain loss 使用 log 品質差／原單位 MV 差。
對正濃度，log 是單調函數，局部導數的符號與濃度導數相同；數值大小則不同。''',
    832: '''collect_debug 才額外還原成濃度並裁到 1e-6；debug 結果 detach 後不參與 loss。''',
    839: '''AR 數值上仍使用先前預測；detach 讓後續 loss 不沿回填歷史一路反傳到前面各步。
保存在 gain_log_steps_local 的當步輸出仍有梯度，並非整條 rollout 都無法訓練。''',
    847: '''stack(dim=0) 把每步 [B,Q] 疊成 [H_gain,B,Q]，時間維放在最前面。''',
    850: '''baseline 操作點視為固定輸入；detach().clone() 隔開外部圖與後續原地擾動修改。''',
    857: '''只選定 tail 段參與 Gain loss；warmup 的預測仍會影響後續歷史與數值軌跡。''',
    864: '''每次只擾動一個 MV，其他 MV 保持 baseline；不同 MV 使用各自的 std 換算 Δu。''',
    869: '''clone 出兩份操作值再加減，避免把 baseline 本身或另一個方向一起改掉。''',
    874: '''分別重跑 +Δu、-Δu 的完整時間軌跡，選取與 baseline 相同的 tail 時間段。''',
    887: '''逐步的單邊有限差分：K+ = (y+ - y0)/Δu；K- = (y0 - y-)/Δu。
minus 的分子順序使兩方向都代表「輸出隨 MV 增加」的斜率；不先平均正負方向。''',
    892: '''先疊 MV 維，再 flatten(0,1) 合併時間與 batch：[T,B,Q,M] → [T*B,Q,M]。
接著疊正負方向為 [2,T*B,Q,M]；每個元素都會獨立做符號判斷與懲罰。''',
    896: '''teacher 每個操作點只算一次，repeat 按時間複製成 [T*B,Q,M]，與 flatten 的順序對齊。''',
    899: '''只在每個 epoch 第一批畫第 0 個穩態點的示意軌跡，供觀察數值反應。
圖中的 ANN 水平線是對擾動後操作點重新推論的穩態值，不是 Gain loss 的數值真值。''',
    904: '''繪圖時才把 z 真值反標準化、把 log 品質 exp 回濃度，方便用原始物理量閱讀。''',
    920: '''這裡是額外的繪圖 rollout，不參與訓練，所以可使用 torch.no_grad() 節省計算圖記憶體。''',
    930: '''每個 MV 畫 plus/minus 曲線；signed_delta 決定這次加上正擾動或負擾動。''',
    944: '''teacher 擾動同樣的原單位 Δu，轉成 teacher z 增量時必須除以 teacher 的 std。''',
    953: '''第一列是 baseline 與 Excel 穩態；後續每列是一個 MV 的正負擾動與 ANN 參考。''',
    961: '''逐品質設定座標軸、曲線、水平參考線及圖例；這些設定不影響 loss。''',
    985: '''tight_layout 調整排版；latest 圖每次覆寫，close 釋放這張 matplotlib 圖的資源。''',
    991: '''unsqueeze(0) 將 teacher 符號變成 [1,T*B,Q,M]，由 broadcasting 對齊兩個方向。''',
    994: '''符號懲罰 ReLU(-K_dyn * sign(K_teacher))：同號時為 0，異號時為 |K_dyn|。
這不是「gain 數值誤差」；也沒有要求與 teacher 相同的大小或額外的正向 margin。
特別注意 K_dyn=0：此 loss 是 0，但下面 KCI 的嚴格 >0 判定仍算不一致。''',
    997: '''valid mask 只由 teacher gain 門檻決定；expand_as 將它廣播到方向維，取得同形狀遮罩。''',
    1001: '''有效且乘積嚴格 >0 才算一致；逐「方向 × 時間步 × 穩態點 × 配對」計數。
不要求整段全部一致，也不先對 18 步 gain 平均再判斷正負。''',
    1007: '''沿方向與 time*batch 加總，留下 [Q,M] 的逐配對統計，避免總 KCI 掩蓋單一配對。''',
    1012: '''一般 Gain loss 是所有有效元素的平均；全無有效元素時用可反傳的零值替代。
恢復 Gain 計算前的 train/eval 狀態，供後續一般訓練使用。''',
    1018: '''未啟用群組縮放時，Gain 項就是「固定 gain 權重 × 全體有效平均懲罰」。''',
    1019: '''Difficult-GradNorm：每個 batch 量測梯度並直接算縮放倍率，沒有另一個權重 optimizer。
此實作使用指定符號群組與固定目標比例，不含依各 task 訓練速率學習權重的步驟。''',
    1020: '''基準目標 L_base = L_MSE + w_SS * L_SS；不包含 Gain。
SS 權重為 0 時，基準就是 MSE 梯度；量測範圍僅限 output_dense 參數。''',
    1024: '''只讀出基準梯度，不更新 .grad；retain_graph 保留稍後 total_loss.backward 所需的圖。
allow_unused=True 容許某參數未連到 loss，此時該梯度為 None，norm helper 會略過。''',
    1034: '''每個有效符號群組的目標 norm = target_ratio × 基準輸出層 norm。
例如 CLI 預設 0.015，即每組目標為基準的 1.5%；不是所有群組合計只占 1.5%。''',
    1036: '''先標記要重加權的配對；其餘配對保留原平均中的貢獻，避免同一項被計算兩次。''',
    1042: '''保留配對仍除以「全部有效比較數 valid_items」，而不是只除以 preserved 配對數。
這樣它們的每筆權重與未啟用 GradNorm 時相同。''',
    1049: '''指定配對逐一處理；每組再按 teacher 為正／負分開，因此三配對最多六個符號群組。
群組內包含 plus/minus 兩種擾動方向、tail 各時間步與符合 teacher 符號的穩態點。''',
    1052: '''固定 target/MV 後，pair_loss 與 pair_valid 的形狀是 [2,T*B]。''',
    1056: '''以 teacher 符號選群組，並與有效遮罩取交集；不是按照模型預測的正負分組。
該 batch 沒有某個符號群組時直接略過，不用零樣本去估計倍率。''',
    1063: '''每個符號群組先取自己的平均 loss，少數符號不會被多數符號的樣本數直接稀釋。
再乘固定 Gain 權重，量測「這個加權群組」在輸出層造成的原始梯度 norm。''',
    1075: '''scale = min(max_scale, 目標norm / 原始norm)：梯度弱時放大，過強時也可縮小到 <1。
只設倍率上限，沒有把倍率下限固定成 1；任一 norm 為零時倍率設 0。
norm helper 已回傳 Python float，所以 scale 作常數使用，不對算倍率的過程再反傳。''',
    1082: '''將倍率乘回仍保有計算圖的群組 loss；稍後 backward 才產生真正的參數梯度。''',
    1083: '''保存每批、每配對、每 teacher 符號的有效數、異號懲罰數、原始 norm 與倍率。
estimated_scaled_output_grad_norm 是單組 norm×scale；不是所有組相加後的總梯度 norm。''',
    1100: '''最終 Gain 項 = 原權重保留配對 + 各符號群組的縮放項。
各群組在參數空間可能同向或相衝，所以總梯度 norm 不等於各 norm 的純量相加。
loss_gain 仍記錄縮放前全體平均；weighted_gain_loss 才是實際加入總目標的 Gain 項。''',
    1102: '''診斷分支分別量 MSE、加權 SS、各配對符號群組的梯度，不更新 Transformer。
這裡的 Gain 診斷分量使用固定 effective_gain_loss_weight，並非前面 scale 後的群組項。
若同時啟用 GradNorm，倍率會另存 history，閱讀診斷 CSV 時需區分這兩種資料。''',
    1118: '''逐配對、逐 teacher 符號建立 Gain 診斷分量，以免正負群組混合後掩蓋梯度衝突。''',
    1147: '''對每個 loss 分量求所有模型參數梯度，保留計算圖供下一個分量共用。
儲存 raw/weighted loss、有效數、異號數，以及各架構群的 norm。''',
    1171: '''用清單後半段枚舉兩兩組合，不重複比較 A/B 與 B/A，再計算各參數群的 cosine。''',
    1193: '''達到指定診斷批數就 break；否則 continue 跳過下面 backward/optimizer.step。''',
    1197: '''真正訓練的總目標只有這三項；群組倍率已包含在 weighted_gain_loss 內。
MSE 與 SS 的梯度不乘群組倍率；GradNorm 只重組 Gain 部分。''',
    1205: '''item() 取得純量供 epoch 統計，不把整個 batch 的計算圖保存在 history 中。''',
    1211: '''一次 backward 累加總 loss 對所有 Transformer 參數的梯度。
接著把全模型梯度 L2 norm 裁到最多 1，再由 Adam 更新；這和前面的群組縮放是兩步。
因此 group history 的輸出層目標 norm 是全域裁切前的量，並非最終 Adam 更新量。''',
    1217: '''進度列顯示當批 MSE、加權 SS、實際 Gain 項；KCI_n 是本 epoch 至今的一致／有效計數。''',
    1225: '''診斷只執行一輪指定批次，輸出 CSV 後 return；不進入驗證、模型儲存或一般訓練圖表。''',
    1238: '''loss 是按執行批數取平均；KCI 則先累加所有有效比較的分子／分母，再取比例。
無有效比較時 epoch_kci 的程式預設為 1.0，不能據此解讀為真的達到 100% 一致。''',
    1249: '''將配對計數搬到 CPU 後產生報表，避免在 Python 逐項讀取 GPU 值。''',
    1261: '''ann_sign_avg 是有效 teacher 的平均 -1/+1，範圍 [-1,1]，並不是正號百分比。
dynamic_positive_percent 才是動態 gain 為正的比例；應與符號分布一起閱讀 KCI。''',
    1278: '''該配對沒有有效 teacher 時存 NaN 與 total=0，避免把無資料誤認為高／低 KCI。''',
    1291: '''彙整本 epoch teacher 輸出值的範圍與 <=1e-6 比例，用來察看 teacher 數值狀態。''',
    1302: '''保存跨 epoch 曲線所需純量；這裡的 KCI 是訓練抽到的穩態點，不是 held-out 評估。''',
    1306: '''驗證只計算動態逐步 MSE，不計算驗證 SS/Gain；eval+no_grad 停用隨機層與梯度圖。''',
    1316: '''沒有 valid_loader 時，這個變數改以本 epoch 訓練總 loss 代替，並非真正驗證 MSE。''',
    1321: '''以 val_mse_loss 決定最佳 checkpoint；不以 KCI 最高或訓練總 loss 最低選模型。
torch.save 僅存 state_dict 權重，不包含 optimizer 狀態或完整訓練 history。''',
    1331: '''連續 patience 個 epoch 沒改善就結束；最後記憶體中模型不會重新載入最佳 checkpoint。''',
    1335: '''訓練結束後輸出本次實際執行的 history；最佳模型已在上方驗證階段儲存。''',
    1342: '''同圖畫訓練總 loss 與驗證 MSE；兩條曲線包含的 loss 項不同，閱讀時需看圖例。''',
    1357: '''訓練迴圈記錄的 KCI 是 0~1 比例，輸出 CSV／圖表前才乘 100 轉為百分比。''',
    1365: '''SS history 同時保留未加權 loss、權重與加權值；SS 權重為零時仍能查看原始偏差。''',
    1374: '''只有實際產生群組縮放紀錄才輸出 GradNorm history，避免建立無內容的報表。''',
    1381: '''輸出訓練期間逐 epoch 的 KCI 曲線；它與獨立測試資料的 pair comparison 圖用途不同。''',
    1393: '''輸出完整逐配對 history，另選最後一個訓練 epoch 畫柱狀圖。
這是最後 epoch 的紀錄，不一定對應先前存下的最佳驗證 checkpoint。''',
    1402: '''顏色依 KCI 門檻標示；95% 水平線是視覺參考，不參與 loss 或 early stopping。''',
    1412: '''在每個柱子上加百分比與 n；n 是方向×時間步×有效穩態點的比較數，不是獨立點數。''',
    1427: '''最後列出 checkpoint 與各報表路徑，方便確認產物位置。''',
    1438: '''只有直接執行此檔案才解析 CLI 並開始工作；被其他模組 import 時不會自動訓練。
訓練範例：python scripts/training/train_pgin_stepwise_gain_difficult_group_gradnorm.py --config <YAML路徑> --no-ss
--gradient-diagnostics-only 則改成載入既有 checkpoint 做梯度診斷。''',
    1441: '''store_true：命令列有 --no-ss 才為 True；效果是覆寫 SS 權重為 0。''',
    1446: '''CLI 的 target_ratio 預設為 0.015；直接呼叫 main 而不傳值時，函式預設是 0.02。
max_scale 只限制各群組的倍率上限，不是整個模型的 gradient clipping 上限。''',
    1449: '''診斷選項：checkpoint 指定權重，batches 限制量測批數，output-dir 指定 CSV 位置。''',
    1461: '''parse_args 取得 Namespace；先檢查診斷批數，再決定本次實驗名稱後綴。''',
    1465: '''若 YAML 的 exp_name 已含 Difficult-GradNorm 後綴，就不重複加字。
--experiment-suffix 明確指定時會直接使用該值；其餘才按 --no-ss 自動決定後綴。''',
    1476: '''CLI 固定啟用 DIFFICULT_PAIRS 的群組縮放並使用八份 baseline 穩態訓練來源。
因此此入口不只是讀 YAML；還會把命令列倍率、來源與可選的 SS=0 override 傳給 main。''',
}

newline = '\r\n' if b'\r\n' in original_bytes else '\n'
output = []
for number, line in enumerate(lines, start=1):
    if number in notes:
        indent = line[:len(line) - len(line.lstrip())]
        for comment in notes[number].splitlines():
            output.append(indent + ('# ' + comment if comment else '#') + newline)
    output.append(line)
annotated = ''.join(output)

def code_tokens(source):
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING, tokenize.ENDMARKER}
    return [(token.type, token.string) for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type not in ignored]

assert code_tokens(original) == code_tokens(annotated), 'Non-comment token changed.'
assert ast.dump(ast.parse(original), include_attributes=False) == ast.dump(ast.parse(annotated), include_attributes=False), 'AST changed.'
compile(annotated, str(path), 'exec')
backup_dir = Path('logs') / ('comment_annotation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
backup_dir.mkdir()
(backup_dir / path.name).write_bytes(original_bytes)
path.write_bytes(annotated.encode('utf-8'))
print(f'Annotated {len(notes)} blocks; added {len(annotated.splitlines())-len(lines)} comment lines.')
print('Verified: executable tokens and AST unchanged; syntax compiles.')
print('Original source backup:', backup_dir / path.name)
