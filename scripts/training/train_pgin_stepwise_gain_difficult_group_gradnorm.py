"""Train step-wise PGIN with Difficult-GradNorm and optional SS value loss."""

# 閱讀指南：這支程式同時處理三件事。
# 1. 用動態歷史資料訓練 Transformer 的逐步預測（MSE）。
# 2. 可選：讓固定操作點的預測維持在 Excel 穩態值（SS loss）。
# 3. 用凍結 ANN 的局部 gain 正負號，約束 Transformer 的擾動反應方向。
#    Difficult-GradNorm 再調整指定配對、指定 teacher 符號群組的梯度強度。
# 閱讀順序：資料讀取 → main 設定 → 每批次 MSE/SS/Gain → 群組縮放 → 更新模型。
# 常用縮寫：MV＝操作變數；QV＝品質輸出；SS＝穩態；AR＝把預測回填歷史。
# 張量符號：B＝本批穩態點數；W＝歷史長度；T＝用於 Gain 的時間步數；
# Q＝品質輸出數；M＝受擾動的 MV 數；z＝標準化空間；p＝反標準化後的值。
# 注意：H2S/SO2 在 Transformer 端先取 log，因此名稱帶 p 不一定已還原濃度。
#
# 套件分工：os/sys/glob 處理路徑；argparse 讀命令列；yaml 讀實驗設定。
# random/NumPy/PyTorch 管理隨機與數值；pandas 處理表格；tqdm/plt 顯示進度與圖。
import os
import sys
import argparse
import random
import yaml
import glob
import torch
import torch.nn as nn

# Force the math attention path for reproducible Transformer behavior on CUDA.
# 指定 CUDA 的 attention 計算路徑，避免不同加速實作造成結果差異。
# 這三行是全域後端設定，並非凍結模型，也不代表所有 GPU 運算都已保證可重現。
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import contextlib
import io
import matplotlib.pyplot as plt

# 將專案根目錄加進 import 搜尋路徑，讓直接執行 scripts/training/ 下的腳本也能匯入 src。
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model


# 從命令列執行時，main 會收到這八份既定 LHS 訓練來源作為 override。
# 此處是訓練資料；與 evaluate_gain_in_out_distribution.py 的 held-out 測試來源分開。
BASELINE_STEADY_STATE_SOURCES = [
    f"./data/Claus_steady_state/lhs_generated_dynamic_ss_data_{index}.xlsx"
    for index in (10, 11, 15, 16, 17, 18, 19, 20)
]
# 每個 tuple 的順序是（輸出 target，操作變數 MV）。
# 這三組是命令列入口指定要做群組梯度縮放的配對；SO2 ← air2 仍計算 Gain loss，
# 但會走 preserved_gain_loss，維持原本的全體平均權重。
DIFFICULT_PAIRS = [
    ("B35_H2S", "air2_SP"),
    ("B35_H2S", "HEATER2_output_T_SP"),
    ("B35_SO2", "HEATER2_output_T_SP"),
]

# =====================================================================
# 1. Variable name mapping
#    HYSYS/raw tag names -> Python-friendly column names
# =====================================================================
# 把 Aspen 匯出的 tag 改成動態資料與模型共用的欄位名稱。
# 字典左邊是原始名稱，右邊是程式名稱；只改欄名，不在這裡轉換數值或單位。
FULL_MAPPING = {
    'B34.SPo.SPo': 'acidgas_Fm', 'B17.PV.PV': 'air', 'S20.P.P': 'HEATER1_output_P', 'B33.SPo.SPo': 'air2_SP',
    'B17.SPo.SPo': 'air_SP', 'B35.SPo.SPo': 'COG_SP', 'AIR2.Fv.Fv': 'second_air2', 'S4.Fv.Fv': 'COG',
    'B18.SPo.SPo': 'burner_input_T_SP', 'B18.PV.PV': 'burner_input_T_PV', 'B19.SPo.SPo': 'burner_output_T_SP',
    'B19.PV.PV': 'burner_output_T_PV', 'BURNER_PC.SPo.SPo': 'burner_output_P_SP', 'BURNER_PC.PV.PV': 'burner_output_P_PV',
    'FURANCE_PC.SPo.SPo': 'fur_outputP_SP', 'FURANCE_PC.PV.PV': 'fur_outputP_PV', 'FURANCE.T.0.(0)': 'fur_inputT',
    'FURANCE.T.1.(1)': 'fur_temp', 'SEP1_PC.SPo.SPo': 'SEP1_P_SP', 'SEP1_PC.PV.PV': 'SEP1_P_PV', 'SEP1.T.T': 'SEP1_T',
    'SEP2_PC.SPo.SPo': 'SEP2_P_SP', 'SEP2_PC.PV.PV': 'SEP2_P_PV', 'SEP2.T.T': 'SEP2_T', 'SEP3_PC.SPo.SPo': 'SEP3_P_SP',
    'SEP3_PC.PV.PV': 'SEP3_P_PV', 'SEP3.T.T': 'SEP3_T', 'B21.SPo.SPo': 'HEATER1_output_T_SP',
    'B21.PV.PV': 'HEATER1_output_T_PV', 'B20.SPo.SPo': 'HEATER2_output_T_SP', 'B20.PV.PV': 'HEATER2_output_T_PV',
    'CAT1_PC.SPo.SPo': 'cat1_output_P_SP', 'CAT1_PC.PV.PV': 'cat1_output_P_PV', 'CAT2_PC.SPo.SPo': 'cat2_output_P_SP',
    'CAT2_PC.PV.PV': 'cat2_output_P_PV', 'S12.F.F': 'fur_F', 'S12.P.P': 'fur_inputP', 'S15.T.T': 'fur_outputT',
    'S16.F.F': 'WHB_F', 'S16.P.P': 'WHB_inputP', 'S16.T.T': 'WHB_inputT', 'S13.T.T': 'WHB_outputT',
    'S13.P.P': 'WHB_outputP', 'S36.F.F': 'HEATER1_F', 'S36.P.P': 'HEATER1_input_P', 'S36.T.T': 'HEATER1_input_T',
    'S21.F.F': 'cat1_F', 'S21.P.P': 'cat1_input_P', 'S21.T.T': 'cat1_input_temp', 'S22.T.T': 'cat1_output_temp',
    'S25.F.F': 'HEATER2_F', 'S25.P.P': 'HEATER2_input_P', 'S25.T.T': 'HEATER2_input_T', 'S27.F.F': 'cat2_F',
    'S27.P.P': 'cat2_input_P', 'S27.T.T': 'cat2_input_temp', 'S28.T.T': 'cat2_output_temp', 'S14.F.F': 'SEP1_F',
    'S23.F.F': 'SEP2_F', 'S29.F.F': 'SEP3_F', 'ACIDGAS.T.T': 'acidgas_T', 'ACIDGAS.P.P': 'acidgas_P',
    'ACIDGAS.Fcn.H2O.("H2O")': 'acidgas_H2O', 'ACIDGAS.Fcn.H2S.("H2S")': 'acidgas_H2S',
    'ACIDGAS.Fcn.CO2.("CO2")': 'acidgas_CO2', 'S33.Zn.SO2.("SO2")': 'B35_SO2', 'S33.Zn.H2S.("H2S")': 'B35_H2S',
    'S8.P.P': 'burner_inputP'
}

# ANN teacher 的網路結構須與已儲存的 checkpoint 相符；本程式不訓練它。
# nn.Module 提供參數管理、train/eval 模式與 state_dict 載入功能。
class SimpleTabularMLP(nn.Module):
    """Small ANN used as the frozen steady-state gain teacher."""

    # __init__ 建立層與參數；super(...) 先初始化 PyTorch 的父類別。
    def __init__(self, input_dim, output_dim):
        super(SimpleTabularMLP, self).__init__()
        # Sequential 依列出的順序傳遞資料：輸入 → 64 → 32 → 輸出。
        # Linear 做線性映射；GELU 提供非線性；BatchNorm/Dropout 的行為會隨 train/eval 改變。
        # 載入 teacher 後使用 eval()，所以 Dropout 停用，BatchNorm 使用已存的統計值。
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(32, output_dim)
        )

    # 呼叫 mlp_model(x) 時會進入 forward：x 為 [B, input_dim]，回傳 [B, output_dim]。
    def forward(self, x):
        return self.net(x)

# 穩態讀檔入口：接受單一路徑、glob pattern，或多個來源組成的 list/tuple。
# 回傳清理過的 DataFrame；每一列代表一個穩態操作點，並非相鄰時刻。
def load_steady_state_source(path, required_cols, log_target_cols=None):
    """Load LHS steady-state rows and align raw Aspen dynamics names to model columns."""

    # isinstance 判斷是否已是多來源容器；單一字串先包成 list 才能統一迭代。
    # glob 展開萬用字元；sorted 固定各 pattern 的檔案順序。
    patterns = list(path) if isinstance(path, (list, tuple)) else [path]
    paths = []
    for pattern in patterns:
        matched_paths = sorted(glob.glob(pattern))
        if not matched_paths and os.path.exists(pattern):
            matched_paths = [pattern]
        paths.extend(matched_paths)
    # dict.fromkeys 去除重複路徑並保留首次出現的順序，避免同一份資料重複載入。
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(f"Steady-state source file not found: {path}")

    df_list = []
    total_before_drop = 0
    for source_path in paths:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in [".xlsx", ".xlsm", ".xls"]:
            # Excel 第三列是欄名（header=2 為從零起算），下一列是單位列，故再略過一列。
            df = pd.read_excel(source_path, sheet_name=0, header=2)
            df = df.iloc[1:].dropna(how='all').copy()
        elif ext == ".csv":
            df = pd.read_csv(source_path)
        else:
            raise ValueError(f"Unsupported steady-state source file type: {ext}")

        # 如果有 Aspen 執行狀態欄，只保留成功完成的模擬列。
        if 'Status' in df.columns:
            df = df[df['Status'] == 'Run Completed'].copy()

        # 只替換實際存在的欄名；重新命名後若有重複欄，保留第一個。
        # 後續 required_cols 檢查確保 Transformer 歷史、MV 與 ANN 輸入都齊全。
        rename_map = {raw_name: py_name for raw_name, py_name in FULL_MAPPING.items() if raw_name in df.columns}
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            missing_preview = ', '.join(missing_cols[:20])
            extra = f" ... (+{len(missing_cols) - 20} more)" if len(missing_cols) > 20 else ""
            raise ValueError(f"Missing required steady-state columns in {source_path}: {missing_preview}{extra}")

        # 選取需要的欄位並轉成數值；無法轉換的內容變 NaN，之後統一刪除該列。
        df = df[required_cols].apply(pd.to_numeric, errors='coerce')
        # Aspen exports B17.PV.PV as kg/hr; dynamics CSVs store `air` in kmol/hr.
        # 這裡只替 air 做單位換算；air2_SP 不經過這項換算。
        if 'air' in df.columns:
            df['air'] = df['air'] / 28.0408
        total_before_drop += len(df)
        df_list.append(df)

    # 合併來源並重建連續列索引；dropna 刪除任一必要欄位缺值的操作點。
    df = pd.concat(df_list, ignore_index=True)
    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No usable steady-state rows after numeric conversion/dropna: {path}")

    # H2S/SO2 與動態訓練一致，先裁到至少 1e-6，再取自然對數。
    # 因此 df_raw 的品質欄位實際上是 log 濃度；其他欄位仍是原單位。
    for col in log_target_cols or []:
        if col in df.columns:
            df[col] = np.log(np.clip(df[col], 1e-6, None))

    # 列出實際讀到的檔案、有效列數與丟棄列數，方便追查訓練資料來源。
    print(
        f"[Info] Loaded steady-state sources for SS/Gain training: {len(paths)} file(s) | "
        f"{len(df)} usable rows ({total_before_drop - len(df)} dropped)"
    )
    for source_path in paths:
        print(f"  - {source_path}")
    return df

# 將抽到的穩態操作點轉成 Encoder 所需的 W 步歷史，以及該點的操作條件。
# 回傳 historical_dfs（B 份 [W, 欄位數] 表格）與 ss1_p_df（B 列起始穩態點）。
# de_mv/all_cols/std_all/target_mvs 目前保留在介面中，函式本體未使用它們。
def generate_steady_state_batch(df_raw, batch_size, de_mv, all_cols, W, std_all, target_mvs=None, repeat_rows_as_history=False, indices=None):
    """Sample steady-state rows and build encoder histories for gain probing."""

    # 優先使用外部指定 indices；否則隨機抽樣（np.random.choice 預設可重複）。
    # 正式 main 使用 next_steady_state_indices 的排列索引，而非這裡的隨機抽樣分支。
    if indices is not None:
        idx1 = np.asarray(indices)
    elif repeat_rows_as_history:
        idx1 = np.random.choice(len(df_raw), batch_size)
    else:
        idx1 = np.random.choice(len(df_raw) - W - 1, batch_size)

    historical_dfs = []
    ss1_rows = []

    for i in idx1:
        # repeat_rows_as_history=True：把同一個穩態點重複 W 次，模擬固定操作的歷史。
        # False 分支取連續 W 列；對 LHS 表格而言這只是列切片，不保證是真實時間歷史。
        if repeat_rows_as_history:
            ss_row = df_raw.iloc[[i]].reset_index(drop=True)
            hist_w = pd.concat([ss_row] * W, ignore_index=True)
        else:
            hist_w = df_raw.iloc[i : i+W].reset_index(drop=True)
        historical_dfs.append(hist_w)
        ss1_rows.append(hist_w.iloc[-1:])

    # 每段歷史的最後一列作為 baseline 操作點；串接後列順序與 historical_dfs 相同。
    ss1_p_df = pd.concat(ss1_rows, ignore_index=True)
    return historical_dfs, ss1_p_df

# 依 cols 的順序計算 z=(值-mean)/std，再轉成指定裝置上的 float32 張量。
# mean/std 必須來自使用這個張量的模型；teacher 與 Transformer 使用不同統計值。
def dataframe_to_z_tensor(df, cols, mean, std, device):
    """Convert selected physical/log columns to the z-space used by the model."""

    mean_vals = mean[cols].values
    std_vals = std[cols].replace(0, 1).values
    z_values = (df[cols].values - mean_vals) / std_vals
    return torch.tensor(z_values, dtype=torch.float32, device=device)

# 梯度診斷使用「參數清單的索引」分組，不複製參數或改變模型結構。
def diagnostic_parameter_groups(named_parameters):
    """Map model parameters to the architectural groups used in diagnostics."""

    # all 包含所有參數；其他群組依模型參數名稱前綴區分 Encoder/Decoder/輸出層。
    # enumerate 同時提供索引與（名稱，參數）；這裡的 _ 表示不需要使用參數本身。
    groups = {"all": [], "encoder": [], "decoder": [], "output": []}
    for index, (name, _) in enumerate(named_parameters):
        groups["all"].append(index)
        if name.startswith(("enc_dense", "enc_layers")):
            groups["encoder"].append(index)
        elif name.startswith(("dec_dense", "dec_layers")):
            groups["decoder"].append(index)
        elif name.startswith("output_dense"):
            groups["output"].append(index)
    return groups


# 計算指定參數群的梯度 L2 norm：sqrt(sum(g**2))；None 表示該參數未參與此 loss。
# detach() 切斷診斷運算的計算圖，float(...cpu()) 將結果變成一般 Python 數字。
# 因此回傳值用於紀錄或計算倍率，不會對「梯度 norm 本身」再求導。
def gradient_norm(gradients, indices):
    squared_norm = sum(
        torch.sum(gradients[index].detach() ** 2)
        for index in indices
        if gradients[index] is not None
    )
    if not isinstance(squared_norm, torch.Tensor):
        return 0.0
    return float(torch.sqrt(squared_norm).cpu())


# 梯度 cosine＝內積 /（兩邊的 L2 norm 相乘）。
# 接近 +1 表示更新方向相近；負值表示相衝；零梯度或無可比較參數時回傳 NaN。
# 這是 loss 對參數的方向比較，不是 KCI 的輸入／輸出 gain 符號比較。
def gradient_cosine(left, right, indices):
    dot = sum(
        torch.sum(left[index].detach() * right[index].detach())
        for index in indices
        if left[index] is not None and right[index] is not None
    )
    left_sq = sum(
        torch.sum(left[index].detach() ** 2)
        for index in indices
        if left[index] is not None
    )
    right_sq = sum(
        torch.sum(right[index].detach() ** 2)
        for index in indices
        if right[index] is not None
    )
    if not all(isinstance(value, torch.Tensor) for value in (dot, left_sq, right_sq)):
        return np.nan
    denominator = torch.sqrt(left_sq * right_sq)
    if float(denominator.cpu()) <= 0:
        return np.nan
    return float((dot / denominator).cpu())


# 診斷輸出：保留逐批次明細，再按 loss 分量與參數群彙總平均值。
# 此函式只整理已量到的梯度，不執行反向傳播或 optimizer 更新。
def export_gradient_diagnostics(component_rows, cosine_rows, output_dir):
    """Write batch-level gradient diagnostics and simple aggregate summaries."""

    os.makedirs(output_dir, exist_ok=True)
    component_df = pd.DataFrame(component_rows)
    cosine_df = pd.DataFrame(cosine_rows)
    component_df.to_csv(
        os.path.join(output_dir, "gradient_component_detail.csv"), index=False
    )
    cosine_df.to_csv(
        os.path.join(output_dir, "gradient_cosine_detail.csv"), index=False
    )

    # 只挑數值欄位計算平均，避免把分量名稱等文字欄位混入統計。
    component_numeric = [
        "valid_count", "wrong_count", "loss_value", "weighted_loss_value",
        "grad_norm_all", "grad_norm_encoder", "grad_norm_decoder",
        "grad_norm_output",
    ]
    # groupby(...).mean() 的每筆輸入是診斷紀錄，因此是紀錄平均，並非按有效點數加權。
    component_summary = (
        component_df.groupby("component", as_index=False)[component_numeric]
        .mean(numeric_only=True)
    )
    cosine_summary = (
        cosine_df.groupby(["component_a", "component_b", "parameter_group"], as_index=False)["cosine"]
        .mean(numeric_only=True)
    )
    component_summary.to_csv(
        os.path.join(output_dir, "gradient_component_summary.csv"), index=False
    )
    cosine_summary.to_csv(
        os.path.join(output_dir, "gradient_cosine_summary.csv"), index=False
    )
    print(f"[Done] Gradient diagnostics saved to {output_dir}")


# 訓練主程式。CLI 在檔案最下方組好參數，再呼叫這個函式。
# *_override 的優先權高於 YAML；experiment_suffix 用來區分輸出實驗名稱。
# gradient_diagnostics_only=True 會載入 checkpoint、量測梯度並提前返回，不更新權重。
def main(
    config_path: str,
    gradient_diagnostics_only=False,
    diagnostic_checkpoint=None,
    diagnostic_batches=5,
    diagnostic_output_dir="results/gain_gradient_conflict_diagnostics_ss001",
    experiment_suffix="",
    h2s_group_gradnorm=False,
    h2s_group_target_ratio=0.02,
    h2s_group_max_scale=50.0,
    steady_state_source_override=None,
    gain_group_gradnorm_pairs=None,
    ss_loss_weight_override=None,
):
    # yaml.safe_load 將設定轉為巢狀 dict；config['training'].get(key, default) 在缺值時用預設。
    # Load the experiment config first; every tensor shape below is derived from it.
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 同時設定 Python、NumPy、PyTorch 的種子，並指定 cuDNN 的確定性／搜尋行為。
    seed = int(config['training'].get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # exp 決定 checkpoint 與 results 子目錄；只有 use_pretrained=True 才載入預訓練權重。
    # 微調模式也可另外指定 finetune_exp_name 與較小的 learning rate。
    exp = config['exp_name'] + experiment_suffix
    pretrained_path = config['training'].get(
        'pretrained_path',
        './saved_models/transformer_layerwise_71var_decoder_input_sp.pth'
    )
    use_pretrained = config['training'].get('use_pretrained', False)
    finetune_lr = config['training'].get('finetune_learning_rate', 1e-5)
    if use_pretrained:
        exp = config['training'].get('finetune_exp_name', exp + "_Finetuned")
        print(f"[Info] Pre-trained finetuning enabled. Experiment name: {exp}")

    # 設定要求 cuda 且機器有可用 GPU 才用 GPU；否則回退 CPU。
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'

    print("=" * 70)
    print(f"Physics-Informed Gain Training (From Scratch) - Session: {exp}")
    print(f"[Info] Random seed: {seed}")
    print("=" * 70)

    cfg_data = config['data']
    # variable_selection defines the shared contract among encoder, decoder, and targets.
    # variable_selection 決定資料契約：de_mv 是 Decoder 的操作變數，y_sv 是預測輸出，
    # en_mv_and_sv 是 Encoder 歷史欄位；列順序會直接決定張量各維對應的變數。
    # 63-variable 模式下，歷史新列的排列須與後面 cat([MV, prediction]) 一致。
    de_mv, y_sv, non_used, en_mv_and_sv = variable_selection(cfg_data['variables_num'])

    # Keep all columns that may be needed when constructing synthetic gain histories.
    # 整理建立穩態歷史所需的所有動態欄位，去重時保留既有順序。
    all_dynamic_cols = []
    for col in en_mv_and_sv:
        if col not in all_dynamic_cols:
            all_dynamic_cols.append(col)
    for col in y_sv:
        if col not in all_dynamic_cols:
            all_dynamic_cols.append(col)

    # 把實際欄位數寫回 config，get_model 才能建立相符的輸入與輸出層大小。
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)

    # Tabular ANN has a fixed 5-input/2-output steady-state interface.
    # teacher 的輸入順序固定為 air2、T2、酸氣流量、壓力、溫度；輸出固定 H2S、SO2。
    tab_input_cols = [
        'air2_SP',
        'HEATER2_output_T_SP',
        'acidgas_Fm',
        'acidgas_P',
        'acidgas_T',
    ]
    full_model_target_cols = [
        'B35_H2S',
        'B35_SO2',
    ]
    # YAML 可選要約束的品質與 MV；只保留 teacher 與 Transformer 都有的變數。
    gain_target_qv = config['training'].get('gain_target_qv', ["B35_H2S", "B35_SO2"])
    gain_target_mv = config['training'].get('gain_target_mv', ['air2_SP', 'HEATER2_output_T_SP'])
    tab_target_cols = [c for c in gain_target_qv if c in y_sv and c in full_model_target_cols]
    gain_target_mv = [c for c in gain_target_mv if c in de_mv and c in tab_input_cols]
    if len(tab_target_cols) == 0:
        raise ValueError("No valid gain_target_qv columns found in both Transformer outputs and ANN outputs.")
    if len(gain_target_mv) == 0:
        raise ValueError("No valid gain_target_mv columns found in both Transformer decoder inputs and ANN inputs.")
    # 把變數名稱轉成 Decoder 欄位索引，避免假設 MV 一定在某個固定位置。
    gain_target_mv_indices = [de_mv.index(c) for c in gain_target_mv]
    # 相容舊的 H2S-only 開關；有明確 gain_group_gradnorm_pairs 時優先使用明確清單。
    if gain_group_gradnorm_pairs is None and h2s_group_gradnorm:
        gain_group_gradnorm_pairs = [
            ("B35_H2S", mv_name) for mv_name in gain_target_mv
        ]
    # 清單非空就啟用群組梯度縮放；h2s_group_* 是沿用的參數名稱，實際可涵蓋 SO2。
    # 此處驗證每組 target/MV 都在本次 Gain loss 的資料契約中。
    gain_group_gradnorm_pairs = list(gain_group_gradnorm_pairs or [])
    invalid_gradnorm_pairs = [
        pair
        for pair in gain_group_gradnorm_pairs
        if pair[0] not in tab_target_cols or pair[1] not in gain_target_mv
    ]
    if invalid_gradnorm_pairs:
        raise ValueError(f"Invalid gain_group_gradnorm_pairs: {invalid_gradnorm_pairs}")
    group_gradnorm_enabled = bool(gain_group_gradnorm_pairs)

    # Dynamic CSVs provide supervised regression data and the z-score reference.
    print("[Info] Loading historical R5-X dataset and calculating Z-score Stats...")
    from src.dataset import MultiStepS2SDataset
    from torch.utils.data import DataLoader, ConcatDataset
    # interval 的單位是分鐘；W 會用「歷史分鐘數 / 取樣間隔」換算成步數。
    interval = config['window']['sampling_interval_min']
    use_median = config['window'].get('use_median_downsampling', True)

    all_dfs_log = []
    # 逐檔讀取動態資料，保留各檔邊界，後面不會跨檔拼接出時間窗口。
    if 'training_files' in cfg_data and cfg_data['training_files']:
        for fname in cfg_data['training_files']:
            fpath = os.path.join(cfg_data['path'], fname)
            if os.path.exists(fpath):
                df_seg = pd.read_csv(fpath)
                # 中位數降採樣：每 interval 列算一次完整窗口中位數，再每 interval 列取一筆。
                # 關閉 use_median 時改成直接抽取每隔 interval 列的資料。
                if interval > 1:
                    if use_median:
                        df_seg = df_seg.rolling(window=interval, min_periods=interval).median(numeric_only=True)
                        df_seg = df_seg.iloc[interval-1::interval].reset_index(drop=True)
                    else:
                        df_seg = df_seg.iloc[::interval].reset_index(drop=True)
                df_seg.dropna(inplace=True)
                # 先把品質變數取 log，再與其他欄位一起做 z-score，順序須與推論一致。
                target_cols_log = ['B35_H2S', 'B35_SO2']
                df_seg = data_utils.apply_log_transform(df_seg, target_cols_log)
                all_dfs_log.append(df_seg)
            else:
                print(f"  [Warning] Training file not found, skipped: {fpath}")

    if len(all_dfs_log) == 0:
        raise ValueError("No training data found to calculate Z-score stats.")

    # 這裡以 training_files 載入的全部資料估計 mean/std；時間切分在後面才進行。
    # 每份原檔仍另存於 all_dfs_log，合併表僅用來計算共用的標準化統計值。
    df_all_log = pd.concat(all_dfs_log, ignore_index=True)
    mean_all, std_all = data_utils.calculate_zscore_stats(df_all_log)

    # 一般訓練會存下 Transformer 的 mean/std 供推論使用；診斷模式跳過這份輸出。
    # Save z-score statistics during training, but keep diagnostics read-only.
    if not gradient_diagnostics_only:
        zscore_dir = f'./results/{exp}/'
        os.makedirs(zscore_dir, exist_ok=True)
        mean_all.to_csv(os.path.join(zscore_dir, 'zscore_mean.csv'))
        std_all.to_csv(os.path.join(zscore_dir, 'zscore_std.csv'))
        print(f"  [Save] Z-score stats saved to {zscore_dir}")

    # teacher 使用自己的訓練統計值，不能直接套用 Transformer 的 mean_all/std_all。
    tab_mean_path = './results/Tabular_MLP_New/zscore_mean.csv'
    tab_std_path = './results/Tabular_MLP_New/zscore_std.csv'
    if not os.path.exists(tab_mean_path) or not os.path.exists(tab_std_path):
        raise FileNotFoundError(
            "Tabular MLP z-score stats are required for ANN steady-state targets: "
            f"{tab_mean_path}, {tab_std_path}"
        )

    tab_mean = pd.read_csv(tab_mean_path, index_col=0).squeeze("columns")
    tab_std = pd.read_csv(tab_std_path, index_col=0).squeeze("columns").replace(0, 1)
    missing_tab_stats = [c for c in tab_input_cols + full_model_target_cols if c not in tab_mean.index or c not in tab_std.index]
    if missing_tab_stats:
        missing_preview = ', '.join(missing_tab_stats[:20])
        extra = f" ... (+{len(missing_tab_stats) - 20} more)" if len(missing_tab_stats) > 20 else ""
        raise ValueError(f"Missing Tabular MLP z-score stats for: {missing_preview}{extra}")

    # 若動態統計值缺少 teacher 必需欄位，才用 teacher 統計值補入，不覆蓋既有欄位。
    for col in tab_input_cols + full_model_target_cols:
        if col not in mean_all.index and col in tab_mean.index:
            mean_all[col] = tab_mean[col]
            std_all[col] = tab_std[col]

    # 逐份動態資料套用同一組 Transformer z-score，仍維持原本的檔案分界。
    all_dfs_z = []
    for df_log in all_dfs_log:
        df_z = data_utils.apply_zscore(df_log, mean_all, std_all)
        all_dfs_z.append(df_z)

    # 三種時間長度分開：回歸預設 36 步、SS 使用 H_out、Gain 使用 warmup+H_out。
    # W 是輸入歷史長度，與「接下來要預測幾步」不是同一個參數。
    # Regression horizon is intentionally independent from the gain rollout horizon.
    train_datasets = []
    valid_datasets = []
    W = int(config['window']['train_window_mins'] / config['window']['sampling_interval_min'])
    H_out = config['window']['prediction_length']
    regression_steps = int(config['training'].get('stepwise_regression_steps', 36))
    if regression_steps < 1:
        raise ValueError("stepwise_regression_steps must be >= 1.")
    dataset_H = regression_steps
    print(f"[Info] Step-wise regression training length: {dataset_H} steps")

    for df_z in all_dfs_z:
        # 每份檔案按時間先留末段 test，再從前段留出 valid；本程式不建立 test_loader。
        split_point1 = int(len(df_z) * (1 - cfg_data['test_data_split']))
        split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))

        train_df = df_z.iloc[:split_point2]
        if len(train_df) > W + dataset_H:
            # Dataset 每個樣本回傳（過去 Encoder 歷史，未來 MV，未來輸出真值）。
            # DataLoader 組批後分別是 [B,W,F_en]、[B,regression_steps,F_mv]、[B,regression_steps,F_y]。
            train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, dataset_H)
            train_datasets.append(train_ds)

        valid_df = df_z.iloc[split_point2:split_point1]
        if len(valid_df) > W + dataset_H:
            valid_ds = MultiStepS2SDataset(valid_df, en_mv_and_sv, de_mv, y_sv, W, dataset_H)
            valid_datasets.append(valid_ds)

    # ConcatDataset 合併各檔的樣本集合；shuffle 打亂窗口抽取順序，不打亂窗口內時間。
    train_ds = ConcatDataset(train_datasets)
    batch_size = config['training'].get('batch_size', 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print(f'[Info] Loaded Dynamic Train Loader: {len(train_loader)} batches')

    if len(valid_datasets) > 0:
        valid_ds = ConcatDataset(valid_datasets)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        print(f'[Info] Loaded Dynamic Valid Loader: {len(valid_loader)} batches')
    else:
        valid_loader = None
        print('[Warning] No Validation data available!')

    # LHS 穩態資料提供兩種監督：Excel 值供 SS loss，操作條件供 ANN 算 gain 符號。
    # Steady-state rows provide all Transformer fixed-point targets and ANN gain probes.
    print('[Info] Preparing LHS steady-state data for SS and Gain losses...')
    keep_cols = []
    for c in all_dynamic_cols + tab_input_cols:
        if c not in keep_cols:
            keep_cols.append(c)

    # 來源優先順序：函式 override → YAML steady_state_source_path → 預設 glob。
    # 本檔 CLI 固定傳入上方八份 BASELINE_STEADY_STATE_SOURCES。
    steady_state_source_path = steady_state_source_override or config['training'].get(
        'steady_state_source_path',
        './data/Claus_steady_state/lhs_generated_dynamic_ss_data*.xlsx'
    )
    df_raw = load_steady_state_source(
        steady_state_source_path,
        keep_cols,
        log_target_cols=['B35_H2S', 'B35_SO2']
    )

    # 依設定建立 Transformer；微調或診斷模式才會另外載入既有 state_dict。
    print("[Info] Initializing Transformer Model...")
    dynamic_model = get_model(config).to(device)
    if use_pretrained:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Configured pretrained_path does not exist: {pretrained_path}")
        print(f"  [Info] Loading pre-trained MSE weights: {pretrained_path}")
        print(f"  [Info] Loading weights for Finetuning...")
        dynamic_model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("  [Info] Pre-trained loading disabled. Training entirely from scratch.")
    # 診斷 checkpoint 優先用命令列指定路徑；否則用本次 exp 對應的已儲存權重。
    if gradient_diagnostics_only:
        checkpoint_path = diagnostic_checkpoint or f'./saved_models/{exp}.pth'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Gradient diagnostic checkpoint does not exist: {checkpoint_path}"
            )
        dynamic_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  [Info] Gradient diagnostics checkpoint: {checkpoint_path}")
    dynamic_model.train()

    # teacher 參數不訓練，但稍後仍須對「teacher 輸入」求導，所以不能整段用 no_grad。
    # The ANN teacher is frozen; gradients through its inputs give the target gain signs.
    print("[Info] Loading Pre-trained TabularMLP Model...")
    # teacher 輸出以它的 mean/std 還原成濃度；Transformer 品質輸出則還原到 log 濃度。
    # 兩者 gain 數值尺度不同，本程式只取 teacher gain 的符號作為方向參考。
    tab_target_mean_tensor = torch.tensor(tab_mean[full_model_target_cols].values, dtype=torch.float32, device=device)
    tab_target_std_tensor = torch.tensor(tab_std[full_model_target_cols].values, dtype=torch.float32, device=device)
    dynamic_full_target_mean_tensor = torch.tensor(mean_all[full_model_target_cols].values, dtype=torch.float32, device=device)
    dynamic_full_target_std_tensor = torch.tensor(std_all[full_model_target_cols].values, dtype=torch.float32, device=device)
    mlp_model = SimpleTabularMLP(input_dim=len(tab_input_cols), output_dim=len(full_model_target_cols))
    # 先找共用 teacher checkpoint；不存在時才使用以實驗名稱命名的備援檔。
    mlp_path = f'./saved_models/Tabular_MLP_5in_2out_QV.pth'
    if not os.path.exists(mlp_path):
        mlp_path = f'./saved_models/{exp}_tabular_mlp.pth'
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp_model.to(device)
    # eval() 控制 BatchNorm/Dropout；requires_grad_(False) 才是凍結 teacher 參數。
    # 這兩個操作不會阻止稍後 requires_grad=True 的輸入取得梯度。
    mlp_model.eval()
    mlp_model.requires_grad_(False)

    # steps_per_epoch 限制每個 epoch 只跑多少動態 batch，並非一定跑完整個 train_loader。
    epochs = config['training'].get('epochs', 20)
    steps_per_epoch = min(config['training'].get('steps_per_epoch', 50), len(train_loader))
    base_lr = config['training'].get('learning_rate', 1e-4)
    if use_pretrained:
        base_lr = finetune_lr
        print(f"  [Info] Using fine-tuning Learning Rate: {base_lr}")

    # 唯一負責更新參數的 optimizer；它只拿到 Transformer 參數，沒有 teacher 或群組倍率。
    optimizer = optim.Adam(dynamic_model.parameters(), lr=base_lr)

    # Gain 時間軸：先維持 baseline MV，再從 step_change_step 起維持擾動後的 MV。
    # 目前對應 YAML 是 35 步 warmup + 18 步輸出，共 53 步；參數仍以實際設定為準。
    # 設定中的 step 從 1 起算，Python 索引從 0 起算，故後面需要減 1。
    # Gain rollout: warm up on baseline MV, then apply the step change.
    W = int(config['window']['train_window_mins'] / config['window']['sampling_interval_min'])
    gain_warmup_steps = int(config['training'].get('stepwise_gain_warmup_steps', 60))
    H_gain = int(config['training'].get('stepwise_gain_steps', gain_warmup_steps + H_out))
    H_ss = H_out
    if H_gain < 1:
        raise ValueError("stepwise_gain_steps must be >= 1.")
    if gain_warmup_steps < 0 or gain_warmup_steps >= H_gain:
        raise ValueError("stepwise_gain_warmup_steps must be >= 0 and less than stepwise_gain_steps.")
    dynamic_gain_step_change_step = int(
        config['training'].get('dynamic_gain_step_change_step', gain_warmup_steps + 1)
    )
    if dynamic_gain_step_change_step < 1 or dynamic_gain_step_change_step > H_gain:
        raise ValueError("dynamic_gain_step_change_step must be 1-based and within stepwise_gain_steps.")
    dynamic_gain_step_change_idx = dynamic_gain_step_change_step - 1
    # runtime 圖只在正常訓練開啟；診斷模式不產生這張訓練中即時圖。
    pgin_runtime_plot = (
        config['training'].get('pgin_runtime_plot', True)
        and not gradient_diagnostics_only
    )

    # 記錄最佳驗證 loss 與連續未改善次數，供儲存模型與 early stopping 使用。
    best_loss = float('inf')
    early_stop_patience = config['training'].get('patience', 10)
    epochs_no_improve = 0

    # 把輸出欄位的 mean/std 放到相同裝置；接近零的 std 換成 1，避免除零。
    y_mean_tensor = torch.tensor(mean_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_tensor = torch.tensor(std_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_safe = torch.where(torch.abs(y_std_tensor) < 1e-6, torch.ones_like(y_std_tensor), y_std_tensor)

    # Safe std tensors avoid divide-by-zero when converting ANN/dynamic outputs.
    tab_target_std_safe = torch.where(torch.abs(tab_target_std_tensor) < 1e-6, torch.ones_like(tab_target_std_tensor), tab_target_std_tensor)
    dynamic_full_target_std_safe = torch.where(
        torch.abs(dynamic_full_target_std_tensor) < 1e-6,
        torch.ones_like(dynamic_full_target_std_tensor),
        dynamic_full_target_std_tensor
    )
    # SS loss 可指定輸出欄位清單；否則按 scope 使用全部 y_sv 或只使用品質欄位。
    ss_target_cols_cfg = config['training'].get('ss_target_cols')
    if ss_target_cols_cfg:
        ss_target_cols = [c for c in ss_target_cols_cfg if c in y_sv]
        missing_ss_targets = [c for c in ss_target_cols_cfg if c not in ss_target_cols]
        if missing_ss_targets:
            raise ValueError(f"Invalid ss_target_cols entries: {missing_ss_targets}")
    else:
        ss_target_scope = config['training'].get('ss_target_scope', 'all_48_outputs')
        if ss_target_scope == 'all_48_outputs':
            ss_target_cols = list(y_sv)
        elif ss_target_scope == 'quality_only':
            ss_target_cols = [c for c in full_model_target_cols if c in y_sv]
        else:
            raise ValueError(
                "ss_target_scope must be 'all_48_outputs' or 'quality_only'."
            )
    if len(ss_target_cols) == 0:
        raise ValueError("No valid steady-state target columns configured.")
    # 建立不同欄位集合之間的索引映射：SS 對 Transformer，監測品質對 teacher。
    ss_dyn_target_idx = [y_sv.index(c) for c in ss_target_cols]
    ann_monitor_idx = [full_model_target_cols.index(c) for c in tab_target_cols]

    # 此處的 rollout 是預測回填 Encoder 歷史的自回歸運算，不是實際 Aspen/MPC 閉迴路控制。
    # SS 的值參考來自 Excel；ANN 在此主要負責提供 Gain 方向。
    # The SS term matches a closed-loop Transformer rollout to Excel steady-state targets.
    # SS 權重優先採用 override；--no-ss 會傳入 0，YAML 也可直接設定 ss_loss_weight=0。
    # 權重為 0 時不貢獻總 loss，但下方仍會計算 SS rollout 與原始 SS loss。
    ss_loss_weight = (
        config['training'].get('ss_loss_weight', 0.05)
        if ss_loss_weight_override is None
        else float(ss_loss_weight_override)
    )
    # Gain 開關與權重分開：關閉 enable_gain_loss 時有效權重為 0，仍可能只監測 KCI。
    gain_loss_weight = config['training'].get('gain_loss_weight', config['training'].get('pgin_loss_weight', 0.3))
    enable_gain_loss = config['training'].get('enable_gain_loss', True)
    monitor_gain_kci = config['training'].get('monitor_gain_kci', True)
    effective_gain_loss_weight = gain_loss_weight if enable_gain_loss else 0.0
    # 群組縮放需要正的 Gain 權重；target_ratio 與 max_scale 也先檢查合法範圍。
    if group_gradnorm_enabled and effective_gain_loss_weight <= 0:
        raise ValueError("Group gradient normalization requires a positive gain loss weight.")
    if h2s_group_target_ratio <= 0:
        raise ValueError("h2s_group_target_ratio must be positive.")
    if h2s_group_max_scale < 1:
        raise ValueError("h2s_group_max_scale must be at least 1.")
    # 只要訓練 Gain、監測 KCI 或設定 runtime 圖任一需要，就會計算 Gain rollout。
    compute_gain_metrics = enable_gain_loss or monitor_gain_kci or config['training'].get('pgin_runtime_plot', True)
    # teacher gain 太接近零時，正負號不穩定，因此低於此門檻的比較不納入 loss/KCI。
    gain_valid_delta_threshold = config['training'].get('gain_valid_delta_threshold', 1e-5)
    # Gain tail 是選來計算 loss/KCI 的時間段，起點至少是 step change 開始的那一步。
    # stepwise_gain_tail_start_step 可覆寫；沒有時使用 dynamic_gain_tail_start_step 與 step change 的較大值。
    # 這是訓練端的可設定 tail；獨立評估腳本則固定取最後 18 步，兩者不要混為同一段。
    dynamic_gain_tail_start_step = config['training'].get(
        'stepwise_gain_tail_start_step',
        max(config['training'].get('dynamic_gain_tail_start_step', dynamic_gain_step_change_step), dynamic_gain_step_change_step)
    )
    # 實際擾動量 Δu＝finite_diff_delta_std × 該 MV 的 Transformer 訓練標準差。
    # 這裡讀取訓練 YAML，與評估腳本內獨立設定的 sigma 不會自動連動。
    finite_diff_delta_std = config['training'].get('finite_diff_delta_std', 0.5)
    ss_batch_size = config['training'].get('ss_batch_size', 4)
    if ss_batch_size < 1:
        raise ValueError("ss_batch_size must be >= 1.")
    if dynamic_gain_tail_start_step < 1:
        raise ValueError("dynamic_gain_tail_start_step must be 1-based and >= 1.")
    # 將 tail 起點轉成合法的零起算索引；切片 end 不包含該索引，所以用 H_gain。
    dynamic_gain_tail_start_idx = min(max(dynamic_gain_tail_start_step, dynamic_gain_step_change_step) - 1, H_gain - 1)
    dynamic_gain_tail_end_idx = H_gain
    # 穩態點用隨機排列依序取批；cursor 會跨 epoch 保留，走完整份資料才重新洗牌。
    ss_perm = np.random.permutation(len(df_raw))
    ss_cursor = 0

    # 此巢狀函式共用 main 的 ss_perm/ss_cursor，避免每批重新抽樣而遺漏某些穩態點。
    def next_steady_state_indices(batch_size):
        """Return non-repeating steady-state indices until every row is covered."""

        # nonlocal 表示更新外層函式的狀態；末批只回傳剩餘點，不補抽成完整 batch。
        nonlocal ss_perm, ss_cursor
        if ss_cursor >= len(ss_perm):
            ss_perm = np.random.permutation(len(df_raw))
            ss_cursor = 0

        end_cursor = min(ss_cursor + batch_size, len(ss_perm))
        batch_indices = ss_perm[ss_cursor:end_cursor]
        ss_cursor = end_cursor
        return batch_indices

    # 列出本次真正生效的權重、時間軸與配對；遇到設定疑問時先核對這份摘要。
    print(
        f"  [Info] Loss weights | MSE: 1.0, "
        f"SS: {ss_loss_weight}, Gain: {effective_gain_loss_weight}; "
        f"SS rollout: {H_ss} steps; SS targets ({len(ss_target_cols)}): {ss_target_cols}; "
        f"Gain loss enabled: {enable_gain_loss}; "
        f"KCI monitor: {compute_gain_metrics}; "
        f"Gain targets: {tab_target_cols} x {gain_target_mv}; "
        f"Gain valid delta threshold: {gain_valid_delta_threshold}; "
        f"Dynamic gain method: finite_difference; "
        f"Dynamic gain rollout: {H_gain} step-wise steps; "
        f"Step change starts at step {dynamic_gain_step_change_step}; "
        f"Dynamic gain output: steps {dynamic_gain_tail_start_idx + 1}-{dynamic_gain_tail_end_idx}; "
        f"Gain reduction: per-step sign penalty, then mean; KCI counts step comparisons; "
        f"FD delta std: {finite_diff_delta_std}; "
        f"SS batch: {ss_batch_size}; SS coverage: full permutation without replacement; "
        f"Group GradNorm pairs: {gain_group_gradnorm_pairs or 'none'}"
    )

    # history_* 儲存跨 epoch 的結果；diagnostic_* 儲存診斷模式各 batch 的梯度明細。
    history_losses = []
    history_val_losses = []
    history_kci = []
    history_ss_losses = []
    history_gain_pair_rows = []
    history_group_gradnorm_rows = []
    diagnostic_component_rows = []
    diagnostic_cosine_rows = []
    # 保留固定的參數清單順序，讓 autograd.grad 回傳的梯度能與診斷群組索引對齊。
    named_diagnostic_parameters = list(dynamic_model.named_parameters())
    diagnostic_parameters = [parameter for _, parameter in named_diagnostic_parameters]
    diagnostic_groups = diagnostic_parameter_groups(named_diagnostic_parameters)
    # 群組縮放只以 output_dense 的權重／bias 梯度 norm 作為量尺。
    # 實際 total_loss.backward() 仍會更新整個 Transformer，不是只訓練輸出層。
    gradnorm_reference_parameters = [
        parameter
        for name, parameter in named_diagnostic_parameters
        if name.startswith("output_dense")
    ]

    # 外部 helper 每步預測、計算 MSE，再把 detach 後的預測回填歷史；最後平均所有步的 loss。
    from src.engine import step_wise_rolling_training_step
    # Regression criterion: average MSE over the step-wise rolling horizon.
    criterion = nn.MSELoss()

    # Main optimization loop mixes supervised regression, SS loss, and gain loss.
    # 每個 epoch 重設統計計數；模型重新切回 train 模式，讓一般回歸使用訓練行為。
    for epoch in range(epochs):
        dynamic_model.train()
        epoch_gain_loss = 0.0
        epoch_ss_loss = 0.0
        epoch_correct_dir = 0
        epoch_total_eval = 0
        epoch_mse_loss = 0.0
        epoch_total_loss = 0.0
        # 配對統計形狀 [Q,M]；correct/total 計數，另外記錄 teacher 符號與動態正號比例。
        pair_correct = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        pair_total = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        pair_ann_sign_sum = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        pair_dyn_pos_sum = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        # 以正／負無限大初始化本 epoch 的 teacher 輸出極值，再由每批結果逐次更新。
        ann_monitor_min = torch.full((len(tab_target_cols),), float('inf'), device=device)
        ann_monitor_max = torch.full((len(tab_target_cols),), float('-inf'), device=device)
        ann_monitor_clamped = 0
        ann_monitor_count = 0

        # enumerate 提供批次索引；手動 break 確保只執行 step_limit 批。
        step_limit = min(steps_per_epoch, len(train_loader))
        pbar = tqdm(enumerate(train_loader), total=step_limit, desc=f"Epoch {epoch+1}/{epochs}")

        for step, mse_batch in pbar:
            if step >= step_limit:
                break

            # 清除上一批累積在參數 .grad 的梯度；稍後只對本批的 total_loss 做一次更新。
            optimizer.zero_grad()

            # Supervised dynamic regression loss over the configured rolling horizon.
            # MSE 分支使用真實動態 batch，與接下來抽取的穩態 batch 不是同一組資料。
            mse_loss_val = step_wise_rolling_training_step(
                dynamic_model, mse_batch, criterion, device
            )

            # Draw steady-state rows for ANN teacher gradients and dynamic gain probes.
            # 穩態 batch 大小獨立於動態 batch；排列最後一批可能較短，所以使用 B_actual。
            ss_indices = next_steady_state_indices(ss_batch_size)
            historical_dfs, ss1_df = generate_steady_state_batch(
                df_raw, ss_batch_size, de_mv, keep_cols, W, std_all, gain_target_mv,
                repeat_rows_as_history=True, indices=ss_indices
            )
            B_actual = len(historical_dfs)

            # 暫時把標準化 helper 的 stdout 導到記憶體，避免每個穩態點都印出訊息。
            with contextlib.redirect_stdout(io.StringIO()):
                x_en_z_list = []
                for b_df in historical_dfs:
                    b_z = data_utils.apply_zscore(b_df, mean_all, std_all).fillna(0.0)
                    x_en_z_list.append(torch.tensor(b_z[en_mv_and_sv].values, dtype=torch.float32, device=device))
                # stack 把 B 份 [W,F_en] 歷史組成 [B,W,F_en]；同一點的 W 列目前都是相同穩態值。
                x_en_z_history = torch.stack(x_en_z_list)

                ss1_z_df = data_utils.apply_zscore(ss1_df, mean_all, std_all).fillna(0.0)

            # Decoder baseline 操作值 [B,F_mv] 保留原單位；做擾動時先改原單位，再轉回 z。
            ss1_de_p = torch.tensor(ss1_df[de_mv].values, dtype=torch.float32, device=device)

            # 同一穩態點改用 teacher 的 scaler 轉成 [B,5]；啟用輸入梯度以取得局部 gain。
            mlp_x_z_ss1 = dataframe_to_z_tensor(ss1_df, tab_input_cols, tab_mean, tab_std, device)
            mlp_x_z_ss1.requires_grad_(True)

            # Physical ANN outputs are used both for monitoring and gain-direction gradients.
            # teacher 輸出先反標準化為濃度，再用 detach 的副本做 min/max 與非正值監測。
            # ann_monitor_clamped 是 <=1e-6 的計數；這幾行並未真的裁切 y_mlp_p_ss1。
            y_mlp_z_ss1 = mlp_model(mlp_x_z_ss1)
            y_mlp_p_ss1 = y_mlp_z_ss1 * tab_target_std_safe + tab_target_mean_tensor
            ann_monitor_vals = y_mlp_p_ss1.detach()[:, ann_monitor_idx]
            ann_monitor_min = torch.minimum(ann_monitor_min, ann_monitor_vals.min(dim=0).values)
            ann_monitor_max = torch.maximum(ann_monitor_max, ann_monitor_vals.max(dim=0).values)
            ann_monitor_clamped += (ann_monitor_vals <= 1e-6).sum().item()
            ann_monitor_count += ann_monitor_vals.numel()

            # SS 的監督真值是 Excel 操作點在 Transformer z 空間中的值，並非 teacher 預測值。
            ss_target_z = torch.tensor(ss1_z_df[ss_target_cols].values, dtype=torch.float32, device=device)

            # SS 分支固定所有 MV，從重複穩態歷史預測 H_ss 步，檢查模型是否維持固定點。
            # A repeated steady-state history should remain at the same Excel target.
            ss1_de_z = torch.tensor(ss1_z_df[de_mv].values, dtype=torch.float32, device=device)
            ss_rollout_history = x_en_z_history.clone()
            ss_pred_z_steps = []
            for _ in range(H_ss):
                _, ss_context = dynamic_model.encoder(ss_rollout_history)
                # unsqueeze(1) 增加時間維：[B,F_mv] → [B,1,F_mv]，Decoder 每次只預測一步。
                ss_de_input = ss1_de_z.unsqueeze(1)
                ss_prediction = dynamic_model.decoder(ss_de_input, ss_context)
                ss_pred_z_steps.append(ss_prediction[:, 0, ss_dyn_target_idx])

                # 每次丟掉最舊歷史，加入 [MV, 當步預測]；detach 只切斷跨步歷史回填的梯度。
                # ss_pred_z_steps 仍保存未 detach 的當步輸出，因此每步 SS loss 都能訓練模型。
                if len(ss_pred_z_steps) < H_ss:
                    # Match the truncated autoregressive history update used by MSE training.
                    ss_new_step = torch.cat(
                        [ss_de_input, ss_prediction.detach()],
                        dim=2
                    )
                    ss_rollout_history = torch.cat(
                        [ss_rollout_history[:, 1:, :], ss_new_step],
                        dim=1
                    )

            # 沿 dim=1 疊成 [B,H_ss,SS輸出數]；expand_as 讓同一穩態真值對應每個時間步。
            ss_pred_z_stack = torch.stack(ss_pred_z_steps, dim=1)
            ss_target_z_stack = ss_target_z.unsqueeze(1).expand_as(ss_pred_z_stack)
            ss_loss_val = criterion(ss_pred_z_stack, ss_target_z_stack)

            # Gain 沒有計算時先使用零 loss；記住目前模式，計算完 Gain 後再恢復。
            loss_gain = torch.tensor(0.0, device=device)
            gain_prev_training_mode = dynamic_model.training
            if compute_gain_metrics:
                # eval() 關閉 Dropout 等訓練隨機性，讓 baseline/plus/minus 的差異來自輸入擾動。
                # eval() 不等於 no_grad()；Gain 分支仍建立 Transformer 參數的反向傳播圖。
                dynamic_model.eval()
                # K_ss_matrix is ANN d(target) / d(MV physical), used only for sign.
                # teacher Jacobian 形狀 [B,Q,M]，每個元素是該操作點某品質對某 MV 的局部偏導。
                K_ss_matrix = torch.zeros(B_actual, len(tab_target_cols), len(gain_target_mv), device=device)
                for t_idx, tgt_col in enumerate(tab_target_cols):
                    if tgt_col in full_model_target_cols:
                        full_out_idx = full_model_target_cols.index(tgt_col)
                        # grad_outputs 只把當前品質輸出設為 1，其他設 0，以選出要微分的輸出欄位。
                        grad_outputs = torch.zeros_like(y_mlp_p_ss1)
                        grad_outputs[:, full_out_idx] = 1.0

                        # autograd.grad 直接回傳對輸入的梯度，不寫進模型參數的 .grad。
                        # create_graph=False：teacher 導數作固定參考；retain_graph=True：後續品質還要共用此圖。
                        grads_x, = torch.autograd.grad(
                            outputs=y_mlp_p_ss1,
                            inputs=mlp_x_z_ss1,
                            grad_outputs=grad_outputs,
                            create_graph=False,
                            retain_graph=True
                        )

                        for col_idx_in_gain, col in enumerate(gain_target_mv):
                            # 目前求到的是 d(濃度)/d(MV_z)；除以 teacher 的 MV std，換成 d(濃度)/d(MV原單位)。
                            col_idx_in_tab = tab_input_cols.index(col)
                            s_val = tab_std.get(col, 1.0)
                            s = s_val if abs(s_val) > 1e-6 else 1.0
                            K_ss_matrix[:, t_idx, col_idx_in_gain] = grads_x[:, col_idx_in_tab] / s

                # torch.sign 將 gain 轉為 -1/0/+1；後面只要求方向一致，不要求 gain 大小相等。
                K_ss_direction = torch.sign(K_ss_matrix)

                # 對應 Transformer 的品質欄位索引；log_target_idx 只用於顯示時還原濃度。
                tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                log_cols_inv = [c for c in ['B35_H2S', 'B35_SO2'] if c in y_sv]
                log_target_idx = [y_sv.index(c) for c in log_cols_inv]
                gain_start_history = x_en_z_history.clone()

                # 內部 helper 使用外層的 de_mv、mean_all/std_all，將 [B,F_mv] 原單位操作值轉回 z。
                def de_physical_to_z(de_p_tensor):
                    """Map decoder MV values from physical units to model z-space."""

                    de_z_cols = []
                    for col_idx_in_de, col in enumerate(de_mv):
                        m = mean_all.get(col, 0.0)
                        s_val = std_all.get(col, 1.0)
                        s = s_val if abs(s_val) > 1e-6 else 1.0
                        de_z_cols.append((de_p_tensor[:, col_idx_in_de] - m) / s)
                    return torch.stack(de_z_cols, dim=1)

                # 一次完整 Gain rollout：baseline 與擾動案例都從相同初始歷史出發。
                # step_de_p_tensor=None 表示完全不擾動；否則只在指定 step change 之後改用擾動值。
                # 回傳 [H_gain,B,Q] 的反標準化品質軌跡（H2S/SO2 仍在 log 空間），另附 MV z 與 debug。
                def rollout_gain_stack(base_de_p_tensor, step_de_p_tensor=None, collect_debug=False):
                    """Run a full step-wise AR gain rollout and collect target traces."""

                    base_de_z_graph = de_physical_to_z(base_de_p_tensor)
                    step_de_z_graph = (
                        base_de_z_graph if step_de_p_tensor is None else de_physical_to_z(step_de_p_tensor)
                    )
                    # 每次呼叫都 clone 初始歷史，避免 plus/minus 接續 baseline 的最終狀態。
                    gain_history = gain_start_history.clone()
                    gain_log_steps_local = []
                    debug_steps_local = []
                    for step_idx in range(H_gain):
                        # The configured step change starts after the baseline warmup window.
                        # 擾動前使用 baseline MV；擾動開始後維持 step MV，不是在每一步累加一次 Δu。
                        current_de_z = (
                            step_de_z_graph
                            if step_idx >= dynamic_gain_step_change_idx
                            else base_de_z_graph
                        )
                        single_step_de_input = current_de_z.unsqueeze(1)
                        _, context = dynamic_model.encoder(gain_history)
                        single_step_prediction = dynamic_model.decoder(single_step_de_input, context)
                        # 反 z-score，但尚未 exp：本程式的 Gain loss 使用 log 品質差／原單位 MV 差。
                        # 對正濃度，log 是單調函數，局部導數的符號與濃度導數相同；數值大小則不同。
                        pred_log_or_p_gain = single_step_prediction.squeeze(1) * y_std_safe + y_mean_tensor
                        gain_log_steps_local.append(pred_log_or_p_gain[:, tab_target_idx])

                        # collect_debug 才額外還原成濃度並裁到 1e-6；debug 結果 detach 後不參與 loss。
                        if collect_debug:
                            pred_p_gain = pred_log_or_p_gain.clone()
                            if len(log_target_idx) > 0:
                                pred_p_gain[:, log_target_idx] = torch.exp(pred_p_gain[:, log_target_idx])
                            pred_p_gain[:, tab_target_idx] = torch.clamp(pred_p_gain[:, tab_target_idx], min=1e-6)
                            debug_steps_local.append(pred_p_gain[:, tab_target_idx].detach())

                        # AR 數值上仍使用先前預測；detach 讓後續 loss 不沿回填歷史一路反傳到前面各步。
                        # 保存在 gain_log_steps_local 的當步輸出仍有梯度，並非整條 rollout 都無法訓練。
                        if step_idx < H_gain - 1:
                            # Same truncated-AR update as regression training.
                            new_step_features = torch.cat(
                                [single_step_de_input, single_step_prediction.detach()],
                                dim=2
                            )
                            gain_history = torch.cat([gain_history[:, 1:, :], new_step_features], dim=1)

                    # stack(dim=0) 把每步 [B,Q] 疊成 [H_gain,B,Q]，時間維放在最前面。
                    gain_log_stack_local = torch.stack(gain_log_steps_local, dim=0)
                    return gain_log_stack_local, base_de_z_graph, debug_steps_local

                # baseline 操作點視為固定輸入；detach().clone() 隔開外部圖與後續原地擾動修改。
                baseline_de_p = ss1_de_p.detach().clone()
                gain_log_stack, _, _ = rollout_gain_stack(
                    baseline_de_p,
                    step_de_p_tensor=None,
                    collect_debug=False
                )

                # 只選定 tail 段參與 Gain loss；warmup 的預測仍會影響後續歷史與數值軌跡。
                baseline_target_steps = gain_log_stack[
                    dynamic_gain_tail_start_idx:dynamic_gain_tail_end_idx, :, :
                ]

                # Keep each time step: apply the sign penalty before averaging losses.
                K_dyn_plus_by_mv = []
                K_dyn_minus_by_mv = []
                # 每次只擾動一個 MV，其他 MV 保持 baseline；不同 MV 使用各自的 std 換算 Δu。
                for mv_name in gain_target_mv:
                    mv_de_idx = de_mv.index(mv_name)
                    std_val = std_all[mv_name] if abs(std_all[mv_name]) > 1e-6 else 1.0
                    delta = finite_diff_delta_std * std_val

                    # clone 出兩份操作值再加減，避免把 baseline 本身或另一個方向一起改掉。
                    plus_de_p = baseline_de_p.clone()
                    minus_de_p = baseline_de_p.clone()
                    plus_de_p[:, mv_de_idx] += delta
                    minus_de_p[:, mv_de_idx] -= delta

                    # 分別重跑 +Δu、-Δu 的完整時間軌跡，選取與 baseline 相同的 tail 時間段。
                    plus_log_stack, _, _ = rollout_gain_stack(
                        baseline_de_p, step_de_p_tensor=plus_de_p, collect_debug=False
                    )
                    minus_log_stack, _, _ = rollout_gain_stack(
                        baseline_de_p, step_de_p_tensor=minus_de_p, collect_debug=False
                    )
                    plus_target_steps = plus_log_stack[
                        dynamic_gain_tail_start_idx:dynamic_gain_tail_end_idx, :, :
                    ]
                    minus_target_steps = minus_log_stack[
                        dynamic_gain_tail_start_idx:dynamic_gain_tail_end_idx, :, :
                    ]

                    # 逐步的單邊有限差分：K+ = (y+ - y0)/Δu；K- = (y0 - y-)/Δu。
                    # minus 的分子順序使兩方向都代表「輸出隨 MV 增加」的斜率；不先平均正負方向。
                    K_dyn_plus_by_mv.append((plus_target_steps - baseline_target_steps) / delta)
                    K_dyn_minus_by_mv.append((baseline_target_steps - minus_target_steps) / delta)

                # [time, batch, target, MV] -> [time*batch, target, MV].
                # Existing pair/group reductions now average over time as well.
                # 先疊 MV 維，再 flatten(0,1) 合併時間與 batch：[T,B,Q,M] → [T*B,Q,M]。
                # 接著疊正負方向為 [2,T*B,Q,M]；每個元素都會獨立做符號判斷與懲罰。
                K_dyn_plus_matrix = torch.stack(K_dyn_plus_by_mv, dim=-1).flatten(0, 1)
                K_dyn_minus_matrix = torch.stack(K_dyn_minus_by_mv, dim=-1).flatten(0, 1)
                K_dyn_matrix_stack = torch.stack([K_dyn_plus_matrix, K_dyn_minus_matrix], dim=0)
                gain_step_count = baseline_target_steps.shape[0]
                # teacher 每個操作點只算一次，repeat 按時間複製成 [T*B,Q,M]，與 flatten 的順序對齊。
                K_ss_matrix = K_ss_matrix.repeat(gain_step_count, 1, 1)
                K_ss_direction = torch.sign(K_ss_matrix)

                # 只在每個 epoch 第一批畫第 0 個穩態點的示意軌跡，供觀察數值反應。
                # 圖中的 ANN 水平線是對擾動後操作點重新推論的穩態值，不是 Gain loss 的數值真值。
                if pgin_runtime_plot and step == 0:
                    # Runtime plot exposes baseline/plus/minus dynamic traces against ANN targets.
                    runtime_dir = './results/PGIN_Visualizations'
                    os.makedirs(runtime_dir, exist_ok=True)

                    # 繪圖時才把 z 真值反標準化、把 log 品質 exp 回濃度，方便用原始物理量閱讀。
                    ss_target_sample = ss_target_z[0].detach().cpu().numpy()
                    ss_target_mean_np = mean_all[ss_target_cols].values
                    ss_target_std_np = std_all[ss_target_cols].replace(0, 1).values
                    ss_pred_phys = gain_log_stack[:, 0, :].detach().cpu().numpy()
                    ss_target_phys = ss_target_sample * ss_target_std_np + ss_target_mean_np
                    for log_idx, col_name in enumerate(tab_target_cols):
                        if col_name in ['B35_H2S', 'B35_SO2']:
                            ss_pred_phys[:, log_idx] = np.exp(ss_pred_phys[:, log_idx])
                    for log_idx, col_name in enumerate(ss_target_cols):
                        if col_name in ['B35_H2S', 'B35_SO2']:
                            ss_target_phys[log_idx] = np.exp(ss_target_phys[log_idx])

                    time_axis = np.arange(1, H_gain + 1)

                    response_curves = {}
                    response_ann_targets = {}
                    # 這裡是額外的繪圖 rollout，不參與訓練，所以可使用 torch.no_grad() 節省計算圖記憶體。
                    with torch.no_grad():
                        for mv_name in gain_target_mv:
                            mv_de_idx = de_mv.index(mv_name)
                            mv_tab_idx = tab_input_cols.index(mv_name)
                            delta = finite_diff_delta_std * (
                                std_all[mv_name] if abs(std_all[mv_name]) > 1e-6 else 1.0
                            )
                            response_curves[mv_name] = {}
                            response_ann_targets[mv_name] = {}

                            # 每個 MV 畫 plus/minus 曲線；signed_delta 決定這次加上正擾動或負擾動。
                            for direction, signed_delta in [("plus", delta), ("minus", -delta)]:
                                response_de_p = ss1_de_p.detach().clone()
                                response_de_p[:, mv_de_idx] += signed_delta
                                response_log_stack, _, _ = rollout_gain_stack(
                                    baseline_de_p,
                                    step_de_p_tensor=response_de_p,
                                    collect_debug=False
                                )
                                response_phys_stack = response_log_stack[:, 0, :].detach().cpu().numpy()
                                for log_idx, col_name in enumerate(tab_target_cols):
                                    if col_name in ['B35_H2S', 'B35_SO2']:
                                        response_phys_stack[:, log_idx] = np.exp(response_phys_stack[:, log_idx])
                                response_curves[mv_name][direction] = response_phys_stack

                                # teacher 擾動同樣的原單位 Δu，轉成 teacher z 增量時必須除以 teacher 的 std。
                                response_mlp_x_z = mlp_x_z_ss1.detach().clone()
                                tab_s_val = tab_std.get(mv_name, 1.0)
                                tab_s = tab_s_val if abs(tab_s_val) > 1e-6 else 1.0
                                response_mlp_x_z[:, mv_tab_idx] += signed_delta / tab_s
                                response_mlp_p = mlp_model(response_mlp_x_z) * tab_target_std_safe + tab_target_mean_tensor
                                response_ann_targets[mv_name][direction] = (
                                    response_mlp_p[0, ann_monitor_idx].detach().cpu().numpy()
                                )

                    # 第一列是 baseline 與 Excel 穩態；後續每列是一個 MV 的正負擾動與 ANN 參考。
                    fig, axes = plt.subplots(
                        1 + len(gain_target_mv),
                        len(tab_target_cols),
                        figsize=(6 * len(tab_target_cols), 4 * (1 + len(gain_target_mv)))
                    )
                    if len(tab_target_cols) == 1:
                        axes = axes.reshape(1 + len(gain_target_mv), 1)

                    # 逐品質設定座標軸、曲線、水平參考線及圖例；這些設定不影響 loss。
                    for tgt_idx, tgt_name in enumerate(tab_target_cols):
                        ss_target_col_idx = ss_target_cols.index(tgt_name)
                        axes[0, tgt_idx].plot(time_axis, ss_pred_phys[:, tgt_idx], color='purple', linewidth=2, label='Transformer rollout')
                        axes[0, tgt_idx].axhline(ss_target_phys[ss_target_col_idx], color='black', linestyle='--', linewidth=1.5, label='Excel S.S. target')
                        axes[0, tgt_idx].set_title(f'{tgt_name} rollout | epoch {epoch + 1}, step {step}')
                        axes[0, tgt_idx].set_xlabel('Decoder step')
                        axes[0, tgt_idx].set_ylabel('Physical value')
                        axes[0, tgt_idx].grid(True, linestyle='--', alpha=0.5)
                        axes[0, tgt_idx].legend(loc='best')

                        for mv_plot_idx, mv_name in enumerate(gain_target_mv, start=1):
                            ax = axes[mv_plot_idx, tgt_idx]
                            ax.plot(time_axis, ss_pred_phys[:, tgt_idx], color='black', linewidth=1.8, label='baseline')
                            ax.plot(time_axis, response_curves[mv_name]["plus"][:, tgt_idx], color='crimson', linewidth=2, label=f'{mv_name} +{finite_diff_delta_std:g} std')
                            ax.plot(time_axis, response_curves[mv_name]["minus"][:, tgt_idx], color='steelblue', linewidth=2, label=f'{mv_name} -{finite_diff_delta_std:g} std')
                            ax.axhline(ss_target_phys[ss_target_col_idx], color='black', linestyle='--', linewidth=1, alpha=0.8, label='Excel baseline')
                            ax.axhline(response_ann_targets[mv_name]["plus"][tgt_idx], color='crimson', linestyle='--', linewidth=1, alpha=0.8, label='ANN +step')
                            ax.axhline(response_ann_targets[mv_name]["minus"][tgt_idx], color='steelblue', linestyle='--', linewidth=1, alpha=0.8, label='ANN -step')
                            ax.set_title(f'{tgt_name} response to {mv_name}')
                            ax.set_xlabel('Decoder step')
                            ax.set_ylabel('Physical value')
                            ax.grid(True, linestyle='--', alpha=0.5)
                            ax.legend(loc='best', fontsize=8)

                    # tight_layout 調整排版；latest 圖每次覆寫，close 釋放這張 matplotlib 圖的資源。
                    plt.tight_layout()
                    latest_runtime_path = os.path.join(runtime_dir, 'pgin_runtime_latest.png')
                    plt.savefig(latest_runtime_path, dpi=180)
                    plt.close()
                    print(f"\n  [Plot] Updated PGIN runtime rollout/gain plot: {latest_runtime_path}")

                # unsqueeze(0) 將 teacher 符號變成 [1,T*B,Q,M]，由 broadcasting 對齊兩個方向。
                K_ss_direction_exp = K_ss_direction.unsqueeze(0)

                # Penalize only sign disagreement between dynamic gain and ANN teacher gain.
                # 符號懲罰 ReLU(-K_dyn * sign(K_teacher))：同號時為 0，異號時為 |K_dyn|。
                # 這不是「gain 數值誤差」；也沒有要求與 teacher 相同的大小或額外的正向 margin。
                # 特別注意 K_dyn=0：此 loss 是 0，但下面 KCI 的嚴格 >0 判定仍算不一致。
                loss_matrix_stack = torch.nn.functional.relu(-K_dyn_matrix_stack * K_ss_direction_exp)

                # Ignore weak ANN teacher gradients because their sign is not reliable.
                # valid mask 只由 teacher gain 門檻決定；expand_as 將它廣播到方向維，取得同形狀遮罩。
                valid_mlp_mask = torch.abs(K_ss_matrix) >= gain_valid_delta_threshold
                valid_mlp_mask_expanded = valid_mlp_mask.unsqueeze(0).expand_as(loss_matrix_stack)

                final_mask = valid_mlp_mask_expanded
                # 有效且乘積嚴格 >0 才算一致；逐「方向 × 時間步 × 穩態點 × 配對」計數。
                # 不要求整段全部一致，也不先對 18 步 gain 平均再判斷正負。
                correct_mask = (K_dyn_matrix_stack * K_ss_direction_exp > 0) & final_mask

                valid_items = final_mask.sum().item()
                epoch_correct_dir += correct_mask.sum().item()
                epoch_total_eval += valid_items

                # 沿方向與 time*batch 加總，留下 [Q,M] 的逐配對統計，避免總 KCI 掩蓋單一配對。
                pair_correct += correct_mask.sum(dim=(0, 1))
                pair_total += final_mask.sum(dim=(0, 1))
                pair_ann_sign_sum += (K_ss_direction_exp.expand_as(final_mask).float() * final_mask.float()).sum(dim=(0, 1))
                pair_dyn_pos_sum += ((K_dyn_matrix_stack > 0).float() * final_mask.float()).sum(dim=(0, 1))

                # 一般 Gain loss 是所有有效元素的平均；全無有效元素時用可反傳的零值替代。
                # 恢復 Gain 計算前的 train/eval 狀態，供後續一般訓練使用。
                if valid_items > 0:
                    loss_gain = torch.mean(loss_matrix_stack[final_mask])
                else:
                    loss_gain = torch.tensor(0.0, device=device, requires_grad=True)
                dynamic_model.train(gain_prev_training_mode)

            # 未啟用群組縮放時，Gain 項就是「固定 gain 權重 × 全體有效平均懲罰」。
            weighted_gain_loss = effective_gain_loss_weight * loss_gain
            # Difficult-GradNorm：每個 batch 量測梯度並直接算縮放倍率，沒有另一個權重 optimizer。
            # 此實作使用指定符號群組與固定目標比例，不含依各 task 訓練速率學習權重的步驟。
            if group_gradnorm_enabled:
                # 基準目標 L_base = L_MSE + w_SS * L_SS；不包含 Gain。
                # SS 權重為 0 時，基準就是 MSE 梯度；量測範圍僅限 output_dense 參數。
                base_objective = (
                    mse_loss_val
                    + ss_loss_weight * ss_loss_val
                )
                # 只讀出基準梯度，不更新 .grad；retain_graph 保留稍後 total_loss.backward 所需的圖。
                # allow_unused=True 容許某參數未連到 loss，此時該梯度為 None，norm helper 會略過。
                base_reference_gradients = torch.autograd.grad(
                    base_objective,
                    gradnorm_reference_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                base_reference_norm = gradient_norm(
                    base_reference_gradients,
                    range(len(base_reference_gradients)),
                )
                # 每個有效符號群組的目標 norm = target_ratio × 基準輸出層 norm。
                # 例如 CLI 預設 0.015，即每組目標為基準的 1.5%；不是所有群組合計只占 1.5%。
                desired_group_norm = h2s_group_target_ratio * base_reference_norm

                # 先標記要重加權的配對；其餘配對保留原平均中的貢獻，避免同一項被計算兩次。
                normalized_pair_mask = torch.zeros_like(final_mask)
                for target_name, mv_name in gain_group_gradnorm_pairs:
                    normalized_pair_mask[
                        :, :, tab_target_cols.index(target_name), gain_target_mv.index(mv_name)
                    ] = True
                preserved_mask = final_mask & ~normalized_pair_mask
                # 保留配對仍除以「全部有效比較數 valid_items」，而不是只除以 preserved 配對數。
                # 這樣它們的每筆權重與未啟用 GradNorm 時相同。
                preserved_gain_loss = (
                    effective_gain_loss_weight
                    * loss_matrix_stack[preserved_mask].sum()
                    / max(valid_items, 1)
                )
                scaled_group_terms = []

                # 指定配對逐一處理；每組再按 teacher 為正／負分開，因此三配對最多六個符號群組。
                # 群組內包含 plus/minus 兩種擾動方向、tail 各時間步與符合 teacher 符號的穩態點。
                for target_name, mv_name in gain_group_gradnorm_pairs:
                    target_idx = tab_target_cols.index(target_name)
                    mv_idx = gain_target_mv.index(mv_name)
                    # 固定 target/MV 後，pair_loss 與 pair_valid 的形狀是 [2,T*B]。
                    pair_loss = loss_matrix_stack[:, :, target_idx, mv_idx]
                    pair_valid = final_mask[:, :, target_idx, mv_idx]
                    teacher_direction = K_ss_direction[:, target_idx, mv_idx]
                    for teacher_label, teacher_value in (("positive", 1), ("negative", -1)):
                        # 以 teacher 符號選群組，並與有效遮罩取交集；不是按照模型預測的正負分組。
                        # 該 batch 沒有某個符號群組時直接略過，不用零樣本去估計倍率。
                        group_mask = pair_valid & (
                            teacher_direction == teacher_value
                        ).unsqueeze(0)
                        valid_group_count = int(group_mask.sum().item())
                        if valid_group_count == 0:
                            continue

                        # 每個符號群組先取自己的平均 loss，少數符號不會被多數符號的樣本數直接稀釋。
                        # 再乘固定 Gain 權重，量測「這個加權群組」在輸出層造成的原始梯度 norm。
                        group_loss = pair_loss[group_mask].mean()
                        raw_weighted_group_loss = effective_gain_loss_weight * group_loss
                        group_reference_gradients = torch.autograd.grad(
                            raw_weighted_group_loss,
                            gradnorm_reference_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        raw_group_norm = gradient_norm(
                            group_reference_gradients,
                            range(len(group_reference_gradients)),
                        )
                        # scale = min(max_scale, 目標norm / 原始norm)：梯度弱時放大，過強時也可縮小到 <1。
                        # 只設倍率上限，沒有把倍率下限固定成 1；任一 norm 為零時倍率設 0。
                        # norm helper 已回傳 Python float，所以 scale 作常數使用，不對算倍率的過程再反傳。
                        if raw_group_norm > 0 and desired_group_norm > 0:
                            scale = min(
                                h2s_group_max_scale,
                                desired_group_norm / raw_group_norm,
                            )
                        else:
                            scale = 0.0
                        # 將倍率乘回仍保有計算圖的群組 loss；稍後 backward 才產生真正的參數梯度。
                        scaled_group_terms.append(scale * raw_weighted_group_loss)
                        # 保存每批、每配對、每 teacher 符號的有效數、異號懲罰數、原始 norm 與倍率。
                        # estimated_scaled_output_grad_norm 是單組 norm×scale；不是所有組相加後的總梯度 norm。
                        history_group_gradnorm_rows.append(
                            {
                                "epoch": epoch + 1,
                                "step": step + 1,
                                "target": target_name,
                                "mv": mv_name,
                                "teacher_sign": teacher_label,
                                "valid_count": valid_group_count,
                                "wrong_count": int((pair_loss[group_mask] > 0).sum().item()),
                                "base_output_grad_norm": base_reference_norm,
                                "raw_group_output_grad_norm": raw_group_norm,
                                "target_output_grad_norm": desired_group_norm,
                                "scale": scale,
                                "estimated_scaled_output_grad_norm": raw_group_norm * scale,
                            }
                        )

                # 最終 Gain 項 = 原權重保留配對 + 各符號群組的縮放項。
                # 各群組在參數空間可能同向或相衝，所以總梯度 norm 不等於各 norm 的純量相加。
                # loss_gain 仍記錄縮放前全體平均；weighted_gain_loss 才是實際加入總目標的 Gain 項。
                weighted_gain_loss = preserved_gain_loss + sum(scaled_group_terms)

            # 診斷分支分別量 MSE、加權 SS、各配對符號群組的梯度，不更新 Transformer。
            # 這裡的 Gain 診斷分量使用固定 effective_gain_loss_weight，並非前面 scale 後的群組項。
            # 若同時啟用 GradNorm，倍率會另存 history，閱讀診斷 CSV 時需區分這兩種資料。
            if gradient_diagnostics_only:
                diagnostic_components = {
                    "mse": {
                        "loss": mse_loss_val,
                        "raw_loss": mse_loss_val,
                        "valid_count": int(mse_batch[0].shape[0]),
                        "wrong_count": np.nan,
                    },
                    "ss_weighted": {
                        "loss": ss_loss_weight * ss_loss_val,
                        "raw_loss": ss_loss_val,
                        "valid_count": B_actual,
                        "wrong_count": np.nan,
                    },
                }

                # 逐配對、逐 teacher 符號建立 Gain 診斷分量，以免正負群組混合後掩蓋梯度衝突。
                for target_idx, target_name in enumerate(tab_target_cols):
                    for mv_idx, mv_name in enumerate(gain_target_mv):
                        pair_loss = loss_matrix_stack[:, :, target_idx, mv_idx]
                        pair_valid = final_mask[:, :, target_idx, mv_idx]
                        pair_alignment = (
                            K_dyn_matrix_stack[:, :, target_idx, mv_idx]
                            * K_ss_direction[:, target_idx, mv_idx].unsqueeze(0)
                        )
                        for teacher_label, teacher_value in (("positive", 1), ("negative", -1)):
                            teacher_mask = (
                                K_ss_direction[:, target_idx, mv_idx] == teacher_value
                            ).unsqueeze(0)
                            group_mask = pair_valid & teacher_mask
                            group_count = int(group_mask.sum().item())
                            if group_count == 0:
                                continue
                            group_loss = pair_loss[group_mask].mean()
                            component_name = (
                                f"gain_{target_name}_{mv_name}_teacher_{teacher_label}"
                            )
                            diagnostic_components[component_name] = {
                                "loss": effective_gain_loss_weight * group_loss,
                                "raw_loss": group_loss,
                                "valid_count": group_count,
                                "wrong_count": int(
                                    ((pair_alignment < 0) & group_mask).sum().item()
                                ),
                            }

                # 對每個 loss 分量求所有模型參數梯度，保留計算圖供下一個分量共用。
                # 儲存 raw/weighted loss、有效數、異號數，以及各架構群的 norm。
                component_gradients = {}
                for component_name, component in diagnostic_components.items():
                    gradients = torch.autograd.grad(
                        component["loss"],
                        diagnostic_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    component_gradients[component_name] = gradients
                    diagnostic_component_rows.append(
                        {
                            "batch": step + 1,
                            "component": component_name,
                            "valid_count": component["valid_count"],
                            "wrong_count": component["wrong_count"],
                            "loss_value": float(component["raw_loss"].detach().cpu()),
                            "weighted_loss_value": float(component["loss"].detach().cpu()),
                            **{
                                f"grad_norm_{group_name}": gradient_norm(gradients, indices)
                                for group_name, indices in diagnostic_groups.items()
                            },
                        }
                    )

                # 用清單後半段枚舉兩兩組合，不重複比較 A/B 與 B/A，再計算各參數群的 cosine。
                component_names = list(component_gradients)
                for left_idx, left_name in enumerate(component_names):
                    for right_name in component_names[left_idx + 1:]:
                        for group_name, indices in diagnostic_groups.items():
                            diagnostic_cosine_rows.append(
                                {
                                    "batch": step + 1,
                                    "component_a": left_name,
                                    "component_b": right_name,
                                    "parameter_group": group_name,
                                    "cosine": gradient_cosine(
                                        component_gradients[left_name],
                                        component_gradients[right_name],
                                        indices,
                                    ),
                                }
                            )
                print(
                    f"[Gradient diagnostics] completed batch {step + 1}/"
                    f"{diagnostic_batches}",
                    flush=True,
                )
                # 達到指定診斷批數就 break；否則 continue 跳過下面 backward/optimizer.step。
                if step + 1 >= diagnostic_batches:
                    break
                continue

            # 真正訓練的總目標只有這三項；群組倍率已包含在 weighted_gain_loss 內。
            # MSE 與 SS 的梯度不乘群組倍率；GradNorm 只重組 Gain 部分。
            total_loss = (
                # Main supervised regression term.
                mse_loss_val
                # Excel steady-state value consistency term.
                + ss_loss_weight * ss_loss_val
                # Directional dynamic-gain consistency term.
                + weighted_gain_loss
            )
            # item() 取得純量供 epoch 統計，不把整個 batch 的計算圖保存在 history 中。
            epoch_mse_loss += mse_loss_val.item()
            epoch_ss_loss += ss_loss_val.item()
            epoch_total_loss += total_loss.item()

            total_norm_val = 0.0
            if total_loss.item() > 0:
                # 一次 backward 累加總 loss 對所有 Transformer 參數的梯度。
                # 接著把全模型梯度 L2 norm 裁到最多 1，再由 Adam 更新；這和前面的群組縮放是兩步。
                # 因此 group history 的輸出層目標 norm 是全域裁切前的量，並非最終 Adam 更新量。
                total_loss.backward()
                # clip_grad_norm_ returns the gradient norm before clipping.
                total_norm = torch.nn.utils.clip_grad_norm_(dynamic_model.parameters(), 1.0)
                total_norm_val = total_norm.item()
                optimizer.step()

            # 進度列顯示當批 MSE、加權 SS、實際 Gain 項；KCI_n 是本 epoch 至今的一致／有效計數。
            epoch_gain_loss += loss_gain.item()
            pbar.set_postfix({
                'MSE': f"{mse_loss_val.item():.4f}",
                'SS_w': f"{(ss_loss_weight * ss_loss_val).item():.5f}",
                'Gain_w': f"{weighted_gain_loss.item():.5f}",
                'KCI_n': f"{epoch_correct_dir}/{epoch_total_eval}"
            })

        # 診斷只執行一輪指定批次，輸出 CSV 後 return；不進入驗證、模型儲存或一般訓練圖表。
        if gradient_diagnostics_only:
            export_gradient_diagnostics(
                diagnostic_component_rows,
                diagnostic_cosine_rows,
                diagnostic_output_dir,
            )
            if history_group_gradnorm_rows:
                pd.DataFrame(history_group_gradnorm_rows).to_csv(
                    os.path.join(diagnostic_output_dir, 'gain_group_gradnorm_history.csv'),
                    index=False,
                )
            return

        # loss 是按執行批數取平均；KCI 則先累加所有有效比較的分子／分母，再取比例。
        # 無有效比較時 epoch_kci 的程式預設為 1.0，不能據此解讀為真的達到 100% 一致。
        avg_loss = epoch_total_loss / step_limit
        avg_mse_loss = epoch_mse_loss / step_limit
        avg_ss_loss = epoch_ss_loss / step_limit
        avg_gain_loss = epoch_gain_loss / step_limit
        epoch_kci = epoch_correct_dir / epoch_total_eval if epoch_total_eval > 0 else 1.0
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Total L: {avg_loss:.6f} | "
            f"MSE: {avg_mse_loss:.6f} | "
            f"SS: {avg_ss_loss:.6f} | Gain: {avg_gain_loss:.8f} | "
            f"KCI: {epoch_correct_dir}/{epoch_total_eval} = {epoch_kci*100:.2f}%"
        )
        # 將配對計數搬到 CPU 後產生報表，避免在 Python 逐項讀取 GPU 值。
        if compute_gain_metrics:
            pair_correct_cpu = pair_correct.detach().cpu()
            pair_total_cpu = pair_total.detach().cpu()
            pair_ann_sign_cpu = pair_ann_sign_sum.detach().cpu()
            pair_dyn_pos_cpu = pair_dyn_pos_sum.detach().cpu()
            print("  [Gain Pair KCI]")
            for q_idx, q_name in enumerate(tab_target_cols):
                parts = []
                for mv_idx, mv_name in enumerate(gain_target_mv):
                    total_pair = pair_total_cpu[q_idx, mv_idx].item()
                    correct_pair = pair_correct_cpu[q_idx, mv_idx].item()
                    if total_pair > 0:
                        # ann_sign_avg 是有效 teacher 的平均 -1/+1，範圍 [-1,1]，並不是正號百分比。
                        # dynamic_positive_percent 才是動態 gain 為正的比例；應與符號分布一起閱讀 KCI。
                        kci_pair = 100.0 * correct_pair / total_pair
                        ann_sign_avg = pair_ann_sign_cpu[q_idx, mv_idx].item() / total_pair
                        dyn_pos_pct = 100.0 * pair_dyn_pos_cpu[q_idx, mv_idx].item() / total_pair
                        history_gain_pair_rows.append({
                            'epoch': epoch + 1,
                            'target': q_name,
                            'mv': mv_name,
                            'correct': correct_pair,
                            'total': total_pair,
                            'kci_percent': kci_pair,
                            'ann_sign_avg': ann_sign_avg,
                            'dynamic_positive_percent': dyn_pos_pct,
                        })
                        parts.append(
                            f"{mv_name}: {correct_pair:.0f}/{total_pair:.0f}={kci_pair:.1f}% "
                            f"ann_sign_avg={ann_sign_avg:.2f} dyn_pos={dyn_pos_pct:.1f}%"
                        )
                    # 該配對沒有有效 teacher 時存 NaN 與 total=0，避免把無資料誤認為高／低 KCI。
                    else:
                        history_gain_pair_rows.append({
                            'epoch': epoch + 1,
                            'target': q_name,
                            'mv': mv_name,
                            'correct': 0.0,
                            'total': 0.0,
                            'kci_percent': np.nan,
                            'ann_sign_avg': np.nan,
                            'dynamic_positive_percent': np.nan,
                        })
                        parts.append(f"{mv_name}: no valid")
                print(f"    {q_name} | " + " ; ".join(parts))
        # 彙整本 epoch teacher 輸出值的範圍與 <=1e-6 比例，用來察看 teacher 數值狀態。
        ann_clamp_pct = 100.0 * ann_monitor_clamped / ann_monitor_count if ann_monitor_count > 0 else 0.0
        ann_min_str = ', '.join(
            f"{name}={value:.6g}" for name, value in zip(tab_target_cols, ann_monitor_min.detach().cpu().tolist())
        )
        ann_max_str = ', '.join(
            f"{name}={value:.6g}" for name, value in zip(tab_target_cols, ann_monitor_max.detach().cpu().tolist())
        )
        print(
            f"  [ANN Gain Teacher] target min: {ann_min_str} | max: {ann_max_str} | "
            f"<=1e-6: {ann_clamp_pct:.2f}%"
        )
        # 保存跨 epoch 曲線所需純量；這裡的 KCI 是訓練抽到的穩態點，不是 held-out 評估。
        history_losses.append(avg_loss)
        history_kci.append(epoch_kci)
        history_ss_losses.append(avg_ss_loss)

        # 驗證只計算動態逐步 MSE，不計算驗證 SS/Gain；eval+no_grad 停用隨機層與梯度圖。
        # Validation Phase
        dynamic_model.eval()
        val_mse_loss = 0.0
        if valid_loader is not None:
            with torch.no_grad():
                for mse_batch in valid_loader:
                    v_loss = step_wise_rolling_training_step(dynamic_model, mse_batch, criterion, device)
                    val_mse_loss += v_loss.item()
            val_mse_loss /= len(valid_loader)
            print(f"  [Valid] MSE Loss: {val_mse_loss:.6f}")
        # 沒有 valid_loader 時，這個變數改以本 epoch 訓練總 loss 代替，並非真正驗證 MSE。
        else:
            val_mse_loss = avg_loss

        history_val_losses.append(val_mse_loss)

        # 以 val_mse_loss 決定最佳 checkpoint；不以 KCI 最高或訓練總 loss 最低選模型。
        # torch.save 僅存 state_dict 權重，不包含 optimizer 狀態或完整訓練 history。
        out_model_path = f'./saved_models/{exp}.pth'
        if val_mse_loss < best_loss:
            best_loss = val_mse_loss
            epochs_no_improve = 0
            print(f"  [Save] New Best Model saved to {out_model_path} (Val Loss: {best_loss:.6f})")
            torch.save(dynamic_model.state_dict(), out_model_path)
        else:
            epochs_no_improve += 1
            print(f"  [Info] No improvement for {epochs_no_improve} epochs.")

        # 連續 patience 個 epoch 沒改善就結束；最後記憶體中模型不會重新載入最佳 checkpoint。
        if epochs_no_improve >= early_stop_patience:
            print(f"\n[Early Stopping] Triggered after {epoch + 1} epochs without improvement in validation loss.")
            break

    # 訓練結束後輸出本次實際執行的 history；最佳模型已在上方驗證階段儲存。
    out_dir = f'./results/{exp}'
    os.makedirs(out_dir, exist_ok=True)

    # Export compact training diagnostics for loss, KCI, and final pair-level gain signs.
    actual_epochs = len(history_losses)
    epochs_axis = list(range(1, actual_epochs + 1))

    # 同圖畫訓練總 loss 與驗證 MSE；兩條曲線包含的 loss 項不同，閱讀時需看圖例。
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, history_losses, marker='o', color='blue', label='Train Total', linewidth=2)
    if len(history_val_losses) > 0 and valid_loader is not None:
        plt.plot(epochs_axis, history_val_losses, marker='o', color='red', label='Valid MSE', linewidth=2)
    plt.title(f'Training & Validation Loss - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'training_loss_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # 訓練迴圈記錄的 KCI 是 0~1 比例，輸出 CSV／圖表前才乘 100 轉為百分比。
    kci_percentages = [kci * 100 for kci in history_kci]
    kci_df = pd.DataFrame({
        'epoch': epochs_axis,
        'kci_percent': kci_percentages,
    })
    kci_csv_path = os.path.join(out_dir, 'kci_history.csv')
    kci_df.to_csv(kci_csv_path, index=False)

    # SS history 同時保留未加權 loss、權重與加權值；SS 權重為零時仍能查看原始偏差。
    ss_loss_df = pd.DataFrame({
        'epoch': epochs_axis,
        'ss_loss': history_ss_losses,
        'ss_weight': ss_loss_weight,
        'weighted_ss_loss': [ss_loss_weight * value for value in history_ss_losses],
    })
    ss_loss_csv_path = os.path.join(out_dir, 'ss_loss_history.csv')
    ss_loss_df.to_csv(ss_loss_csv_path, index=False)

    # 只有實際產生群組縮放紀錄才輸出 GradNorm history，避免建立無內容的報表。
    group_gradnorm_csv_path = None
    if history_group_gradnorm_rows:
        group_gradnorm_csv_path = os.path.join(out_dir, 'gain_group_gradnorm_history.csv')
        pd.DataFrame(history_group_gradnorm_rows).to_csv(
            group_gradnorm_csv_path, index=False
        )

    # 輸出訓練期間逐 epoch 的 KCI 曲線；它與獨立測試資料的 pair comparison 圖用途不同。
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, kci_percentages, marker='s', color='orange', linewidth=2)
    plt.title(f'PGIN KCI Consistency - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Consistent Steps (%)')
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    kci_plot_path = os.path.join(out_dir, 'kci_consistency_curve.png')
    plt.savefig(kci_plot_path, dpi=300)
    plt.close()

    # 輸出完整逐配對 history，另選最後一個訓練 epoch 畫柱狀圖。
    # 這是最後 epoch 的紀錄，不一定對應先前存下的最佳驗證 checkpoint。
    if history_gain_pair_rows:
        gain_pair_df = pd.DataFrame(history_gain_pair_rows)
        gain_pair_csv_path = os.path.join(out_dir, 'gain_pair_kci_history.csv')
        gain_pair_df.to_csv(gain_pair_csv_path, index=False)

        last_epoch = int(gain_pair_df['epoch'].max())
        last_pair_df = gain_pair_df[gain_pair_df['epoch'] == last_epoch].copy()
        last_pair_df['pair'] = last_pair_df['target'] + '\n' + last_pair_df['mv']

        # 顏色依 KCI 門檻標示；95% 水平線是視覺參考，不參與 loss 或 early stopping。
        plt.figure(figsize=(11, 5))
        colors = ['seagreen' if v >= 95 else 'goldenrod' if v >= 80 else 'crimson'
                  for v in last_pair_df['kci_percent'].fillna(0)]
        bars = plt.bar(last_pair_df['pair'], last_pair_df['kci_percent'].fillna(0), color=colors)
        plt.axhline(95, color='black', linestyle='--', linewidth=1, alpha=0.7, label='95% reference')
        plt.ylim(0, 105)
        plt.ylabel('KCI (%)')
        plt.title(f'Final Gain Direction Consistency - Epoch {last_epoch}')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        # 在每個柱子上加百分比與 n；n 是方向×時間步×有效穩態點的比較數，不是獨立點數。
        for bar, total in zip(bars, last_pair_df['total']):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                min(height + 2, 103),
                f'{height:.1f}%\nn={int(total)}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        plt.tight_layout()
        gain_pair_plot_path = os.path.join(out_dir, 'final_gain_pair_kci_bar.png')
        plt.savefig(gain_pair_plot_path, dpi=300)
        plt.close()

    # 最後列出 checkpoint 與各報表路徑，方便確認產物位置。
    print(f"\n[Done] Model training complete. Weights securely saved to {out_model_path}")
    print(f"[Info] Training metrics exported to {plot_path}")
    print(f"[Info] KCI curve exported to {kci_plot_path}")
    print(f"[Info] KCI CSV exported to {kci_csv_path}")
    print(f"[Info] Steady-state loss history exported to {ss_loss_csv_path}")
    if group_gradnorm_csv_path is not None:
        print(f"[Info] Group GradNorm history exported to {group_gradnorm_csv_path}")
    if history_gain_pair_rows:
        print(f"[Info] Gain pair KCI bar chart exported to {gain_pair_plot_path}")
        print(f"[Info] Gain pair KCI history exported to {gain_pair_csv_path}")

# 只有直接執行此檔案才解析 CLI 並開始工作；被其他模組 import 時不會自動訓練。
# 訓練範例：python scripts/training/train_pgin_stepwise_gain_difficult_group_gradnorm.py --config <YAML路徑> --no-ss
# --gradient-diagnostics-only 則改成載入既有 checkpoint 做梯度診斷。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    # store_true：命令列有 --no-ss 才為 True；效果是覆寫 SS 權重為 0。
    parser.add_argument(
        '--no-ss',
        action='store_true',
        help='Force the steady-state value-loss weight to zero.',
    )
    # CLI 的 target_ratio 預設為 0.015；直接呼叫 main 而不傳值時，函式預設是 0.02。
    # max_scale 只限制各群組的倍率上限，不是整個模型的 gradient clipping 上限。
    parser.add_argument('--target-ratio', type=float, default=0.015)
    parser.add_argument('--max-scale', type=float, default=50.0)
    parser.add_argument('--experiment-suffix', type=str, default=None)
    # 診斷選項：checkpoint 指定權重，batches 限制量測批數，output-dir 指定 CSV 位置。
    parser.add_argument(
        '--gradient-diagnostics-only',
        action='store_true',
        help='Load an existing checkpoint, export gradient-conflict metrics, and exit.',
    )
    parser.add_argument('--diagnostic-checkpoint', type=str, default=None)
    parser.add_argument('--diagnostic-batches', type=int, default=5)
    parser.add_argument(
        '--diagnostic-output-dir',
        type=str,
        default='results/gain_gradient_conflict_diagnostics_ss001',
    )
    # parse_args 取得 Namespace；先檢查診斷批數，再決定本次實驗名稱後綴。
    args = parser.parse_args()
    if args.diagnostic_batches < 1:
        parser.error('--diagnostic-batches must be at least 1.')
    experiment_suffix = args.experiment_suffix
    # 若 YAML 的 exp_name 已含 Difficult-GradNorm 後綴，就不重複加字。
    # --experiment-suffix 明確指定時會直接使用該值；其餘才按 --no-ss 自動決定後綴。
    if experiment_suffix is None:
        with open(args.config, 'r', encoding='utf-8') as f:
            configured_exp_name = str(yaml.safe_load(f).get('exp_name', ''))
        if configured_exp_name.endswith(('_difficult_group_gradnorm', '_difficult_group_gradnorm_no_ss')):
            experiment_suffix = ''
        else:
            experiment_suffix = (
                '_difficult_group_gradnorm_no_ss'
                if args.no_ss
                else '_difficult_group_gradnorm'
            )
    # CLI 固定啟用 DIFFICULT_PAIRS 的群組縮放並使用八份 baseline 穩態訓練來源。
    # 因此此入口不只是讀 YAML；還會把命令列倍率、來源與可選的 SS=0 override 傳給 main。
    main(
        args.config,
        gradient_diagnostics_only=args.gradient_diagnostics_only,
        diagnostic_checkpoint=args.diagnostic_checkpoint,
        diagnostic_batches=args.diagnostic_batches,
        diagnostic_output_dir=args.diagnostic_output_dir,
        experiment_suffix=experiment_suffix,
        h2s_group_target_ratio=args.target_ratio,
        h2s_group_max_scale=args.max_scale,
        steady_state_source_override=BASELINE_STEADY_STATE_SOURCES,
        gain_group_gradnorm_pairs=DIFFICULT_PAIRS,
        ss_loss_weight_override=0.0 if args.no_ss else None,
    )
