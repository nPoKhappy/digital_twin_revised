"""
比較三組數據的分佈：
  1. Training Data     - Aspen 模擬訓練數據
  2. Plant Simulated   - 本次生成的工廠分佈模擬數據 (6 個檔案)
  3. W251 Plant Data   - 實際現場數據

注意事項：
  - Plant Simulated 的 acidgas_Fv 直接使用（已是體積流率 m³/hr）
  - Plant Simulated 的 air 是 kmol/hr，使用密度 0.0401 kmol/m³ 轉成 m³/hr
  - Plant Simulated 過濾掉 acidgas_Fv == 0 的 padding rows
"""

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── 路徑設定 ────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIM_DIR   = os.path.join(REPO_ROOT, "data", "Claus_dynamic", "plant_simulated_data")
W251_PATH = os.path.join(REPO_ROOT, "data", "Claus_dynamic", "Claus_plant_data",
                         "W251_500area_processed_Training_Data.csv")
TRAIN_FILES = [
    os.path.join(REPO_ROOT, "data", "Claus_dynamic", f)
    for f in [
        "Test_dataform_change_air2_R=5_converted.csv",
        "Test_dataform_change_air2_R=5-1_converted.csv",
        "Test_dataform_change_air2_R=5-2_converted.csv",
        "Test_dataform_change_air2_R=5-6_converted.csv",
    ]
]

OUT_DIR = os.path.join(REPO_ROOT, "results", "plant_sim_distribution")
os.makedirs(OUT_DIR, exist_ok=True)


# ── 字型（避免中文亂碼）──────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════════
# 1. 載入資料
# ════════════════════════════════════════════════════════════

# --- 1a. Plant Simulated ---
print("[1/3] 載入 Plant Simulated Data ...")
sim_dfs = []
for fp in sorted(glob.glob(os.path.join(SIM_DIR, "*.csv"))):
    df = pd.read_csv(fp)
    df["_file"] = os.path.basename(fp)
    sim_dfs.append(df)

df_sim = pd.concat(sim_dfs, ignore_index=True)
# 過濾 padding rows（acidgas_Fv == 0）
before = len(df_sim)
df_sim = df_sim[df_sim["acidgas_Fv"] > 0].copy()
print(f"   原始 {before:,} 行 → 過濾後 {len(df_sim):,} 行 (移除 {before-len(df_sim):,} 個 padding rows)")

df_sim["Source"] = "Plant Simulated"

# --- 1b. Training Data (Aspen) ---
print("[2/3] 載入 Training Data (Aspen) ...")
train_dfs = [pd.read_csv(fp) for fp in TRAIN_FILES if os.path.exists(fp)]
df_train = pd.concat(train_dfs, ignore_index=True).dropna()
df_train["Source"] = "Training (Aspen)"
print(f"   {len(df_train):,} 行")

# --- 1c. W251 Plant Data ---
print("[3/3] 載入 W251 Plant Data ...")
df_w251 = pd.read_csv(W251_PATH).dropna()
df_w251["Source"] = "W251 Plant"
# W251 acidgas_P 是 mbar(g)，轉換為 barA: (mbar_g + 1013.25) / 1000
if "acidgas_P" in df_w251.columns:
    df_w251["acidgas_P"] = (pd.to_numeric(df_w251["acidgas_P"], errors="coerce") + 1013.25) / 1000.0
# W251 cat2_input_temp 對應 HEATER2_output_T_PV
if "cat2_input_temp" in df_w251.columns:
    df_w251["HEATER2_output_T_PV"] = pd.to_numeric(df_w251["cat2_input_temp"], errors="coerce")
print(f"   {len(df_w251):,} 行")

# ════════════════════════════════════════════════════════════
# 2. 定義比較變數（含顯示名稱與單位）
# ════════════════════════════════════════════════════════════
# Format: (display_name, unit, sim_col, train_col, w251_col)
# 與 analyze_distribution_compare.py 相同的 5 個變數
VARS = [
    ("Acid Gas Flow",              "m3/hr", "acidgas_Fv",          "acidgas_Fm",          "acidgas_Fm"),
    ("Acid Gas Temp",              "deg C", "acidgas_T",           "acidgas_T",           "acidgas_T"),
    ("Acid Gas Pressure",          "barA",  "acidgas_P",           "acidgas_P",           "acidgas_P"),
    ("Secondary Air",              "m3/hr", "second_air2",         "second_air2",         "second_air2"),
    ("HEATER2 Out Temp (pre-Cat2)","deg C", "HEATER2_output_T_PV", "HEATER2_output_T_PV", "HEATER2_output_T_PV"),
]

# ════════════════════════════════════════════════════════════
# 3. 統計摘要
# ════════════════════════════════════════════════════════════
records = []
for (name, unit, sc, tc, wc) in VARS:
    for label, df, col in [("Plant Simulated", df_sim, sc),
                            ("Training (Aspen)", df_train, tc),
                            ("W251 Plant",       df_w251, wc)]:
        if col is None or col not in df.columns:
            continue
        s = df[col].dropna()
        records.append({
            "Variable": name,
            "Unit": unit,
            "Source": label,
            "Mean":   s.mean(),
            "Std":    s.std(),
            "Min":    s.min(),
            "P25":    s.quantile(0.25),
            "Median": s.median(),
            "P75":    s.quantile(0.75),
            "Max":    s.max(),
        })

df_summary = pd.DataFrame(records)
summary_path = os.path.join(OUT_DIR, "summary_statistics.csv")
df_summary.to_csv(summary_path, index=False)
print(f"\n統計摘要已儲存至: {summary_path}")
print(df_summary[["Variable","Source","Mean","Std","Median"]].to_string(index=False))

# ════════════════════════════════════════════════════════════
# 4. KDE 分佈圖
# ════════════════════════════════════════════════════════════
PALETTE = {
    "Plant Simulated": "#E74C3C",    # 紅
    "Training (Aspen)": "#2980B9",   # 藍
    "W251 Plant":       "#27AE60",   # 綠
}
ALPHA = 0.35

n = len(VARS)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
axes = axes.flatten()

for idx, (name, unit, sc, tc, wc) in enumerate(VARS):
    ax = axes[idx]
    plotted = False

    for label, df, col in [("Plant Simulated", df_sim, sc),
                             ("Training (Aspen)", df_train, tc),
                             ("W251 Plant",       df_w251, wc)]:
        if col is None or col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        sns.kdeplot(s, ax=ax, label=label, color=PALETTE[label],
                    fill=True, alpha=ALPHA, linewidth=2, bw_adjust=1.2)
        plotted = True

    ax.set_title(f"{name}\n({unit})", fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("密度" if idx % ncols == 0 else "")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

# 關掉多餘子圖
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("分佈比較：Plant Simulated vs Training (Aspen) vs W251 Plant\n(KDE)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
kde_path = os.path.join(OUT_DIR, "distribution_kde.png")
plt.savefig(kde_path, dpi=150, bbox_inches='tight')
print(f"\nKDE 圖已儲存至: {kde_path}")
plt.close()

# ════════════════════════════════════════════════════════════
# 5. Boxplot 圖
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
axes = axes.flatten()

for idx, (name, unit, sc, tc, wc) in enumerate(VARS):
    ax = axes[idx]
    data_dict = {}
    for label, df, col in [("Plant Simulated", df_sim, sc),
                             ("Training (Aspen)", df_train, tc),
                             ("W251 Plant",       df_w251, wc)]:
        if col is None or col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) > 0:
            data_dict[label] = s.values

    if not data_dict:
        axes[idx].set_visible(False)
        continue

    positions = list(range(len(data_dict)))
    labels_order = list(data_dict.keys())
    bp = ax.boxplot([data_dict[k] for k in labels_order],
                    positions=positions,
                    showfliers=False,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    widths=0.5)

    for patch, key in zip(bp['boxes'], labels_order):
        patch.set_facecolor(PALETTE[key])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([k.replace(" (Aspen)", "\n(Aspen)").replace("Plant ", "Plant\n")
                        for k in labels_order], fontsize=8)
    ax.set_title(f"{name}\n({unit})", fontsize=11, fontweight='bold')
    ax.set_ylabel("")
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("分佈比較：Plant Simulated vs Training (Aspen) vs W251 Plant\n(Boxplot，不含離群值)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
box_path = os.path.join(OUT_DIR, "distribution_boxplot.png")
plt.savefig(box_path, dpi=150, bbox_inches='tight')
print(f"Boxplot 圖已儲存至: {box_path}")
plt.close()

# ════════════════════════════════════════════════════════════
# 6. Violin Plot（同時顯示分佈形狀與中位數）
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
axes = axes.flatten()

for idx, (name, unit, sc, tc, wc) in enumerate(VARS):
    ax = axes[idx]
    rows = []
    for label, df, col in [("Plant Simulated", df_sim, sc),
                             ("Training (Aspen)", df_train, tc),
                             ("W251 Plant",       df_w251, wc)]:
        if col is None or col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        tmp = pd.DataFrame({"value": s.values, "Source": label})
        rows.append(tmp)

    if not rows:
        axes[idx].set_visible(False)
        continue

    df_plot = pd.concat(rows, ignore_index=True)
    order = [k for k in ["Plant Simulated", "Training (Aspen)", "W251 Plant"]
             if k in df_plot["Source"].unique()]
    palette = {k: PALETTE[k] for k in order}

    sns.violinplot(data=df_plot, x="Source", y="value", ax=ax,
                   order=order, hue="Source", palette=palette, legend=False,
                   inner="quartile", density_norm="width", cut=0)

    ax.set_title(f"{name}\n({unit})", fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(range(len(order)))
    tick_labels = [k.replace(" (Aspen)", "\n(Aspen)").replace("W251 Plant", "W251\nPlant")
                   for k in order]
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("分佈比較：Plant Simulated vs Training (Aspen) vs W251 Plant\n(Violin Plot)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
violin_path = os.path.join(OUT_DIR, "distribution_violin.png")
plt.savefig(violin_path, dpi=150, bbox_inches='tight')
print(f"Violin 圖已儲存至: {violin_path}")
plt.close()

print("\n[Done] 全部圖表完成。輸出資料夾:", OUT_DIR)
