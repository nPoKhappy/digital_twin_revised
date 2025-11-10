import argparse
import os
import yaml
import pandas as pd
import numpy as np
import torch
from typing import Dict, List

from src.models.tabular_mlp import TabularMLP


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def zscore_fit(df: pd.DataFrame, cols: List[str]):
    s = df[cols].astype(float)
    mean = s.mean()
    std = s.std() + 1e-8
    return mean, std


def zscore_apply(df: pd.DataFrame, mean: pd.Series, std: pd.Series, cols: List[str]):
    s = df[cols].astype(float)
    return (s - mean[cols]) / std[cols]


def ensure_input_columns(df: pd.DataFrame, input_cols: List[str]) -> pd.DataFrame:
    # Allow fallback from air2_SP_m3 -> air2_SP if needed
    if 'air2_SP' in input_cols and 'air2_SP' not in df.columns and 'air2_SP_m3' in df.columns:
        df = df.copy()
        df['air2_SP'] = df['air2_SP_m3']
    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns in selected data: {missing}")
    return df


def build_model(cfg: Dict, num_features: int, num_outputs: int, device: str) -> TabularMLP:
    mcfg = cfg['model']
    model = TabularMLP(
        num_features=num_features,
        num_outputs=num_outputs,
        hidden_dims=mcfg.get('hidden_dims', [128, 64]),
        dropout=mcfg.get('dropout', 0.1),
        activation=mcfg.get('activation', 'relu')
    ).to(device)
    exp = cfg['exp_name']
    ckpt = os.path.join('./saved_models', f'{exp}_tabular_mlp.pth')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_rows(model: TabularMLP, Xz: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        X = torch.tensor(Xz, dtype=torch.float32, device=device)
        Yz = model(X).cpu().numpy()
    return Yz


def main():
    ap = argparse.ArgumentParser(description='Compare model-derived gains vs. ground-truth on selected gain points')
    ap.add_argument('--config', default='configs/tabular_mlp_claus.yaml')
    ap.add_argument('--selected', default='data/my_own_data/selected_gain_points.csv')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tab = cfg['tabular']
    input_cols: List[str] = list(tab['input_cols'])
    # Use the full trained targets to build the model, but only compare Total_S
    trained_target_cols: List[str] = list(tab['target_cols'])
    if 'Total_S' not in trained_target_cols:
        raise ValueError("Config tabular.target_cols 不包含 'Total_S'，請確認設定與資料欄位")
    comp_target = 'Total_S'
    comp_target_idx = trained_target_cols.index(comp_target)

    data_cfg = cfg['data']
    train_csv = os.path.join(data_cfg['path'], data_cfg['filename'])
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    # Load training stats for z-score
    df_train = pd.read_csv(train_csv)
    # keep only necessary columns and dropna
    needed_cols = [c for c in input_cols + trained_target_cols if c in df_train.columns]
    missing_targets = [c for c in trained_target_cols if c not in df_train.columns]
    if missing_targets:
        print(f"[Warn] Training CSV missing target columns (will skip in comparison): {missing_targets}")
    df_train = df_train[needed_cols].dropna()

    mean_all, std_all = zscore_fit(df_train, needed_cols)

    # Load selected points
    sel_path = args.selected
    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"Selected CSV not found: {sel_path}")
    df_sel = pd.read_csv(sel_path)

    # Ensure inputs are available (map air2_SP_m3 -> air2_SP if needed)
    df_sel = ensure_input_columns(df_sel, input_cols)

    # Device
    device = 'cuda' if (torch.cuda.is_available() and cfg['training'].get('device', 'cpu') == 'cuda') else 'cpu'

    # Build model with full number of trained targets
    model = build_model(cfg, num_features=len(input_cols), num_outputs=len(trained_target_cols), device=device)

    # Group by pair and flow
    key_cols = ['target_flow', 'target_air2', 'target_t2']
    for k in key_cols:
        if k not in df_sel.columns:
            raise ValueError(f"Selected CSV missing required key column: {k}")

    results = []

    # Precompute inverse scale for targets (all trained targets)
    # 抽出目標欄位的均值與標準差，用於反標準化
    t_mean = mean_all.reindex(trained_target_cols)
    t_std = std_all.reindex(trained_target_cols)

    # Iterate groups
    for keys, grp in df_sel.groupby(key_cols, dropna=False):
        # Expect selection_type in {base, air2_below, t2_below}
        if 'selection_type' not in grp.columns:
            raise ValueError('Selected CSV must have selection_type column')
        try:
            base_row = grp.loc[grp['selection_type'] == 'base'].iloc[0]
        except IndexError:
            print(f"[Warn] Missing base for group {keys}; skip")
            continue
        try:
            a_below_row = grp.loc[grp['selection_type'] == 'air2_below'].iloc[0]
        except IndexError:
            a_below_row = None
        try:
            t_below_row = grp.loc[grp['selection_type'] == 't2_below'].iloc[0]
        except IndexError:
            t_below_row = None
        if a_below_row is None and t_below_row is None:
            print(f"[Warn] No below rows for group {keys}; skip")
            continue

        # Prepare model inputs for available rows
        rows = [('base', base_row)] + ([('air2_below', a_below_row)] if a_below_row is not None else []) + ([('t2_below', t_below_row)] if t_below_row is not None else [])
        X = []
        order = []
        for tag, r in rows:
            vals = []
            for c in input_cols:
                if c not in mean_all.index:
                    raise ValueError(f"Training stats missing column: {c}")
                vals.append(float(r[c]))
            X.append(vals)
            order.append(tag)
        X = np.array(X, dtype=float)

        # z-score inputs with training stats
        in_mean = mean_all.reindex(input_cols)
        in_std = std_all.reindex(input_cols)
        Xz = (X - in_mean.values) / in_std.values

        # Predict normalized targets, then inverse using trained target stats
        Yz = predict_rows(model, Xz, device=device)
        Y = Yz * t_std.values + t_mean.values  # back to original scale

        # Map predictions by tag
        pred_map = {tag: Y[i, :] for i, tag in enumerate(order)}

        # Ground truth outputs present in selected CSV?
        have_truth = (comp_target in df_sel.columns)

        def calc_gains(base_vec, other_vec):
            if base_vec is None or other_vec is None:
                return None
            return base_vec - other_vec

        base_pred = pred_map.get('base')
        a_pred = pred_map.get('air2_below')
        t_pred = pred_map.get('t2_below')

        a_gain_model = calc_gains(base_pred, a_pred)
        t_gain_model = calc_gains(base_pred, t_pred)

        # Model Total_S for each point (after inverse scaling)
        base_total_s_model = float(base_pred[comp_target_idx]) if base_pred is not None else np.nan
        air2_total_s_model = float(a_pred[comp_target_idx]) if a_pred is not None else np.nan
        t2_total_s_model = float(t_pred[comp_target_idx]) if t_pred is not None else np.nan

        # True Total_S for each point (from selected CSV)
        base_total_s_true = float(base_row.get(comp_target, np.nan))
        air2_total_s_true = float(a_below_row.get(comp_target, np.nan)) if a_below_row is not None else np.nan
        t2_total_s_true = float(t_below_row.get(comp_target, np.nan)) if t_below_row is not None else np.nan

        if have_truth:
            y = comp_target
            base_true = np.array([float(base_row.get(y, np.nan))], dtype=float)
            a_true = np.array([float(a_below_row.get(y, np.nan))], dtype=float) if a_below_row is not None else np.array([np.nan])
            t_true = np.array([float(t_below_row.get(y, np.nan))], dtype=float) if t_below_row is not None else np.array([np.nan])
            a_gain_true = calc_gains(base_true, a_true) if a_below_row is not None else None
            t_gain_true = calc_gains(base_true, t_true) if t_below_row is not None else None
        else:
            a_gain_true = None
            t_gain_true = None

        # Build record (Total_S only)
        rec = {
            'target_flow': keys[0],
            'target_air2': keys[1],
            'target_t2': keys[2],
        }

        # Also keep model outputs for each point
        rec['Total_S_model_base'] = base_total_s_model
        rec['Total_S_model_air2'] = air2_total_s_model
        rec['Total_S_model_t2'] = t2_total_s_model
        # And true outputs for each point
        rec['Total_S_true_base'] = base_total_s_true
        rec['Total_S_true_air2'] = air2_total_s_true
        rec['Total_S_true_t2'] = t2_total_s_true

        # MV deltas (true and model; model equals the inputs used)
        mv_air2 = 'air2_SP' if 'air2_SP' in df_sel.columns else ('air2_SP_m3' if 'air2_SP_m3' in df_sel.columns else 'air2_SP')
        mv_t2 = 'HEATER2_output_T_SP'
        rec['delta_air2_true'] = (float(base_row.get(mv_air2, np.nan)) - float(a_below_row.get(mv_air2, np.nan))) if a_below_row is not None else np.nan
        rec['delta_t2_true'] = (float(base_row.get(mv_t2, np.nan)) - float(t_below_row.get(mv_t2, np.nan))) if t_below_row is not None else np.nan
        # model uses the same input rows, so equal to true
        rec['delta_air2_model'] = rec['delta_air2_true']
        rec['delta_t2_model'] = rec['delta_t2_true']

        # Total_S deltas/gains (model and true)
        y_idx = comp_target_idx  # index of Total_S in trained targets
        mg_a = a_gain_model[y_idx] if a_gain_model is not None else np.nan
        mg_t = t_gain_model[y_idx] if t_gain_model is not None else np.nan
        rec['Total_S_model_air2_delta'] = mg_a
        rec['Total_S_model_t2_delta'] = mg_t
        # per-unit model gains (ΔTotal_S / ΔMV)
        rec['Total_S_model_air2_gain'] = (mg_a / rec['delta_air2_true']) if (pd.notna(mg_a) and pd.notna(rec['delta_air2_true']) and abs(float(rec['delta_air2_true'])) > 1e-12) else np.nan
        rec['Total_S_model_t2_gain'] = (mg_t / rec['delta_t2_true']) if (pd.notna(mg_t) and pd.notna(rec['delta_t2_true']) and abs(float(rec['delta_t2_true'])) > 1e-12) else np.nan

        if a_gain_true is not None and not np.isnan(a_gain_true[0]):
            tg_a = a_gain_true[0]
            rec['Total_S_true_air2_delta'] = tg_a
            # per-unit true air2 gain
            rec['Total_S_true_air2_gain'] = (tg_a / rec['delta_air2_true']) if (pd.notna(tg_a) and pd.notna(rec['delta_air2_true']) and abs(float(rec['delta_air2_true'])) > 1e-12) else np.nan
            rec['air2_sign_match'] = np.sign(mg_a) == np.sign(tg_a)
        else:
            rec['Total_S_true_air2_delta'] = np.nan
            rec['Total_S_true_air2_gain'] = np.nan
            rec['air2_sign_match'] = np.nan

        if t_gain_true is not None and not np.isnan(t_gain_true[0]):
            tg_t = t_gain_true[0]
            rec['Total_S_true_t2_delta'] = tg_t
            # per-unit true t2 gain
            rec['Total_S_true_t2_gain'] = (tg_t / rec['delta_t2_true']) if (pd.notna(tg_t) and pd.notna(rec['delta_t2_true']) and abs(float(rec['delta_t2_true'])) > 1e-12) else np.nan
            rec['t2_sign_match'] = np.sign(mg_t) == np.sign(tg_t)
        else:
            rec['Total_S_true_t2_delta'] = np.nan
            rec['Total_S_true_t2_gain'] = np.nan
            rec['t2_sign_match'] = np.nan

        results.append(rec)

        # Print model vs true at the same positions ONLY when sign mismatches
        def _fmt(v):
            return f"{v:.6g}" if pd.notna(v) else 'nan'
        air2_mismatch = (rec['air2_sign_match'] == False)
        t2_mismatch = (rec['t2_sign_match'] == False)
        if air2_mismatch or t2_mismatch:
            tags = []
            if air2_mismatch:
                tags.append('air2_sign_mismatch')
            if t2_mismatch:
                tags.append('t2_sign_mismatch')
            tag_str = ','.join(tags)
            print(
                f"[flow={keys[0]}, air2={keys[1]}, t2={keys[2]} | {tag_str}] "
                f"ΔTotal_S_air2(model/true)=({_fmt(rec['Total_S_model_air2_delta'])}/{_fmt(rec['Total_S_true_air2_delta'])}); "
                f"ΔTotal_S_t2(model/true)=({_fmt(rec['Total_S_model_t2_delta'])}/{_fmt(rec['Total_S_true_t2_delta'])})"
            )

    out_df = pd.DataFrame(results)

    # Output path
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join('results', cfg['exp_name'])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'gain_comparison.csv')

    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path} ({len(out_df)})")


if __name__ == '__main__':
    main()
