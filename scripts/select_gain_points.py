# scripts/select_gain_points.py - Select rows near specified MV pairs and flow groups for process gain validation
# 主要是為了驗證 claus 製程的 process gain 使用
import argparse
import os
import sys
from typing import List, Tuple
import pandas as pd
import numpy as np

# ------------------------------ Helpers ------------------------------

def _load_from_path(path: str) -> pd.DataFrame:
    """Load a single CSV/XLS/XLSX file (all sheets merged for Excel)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # Read all sheets and concat
        sheets = pd.read_excel(path, sheet_name=None)
        if not sheets:
            raise ValueError(f"No sheets found in Excel file: {path}")
        return pd.concat(list(sheets.values()), ignore_index=True)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_input(input_path: str) -> pd.DataFrame:
    """Allow directory (merge all CSV/XLS/XLSX) or a single file."""
    if os.path.isdir(input_path):
        frames: List[pd.DataFrame] = []
        for root, _, files in os.walk(input_path):
            for fn in files:
                if fn.lower().endswith((".csv", ".xlsx", ".xls")):
                    fp = os.path.join(root, fn)
                    frames.append(_load_from_path(fp))
        if not frames:
            raise FileNotFoundError(f"No CSV/XLS/XLSX files found under: {input_path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = _load_from_path(input_path)
    # Normalize column names (strip) only
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_pairs(pairs_str: str) -> List[Tuple[float, float]]:
    pairs = []
    for token in pairs_str.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' in token:
            a, b = token.split(':', 1)
        elif ';' in token:
            a, b = token.split(';', 1)
        else:
            # support format like "(270,230)"
            token = token.strip('()')
            a, b = token.split(',', 1)
        pairs.append((float(a), float(b)))
    return pairs


def _nearest_flow_group(df: pd.DataFrame, flow_col: str, target_flow: float, tol: float | None) -> pd.DataFrame:
    if tol is not None and tol > 0:
        sub = df[np.abs(df[flow_col] - target_flow) <= tol]
        if len(sub) > 0:
            return sub
    # fallback: pick rows at the nearest available flow value
    diffs = np.abs(df[flow_col] - target_flow)
    min_diff = diffs.min()
    return df[diffs <= (min_diff + 1e-9)]


def _pick_base(sub: pd.DataFrame, air2_col: str, t2_col: str, a: float, t: float) -> int:
    d2 = (sub[air2_col] - a) ** 2 + (sub[t2_col] - t) ** 2
    return int(d2.idxmin())


def _pick_below_air2(sub: pd.DataFrame, air2_col: str, t2_col: str, base_a: float, base_t: float, used: set, fix_tol: float) -> int | None:
    """Pick row with air2 just below base_a while fixing t2 at base_t (within tol).
    Fallbacks: if no exact/within-tol match for t2, use nearest t2; if no below air2, use smallest above.
    """
    # Fix T2 around base_t
    if fix_tol > 0:
        fixed = sub[(sub[t2_col] - base_t).abs() <= fix_tol]
    else:
        fixed = sub[sub[t2_col] == base_t]
    if fixed.empty:
        # nearest t2 fallback
        t2diff = (sub[t2_col] - base_t).abs()
        min_t2 = t2diff.min()
        fixed = sub[t2diff <= (min_t2 + 1e-9)]

    # Prefer air2 < base_a with max air2
    below = fixed[fixed[air2_col] < base_a]
    if not below.empty:
        target_a = below[air2_col].max()
        cand = below[below[air2_col] == target_a]
    else:
        # fallback: smallest air2 above base
        above = fixed[fixed[air2_col] > base_a]
        if above.empty:
            return None
        target_a = above[air2_col].min()
        cand = above[above[air2_col] == target_a]

    # choose one not used (nearest t2 to base_t)
    cand = cand.drop(index=[i for i in cand.index if i in used], errors='ignore')
    if cand.empty:
        return None
    idx = int((cand[t2_col] - base_t).abs().idxmin())
    return idx


def _pick_below_t2(sub: pd.DataFrame, air2_col: str, t2_col: str, base_a: float, base_t: float, used: set, fix_tol: float) -> int | None:
    """Pick row with t2 just below base_t while fixing air2 at base_a (within tol).
    Fallbacks: if no exact/within-tol match for air2, use nearest air2; if no below t2, use smallest above.
    """
    # Fix air2 around base_a
    if fix_tol > 0:
        fixed = sub[(sub[air2_col] - base_a).abs() <= fix_tol]
    else:
        fixed = sub[sub[air2_col] == base_a]
    if fixed.empty:
        # nearest air2 fallback
        a_diff = (sub[air2_col] - base_a).abs()
        min_a = a_diff.min()
        fixed = sub[a_diff <= (min_a + 1e-9)]

    # Prefer t2 < base_t with max t2
    below = fixed[fixed[t2_col] < base_t]
    if not below.empty:
        target_t = below[t2_col].max()
        cand = below[below[t2_col] == target_t]
    else:
        # fallback: smallest t2 above
        above = fixed[fixed[t2_col] > base_t]
        if above.empty:
            return None
        target_t = above[t2_col].min()
        cand = above[above[t2_col] == target_t]

    # choose one not used (nearest air2 to base_a)
    cand = cand.drop(index=[i for i in cand.index if i in used], errors='ignore')
    if cand.empty:
        return None
    idx = int((cand[air2_col] - base_a).abs().idxmin())
    return idx


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Select rows near specified MV pairs and flow groups for process gain validation.")
    ap.add_argument("--input",
                    default=r"C:\\Users\\Administrator\\Desktop\\digital_twin_revised\\data\\my_own_data\\air2(8.13~17.4219_40pts)t2(140~240_40pts)acid_gas(121~160_3pts).csv",
                    help="Path to CSV/XLSX file or a directory containing them (default: preset CSV file)")
    ap.add_argument("--output", default=None, help="Output CSV path (default: selected_gain_points.csv next to input)")
    ap.add_argument("--air2-col", default="air2_SP_m3", help="Column name for air2 MV (default: air2_SP_m3)")
    ap.add_argument("--t2-col", default="HEATER2_output_T_SP", help="Column name for T2 MV (default: HEATER2_output_T_SP)")
    ap.add_argument("--flow-col", default="acidgas_Fm", help="Column name for acid gas flow (default: acidgas_Fm)")
    ap.add_argument("--flows", default="121,140.5,161", help="Comma-separated target flows")
    ap.add_argument("--pairs", default="270:230,270:178,270:155,223:155,190:155,190:178,190:230,223:230", help="Comma-separated MV pairs as air2:t2")
    ap.add_argument("--flow-tol", type=float, default=0.0, help="Tolerance on flow matching. If >0, rows with |flow-target|<=tol are used; else nearest flow group is used")
    ap.add_argument("--fix-tol", type=float, default=0.0, help="Tolerance to fix the other MV at base value (0 => exact match)")
    args = ap.parse_args()

    df = _load_input(args.input)

    # Validate columns
    needed = {args.air2_col, args.t2_col, args.flow_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[Error] Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)[:50]} ...")
        sys.exit(1)

    # Coerce to numeric for relevant columns
    for c in [args.air2_col, args.t2_col, args.flow_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[args.air2_col, args.t2_col, args.flow_col])

    flows = [float(x.strip()) for x in args.flows.split(',') if x.strip()]
    pairs = _parse_pairs(args.pairs)

    selections: List[pd.DataFrame] = []

    for f in flows:
        sub_flow = _nearest_flow_group(df, args.flow_col, f, tol=args.flow_tol)
        if sub_flow.empty:
            print(f"[Warn] No rows near flow {f}; skipping this flow group")
            continue
        for (a, t) in pairs:
            used_idx: set[int] = set()
            # base (closest to specified pair)
            base_idx = _pick_base(sub_flow, args.air2_col, args.t2_col, a, t)
            used_idx.add(base_idx)
            base_a = float(sub_flow.loc[base_idx, args.air2_col])
            base_t = float(sub_flow.loc[base_idx, args.t2_col])
            base_row = sub_flow.loc[[base_idx]].copy()
            base_row['selection_type'] = 'base'
            base_row['target_air2'] = a
            base_row['target_t2'] = t
            base_row['target_flow'] = f
            base_row['base_air2'] = base_a
            base_row['base_t2'] = base_t
            selections.append(base_row)

            # air2 below base, with t2 fixed at base
            idx2 = _pick_below_air2(sub_flow, args.air2_col, args.t2_col, base_a, base_t, used=used_idx, fix_tol=args.fix_tol)
            if idx2 is not None:
                used_idx.add(idx2)
                row2 = sub_flow.loc[[idx2]].copy()
                row2['selection_type'] = 'air2_below'
                row2['target_air2'] = a
                row2['target_t2'] = t
                row2['target_flow'] = f
                row2['base_air2'] = base_a
                row2['base_t2'] = base_t
                selections.append(row2)
            else:
                print(f"[Info] Could not find air2_below (fixed t2 at base) for pair ({a},{t}) at flow {f}")

            # t2 below base, with air2 fixed at base
            idx3 = _pick_below_t2(sub_flow, args.air2_col, args.t2_col, base_a, base_t, used=used_idx, fix_tol=args.fix_tol)
            if idx3 is not None:
                used_idx.add(idx3)
                row3 = sub_flow.loc[[idx3]].copy()
                row3['selection_type'] = 't2_below'
                row3['target_air2'] = a
                row3['target_t2'] = t
                row3['target_flow'] = f
                row3['base_air2'] = base_a
                row3['base_t2'] = base_t
                selections.append(row3)
            else:
                print(f"[Info] Could not find t2_below (fixed air2 at base) for pair ({a},{t}) at flow {f}")

    if not selections:
        print("[Error] No rows selected. Check your inputs, column names, and tolerances.")
        sys.exit(1)

    out_df = pd.concat(selections, ignore_index=True)

    # Order columns: metadata first
    meta_cols = ['selection_type', 'target_flow', 'target_air2', 'target_t2', 'base_air2', 'base_t2']
    remaining = [c for c in out_df.columns if c not in meta_cols]
    out_df = out_df[meta_cols + remaining]

    # Output path
    if args.output:
        out_path = args.output
    else:
        base_dir = os.path.dirname(os.path.abspath(args.input)) if not os.path.isdir(args.input) else args.input
        out_path = os.path.join(base_dir, 'selected_gain_points.csv')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Summary
    print(f"[OK] Selected {len(out_df)} rows.")
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
