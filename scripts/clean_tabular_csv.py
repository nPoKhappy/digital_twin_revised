# clean and standardize tabular CSV/Excel files for ML use
import argparse
import os
import re
import sys
import pandas as pd
import csv

# Normalize a header string: strip quotes/whitespace/newlines, collapse spaces
def _normalize_header(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    s = s.replace('\r', ' ').replace('\n', ' ')
    s = s.strip().strip('"\'')
    s = re.sub(r'\s+', ' ', s)
    return s

# Canonical key for fuzzy matching (lowercase and keep only a-z0-9 and underscores)
def _canon_key(s: str) -> str:
    s = _normalize_header(s).lower()
    s = s.replace(' ', '_')
    s = re.sub(r'[^a-z0-9_]+', '', s)
    return s

# Build a fuzzy map from existing columns to canonical names
def build_fuzzy_map(existing_cols, desired_map):
    # desired_map: dict canonical_key -> final_name
    out = {}
    for col in existing_cols:
        key = _canon_key(col)
        if key in desired_map:
            out[col] = desired_map[key]
    return out

# Default desired columns for your use case (include synonyms)
DEFAULT_CANONICAL = {
    'acidgas_fm': 'acidgas_Fm',
    'airflow': 'acidgas_Fm',  # common synonym seen in your file
    'acidgas_t': 'acidgas_T',
    'acidgas_p': 'acidgas_P',
    'air2_sp': 'air2_SP',
    'heater2_output_t_sp': 'HEATER2_output_T_SP',
    # tail gas analyzers
    'b35_h2s': 'B35_H2S',
    'h2sout': 'B35_H2S',
    'h2s_out': 'B35_H2S',
    'b35_so2': 'B35_SO2',
    'so2out': 'B35_SO2',
    'so2_out': 'B35_SO2',
    # optional
    'total_s': 'Total_S',
}


def _load_table(path: str, encoding: str) -> pd.DataFrame:
    # Prefer Excel if provided
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path, engine='openpyxl' if ext == '.xlsx' else None)

    # Try strict CSV first
    try:
        return pd.read_csv(path, engine='python', encoding=encoding)
    except Exception as e:
        print(f"[Error] read_csv failed: {e}\nTrying robust CSV fallback...")

    # Robust fallback: no quoting, skip malformed lines, auto sep
    try:
        return pd.read_csv(
            path,
            engine='python',
            encoding=encoding,
            sep=None,  # infer
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            on_bad_lines='skip',
            dtype=str,  # read as str to avoid parser chokes, we'll let pandas infer later if needed
        )
    except Exception as e2:
        print(f"[Error] robust read_csv failed: {e2}\nTrying latin1 fallback with the same settings...")

    # Last resort with latin1
    return pd.read_csv(
        path,
        engine='python',
        encoding='latin1',
        sep=None,
        quoting=csv.QUOTE_NONE,
        escapechar='\\',
        on_bad_lines='skip',
        dtype=str,
    )


def main():
    p = argparse.ArgumentParser(description='Clean CSV/Excel headers with optional last-N split.')
    p.add_argument('--input', required=True, help='Path to raw CSV/XLSX')
    p.add_argument('--output', required=True, help='Path to cleaned CSV (full data)')
    p.add_argument('--encoding', default='utf-8', help='File encoding for CSV (try cp950/big5 if needed)')
    p.add_argument('--test-last', type=int, default=0, help='If >0, also write training.csv (exclude last N) and testing.csv (last N) beside output')
    args = p.parse_args()

    df = _load_table(args.input, args.encoding)

    # Clean headers
    clean_cols = [_normalize_header(c) for c in df.columns]
    df.columns = clean_cols

    # Fuzzy rename to canonical if possible
    fuzzy_map = build_fuzzy_map(df.columns, DEFAULT_CANONICAL)
    if fuzzy_map:
        df = df.rename(columns=fuzzy_map)

    # Save cleaned full CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'[OK] Cleaned CSV written to: {args.output}')

    # Optional split: last-N rows as test
    if args.test_last and args.test_last > 0:
        n = len(df)
        test_n = min(args.test_last, n)
        train_df = df.iloc[: n - test_n]
        test_df = df.iloc[n - test_n :]
        base_dir = os.path.dirname(args.output)
        train_out = os.path.join(base_dir, 'training.csv')
        test_out = os.path.join(base_dir, 'testing.csv')
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)
        print('[OK] Train/Test split saved:')
        print(f'     Train: {train_out} ({len(train_df)})')
        print(f'     Test : {test_out} ({len(test_df)})')

    # Report which target/input columns are present
    wanted = set(DEFAULT_CANONICAL.values())
    present = [c for c in df.columns if c in wanted]
    missing = [c for c in wanted if c not in df.columns]
    if present:
        print('[Info] Found canonical columns:', present)
    if missing:
        print('[Warn] Missing canonical columns:', missing)


if __name__ == '__main__':
    main()
