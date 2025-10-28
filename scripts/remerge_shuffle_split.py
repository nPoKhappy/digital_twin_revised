import argparse
import os
import sys
import pandas as pd
from typing import List, Optional


def _load_any(path: str, encoding: Optional[str], sheet_names: Optional[List[str]], all_sheets: bool) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # Excel: read one or many sheets
        if sheet_names:
            frames = []
            for name in sheet_names:
                try:
                    frames.append(pd.read_excel(path, sheet_name=name))
                except Exception as e:
                    print(f"[Error] Failed to read sheet '{name}': {e}")
                    sys.exit(1)
            return pd.concat(frames, ignore_index=True)
        if all_sheets or not sheet_names:
            x = pd.read_excel(path, sheet_name=None)  # dict of sheets
            frames = list(x.values())
            if not frames:
                print("[Error] No sheets found in Excel file.")
                sys.exit(1)
            return pd.concat(frames, ignore_index=True)
    else:
        # CSV and others: fallback to read_csv
        kw = {}
        if encoding:
            kw["encoding"] = encoding
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            print(f"[Error] Failed to read CSV: {e}")
            sys.exit(1)

    print("[Error] Unsupported file format or read failed.")
    sys.exit(1)


def _determine_n_train(total_len: int, arg_train_size: Optional[str]) -> int:
    if total_len <= 1:
        return max(0, total_len - 1)
    if arg_train_size is None:
        return max(1, min(total_len - 1, int(round(total_len * 0.8))))
    try:
        val = float(arg_train_size)
    except ValueError:
        print(f"[Error] Invalid --train-size: {arg_train_size}. Use float (0-1) or int.")
        sys.exit(1)
    if 0 < val < 1:
        n_train = int(round(total_len * val))
    else:
        n_train = int(round(val))
    return max(1, min(total_len - 1, n_train))


def main():
    ap = argparse.ArgumentParser(description="Load one CSV/Excel (optionally multi-sheet), shuffle, and split into training/testing.")
    ap.add_argument("--input", required=True, help="Path to CSV/XLSX input file")
    ap.add_argument("--encoding", default=None, help="CSV encoding if needed (e.g., cp950)")
    ap.add_argument("--sheet-names", default=None, help="Comma-separated sheet names to read (Excel only). If omitted and file is Excel, all sheets are merged.")
    ap.add_argument("--train-size", default=None, help="Float (0-1) fraction or integer number of rows for train. Default 0.8")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    ap.add_argument("--drop-duplicates", action="store_true", help="Drop exact duplicate rows before splitting")
    ap.add_argument("--output-dir", default=None, help="Directory for outputs. Default = same as input")
    ap.add_argument("--train-out", default=None, help="Explicit path for training.csv (overrides output-dir)")
    ap.add_argument("--test-out", default=None, help="Explicit path for testing.csv (overrides output-dir)")
    args = ap.parse_args()

    sheet_list = None
    if args.sheet_names:
        sheet_list = [s.strip() for s in args.sheet_names.split(',') if s.strip()]

    df = _load_any(args.input, args.encoding, sheet_list, all_sheets=not bool(sheet_list))
    print(f"[Info] Loaded data: {df.shape}")

    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if after != before:
            print(f"[Info] Dropped {before - after} duplicate rows; now {after}")

    # Shuffle
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Split
    n_total = len(df)
    n_train = _determine_n_train(n_total, args.train_size)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    # Outputs
    base_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(base_dir, exist_ok=True)
    train_out = args.train_out or os.path.join(base_dir, "training.csv")
    test_out = args.test_out or os.path.join(base_dir, "testing.csv")

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("[OK] Split completed:")
    print(f"     Train -> {train_out} ({len(train_df)})")
    print(f"     Test  -> {test_out} ({len(test_df)})")


if __name__ == "__main__":
    main()
