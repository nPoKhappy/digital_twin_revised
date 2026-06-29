import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


DEFAULT_EXCEL = "data/Claus_steady_state/lhs_generated_dynamic_ss_data.xlsx"

AIR2_COL = "B33.SPo.SPo"
TR2_COL = "B20.SPo.SPo"
H2S_COL = 'S33.Zn.H2S.("H2S")'
SO2_COL = 'S33.Zn.SO2.("SO2")'
SUM_COL = "H2S_plus_SO2"


def resolve_excel_paths(excel_pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(excel_pattern))
    if not paths and Path(excel_pattern).exists():
        paths = [Path(excel_pattern)]
    if not paths:
        raise FileNotFoundError(f"No Excel files found: {excel_pattern}")
    return paths


def lhs_excel_name(file_index: int) -> str:
    suffix = "" if file_index == 1 else f"_{file_index}"
    return f"data/Claus_steady_state/lhs_generated_dynamic_ss_data{suffix}.xlsx"


def load_completed_points(excel_patterns: list[str]) -> pd.DataFrame:
    paths = []
    seen_paths = set()
    for excel_pattern in excel_patterns:
        for path in resolve_excel_paths(excel_pattern):
            resolved = path.resolve()
            if resolved not in seen_paths:
                paths.append(path)
                seen_paths.add(resolved)
    dfs = []

    for excel_path in paths:
        df = pd.read_excel(excel_path, sheet_name=0, header=2)
        df = df.iloc[1:].dropna(how="all").copy()
        df = df[df["Status"].astype(str).str.strip().eq("Run Completed")].copy()

        for col in [AIR2_COL, TR2_COL, H2S_COL, SO2_COL]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[AIR2_COL, TR2_COL, H2S_COL, SO2_COL])
        df["source_file"] = excel_path.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df[SUM_COL] = df[H2S_COL] + df[SO2_COL]
    print(f"Loaded {len(paths)} file(s), Run Completed usable rows={len(df)}:")
    for path in paths:
        print(f"  - {path}")
    return df


def make_surface(df: pd.DataFrame, z_col: str, grid_size: int):
    x = df[AIR2_COL].to_numpy(dtype=float)
    y = df[TR2_COL].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
    return x, y, z, grid_x, grid_y, grid_z


def add_surface_subplot(fig, subplot_id, df, z_col, label, grid_size):
    x, y, z, grid_x, grid_y, grid_z = make_surface(df, z_col, grid_size)

    ax = fig.add_subplot(subplot_id, projection="3d")
    surface = ax.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        cmap="RdYlGn_r",
        linewidth=0,
        antialiased=True,
        alpha=0.88,
    )
    ax.set_xlabel("air2_SP / B33.SPo.SPo", labelpad=10)
    ax.set_ylabel("tr2 = HEATER2_output_T_SP / B20.SPo.SPo", labelpad=10)
    ax.set_zlabel(f"{label} / {z_col}", labelpad=10)
    # Place the low-air2, low-tr2 corner at the lower-left front of the view.
    ax.view_init(elev=26, azim=-135)
    ax.grid(True, alpha=0.45)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.10, label=label)
    return ax


def add_contour_subplot(fig, subplot_id, df, z_col, label, grid_size):
    x, y, _, grid_x, grid_y, grid_z = make_surface(df, z_col, grid_size)

    ax = fig.add_subplot(subplot_id)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=24, cmap="RdYlGn_r")
    ax.contour(
        grid_x,
        grid_y,
        grid_z,
        levels=12,
        colors="black",
        linewidths=0.45,
        alpha=0.48,
    )
    ax.set_xlabel("air2_SP / B33.SPo.SPo")
    ax.set_ylabel("tr2 = HEATER2_output_T_SP / B20.SPo.SPo")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.grid(False)
    fig.colorbar(contour, ax=ax, pad=0.02, label=label)
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Show interactive 3D surfaces and 2D contours for Run Completed LHS data."
    )
    parser.add_argument(
        "--excel",
        action="append",
        default=None,
        help="Excel path or glob pattern. Use multiple --excel flags to combine selected files.",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        help="Load one LHS file by index. Example: --file-index 5 loads lhs_generated_dynamic_ss_data_5.xlsx.",
    )
    parser.add_argument(
        "--include-file5",
        action="store_true",
        help="Load lhs_generated_dynamic_ss_data.xlsx and lhs_generated_dynamic_ss_data_5.xlsx only. This is the default when no source option is given.",
    )
    parser.add_argument(
        "--target",
        choices=["all", "sum", "h2s", "so2"],
        default="all",
        help="Which target to show. Default creates separate H2S, SO2, and H2S + SO2 figures.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=120,
        help="Interpolation grid size. Larger is smoother but slower.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/steady_state_surface_contour_plot",
        help="Directory used to save generated figures.",
    )
    args = parser.parse_args()

    if args.include_file5:
        excel_patterns = [lhs_excel_name(1), lhs_excel_name(5)]
    elif args.file_index:
        excel_patterns = [lhs_excel_name(args.file_index)]
    else:
        excel_patterns = args.excel or [lhs_excel_name(1), lhs_excel_name(5)]

    df = load_completed_points(excel_patterns)
    if df.empty:
        raise ValueError("No Run Completed rows found with valid air2/tr2/H2S/SO2 values.")

    target_map = {
        "h2s": (H2S_COL, "H2S"),
        "so2": (SO2_COL, "SO2"),
        "sum": (SUM_COL, "Tail gas total sulfur = H2S + SO2"),
    }
    target_keys = ["h2s", "so2", "sum"] if args.target == "all" else [args.target]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for target_key in target_keys:
        z_col, label = target_map[target_key]
        fig = plt.figure(figsize=(9.5, 7.2))
        add_surface_subplot(fig, 111, df, z_col, label, args.grid_size)
        contour_fig = plt.figure(figsize=(9.5, 7.2))
        add_contour_subplot(contour_fig, 111, df, z_col, label, args.grid_size)
        figures.extend([
            (fig, output_dir / f"{target_key}_surface_3d.png"),
            (contour_fig, output_dir / f"{target_key}_contour_2d.png"),
        ])

    for fig, output_path in figures:
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
