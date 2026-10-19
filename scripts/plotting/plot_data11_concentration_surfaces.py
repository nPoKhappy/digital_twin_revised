"""Regenerate the six data-11 concentration plots without figure titles."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from show_lhs_air2_tr2_surface import (
    H2S_COL,
    SO2_COL,
    SUM_COL,
    load_completed_points,
    make_surface,
)


INPUT_FILE = "data/Claus_steady_state/lhs_generated_dynamic_ss_data_11.xlsx"
OUTPUT_DIR = Path("results/lhs_steady_state_analysis/data11_concentration_surfaces")

TARGETS = {
    "h2s": (H2S_COL, "H2S"),
    "so2": (SO2_COL, "SO2"),
    "total": (SUM_COL, "H2S + SO2"),
}


def plot_contour(df, z_col: str, label: str, output_path: Path) -> None:
    x, y, _, grid_x, grid_y, grid_z = make_surface(df, z_col, grid_size=180)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    filled = ax.contourf(grid_x, grid_y, grid_z, levels=24, cmap="RdYlGn_r")
    lines = ax.contour(
        grid_x,
        grid_y,
        grid_z,
        levels=10,
        colors="white",
        linewidths=0.55,
        alpha=0.75,
    )
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.2e", colors="white")
    ax.set_xlabel("air2 SP")
    ax.set_ylabel("T2 SP (°C)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    fig.colorbar(filled, ax=ax, pad=0.02, label=f"{label} concentration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_surface(df, z_col: str, label: str, output_path: Path) -> None:
    _, _, _, grid_x, grid_y, grid_z = make_surface(df, z_col, grid_size=180)

    fig = plt.figure(figsize=(9.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        cmap="RdYlGn_r",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    ax.set_xlabel("air2 SP", labelpad=10)
    ax.set_ylabel("T2 SP (°C)", labelpad=10)
    ax.set_zlabel(f"{label} concentration", labelpad=10)
    ax.view_init(elev=26, azim=-135)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.10, label=f"{label} concentration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_completed_points([INPUT_FILE])
    if df.empty:
        raise ValueError("No usable Run Completed rows found in data 11.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, (z_col, label) in TARGETS.items():
        contour_path = OUTPUT_DIR / f"lhs_data11_{key}_2d_contour.png"
        surface_path = OUTPUT_DIR / f"lhs_data11_{key}_3d_surface.png"
        plot_contour(df, z_col, label, contour_path)
        plot_surface(df, z_col, label, surface_path)
        print(f"Saved: {contour_path}")
        print(f"Saved: {surface_path}")


if __name__ == "__main__":
    main()
