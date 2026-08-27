from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INTERNAL_CASES = ["R5", "R5-1", "R5-2", "R5-6"]
EXTERNAL_CASES = ["R5-11", "R5-12"]
CASES = INTERNAL_CASES + EXTERNAL_CASES
TARGETS = ["B35_H2S", "B35_SO2"]
TARGET_LABELS = {"B35_H2S": r"H$_2$S", "B35_SO2": r"SO$_2$"}
EVALUATIONS = ["H1", "H18", "TS"]

RESULT_DIRS = {
    "No-Gain": "transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42",
    "Gain": "transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42",
}

MODEL_STYLES = {
    "No-Gain": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "Gain": {"color": "#D55E00", "marker": "o", "linestyle": "-"},
}


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def selected_split(case: str) -> str:
    return "test" if case in INTERNAL_CASES else "all"


def load_model_metrics(repo_root: Path, model: str, result_dir: str) -> pd.DataFrame:
    root = repo_root / "results" / result_dir
    horizon = pd.read_csv(
        root / "grouped_horizon_parity" / "grouped_parity_metrics.csv"
    )
    rolling = pd.read_csv(
        root / "grouped_time_series" / "grouped_time_series_metrics.csv"
    )
    horizon["horizon"] = pd.to_numeric(horizon["horizon"])

    records = []
    for case in CASES:
        split = selected_split(case)
        for target in TARGETS:
            for evaluation, horizon_step in (("H1", 1), ("H18", 18)):
                row = horizon[
                    (horizon["case"] == case)
                    & (horizon["variable"] == target)
                    & (horizon["split"] == split)
                    & (horizon["horizon"] == horizon_step)
                ]
                if len(row) != 1:
                    raise ValueError(
                        f"Expected one {model} row for {case}, {target}, "
                        f"{evaluation}, split={split}; found {len(row)}."
                    )
                records.append(
                    {
                        "model": model,
                        "case": case,
                        "target": target,
                        "evaluation": evaluation,
                        "RMSE": float(row.iloc[0]["RMSE"]),
                        "R2": float(row.iloc[0]["R2"]),
                    }
                )

            row = rolling[
                (rolling["case"] == case)
                & (rolling["variable"] == target)
                & (rolling["split"] == split)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Expected one {model} rolling row for {case}, {target}, "
                    f"split={split}; found {len(row)}."
                )
            records.append(
                {
                    "model": model,
                    "case": case,
                    "target": target,
                    "evaluation": "TS",
                    "RMSE": float(row.iloc[0]["RMSE"]),
                    "R2": float(row.iloc[0]["R2"]),
                }
            )

    return pd.DataFrame.from_records(records)


def metric_value(
    data: pd.DataFrame,
    model: str,
    case: str,
    target: str,
    evaluation: str,
    metric: str,
) -> float:
    values = data.loc[
        (data["model"] == model)
        & (data["case"] == case)
        & (data["target"] == target)
        & (data["evaluation"] == evaluation),
        metric,
    ]
    if len(values) != 1:
        raise ValueError(
            f"Expected one {metric} for {model}, {case}, {target}, "
            f"{evaluation}; found {len(values)}."
        )
    return float(values.iloc[0])


def build_delta_matrices(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row_keys = [(case, target) for case in CASES for target in TARGETS]
    delta_r2 = np.zeros((len(row_keys), len(EVALUATIONS)))
    delta_rmse = np.zeros_like(delta_r2)

    for row_index, (case, target) in enumerate(row_keys):
        for column_index, evaluation in enumerate(EVALUATIONS):
            no_gain_r2 = metric_value(
                data, "No-Gain", case, target, evaluation, "R2"
            )
            gain_r2 = metric_value(data, "Gain", case, target, evaluation, "R2")
            no_gain_rmse = metric_value(
                data, "No-Gain", case, target, evaluation, "RMSE"
            )
            gain_rmse = metric_value(
                data, "Gain", case, target, evaluation, "RMSE"
            )
            delta_r2[row_index, column_index] = gain_r2 - no_gain_r2
            delta_rmse[row_index, column_index] = 100.0 * (
                gain_rmse / no_gain_rmse - 1.0
            )

    return delta_r2, delta_rmse


def set_annotation_colors(axis: plt.Axes, matrix: np.ndarray, limit: float) -> None:
    for text, value in zip(axis.texts, matrix.flat):
        text.set_color("white" if abs(float(value)) >= 0.55 * limit else "#222222")


def plot_gain_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    delta_r2, delta_rmse = build_delta_matrices(data)
    r2_limit = max(0.01, np.ceil(np.abs(delta_r2).max() / 0.01) * 0.01)
    rmse_limit = max(0.5, np.ceil(np.abs(delta_rmse).max() / 0.5) * 0.5)

    row_labels = []
    for case in CASES:
        suffix = "†" if case in EXTERNAL_CASES else ""
        for target in TARGETS:
            row_labels.append(f"{case}{suffix} | {TARGET_LABELS[target]}")

    r2_annotations = np.vectorize(lambda value: f"{value:+.4f}")(delta_r2)
    rmse_annotations = np.vectorize(lambda value: f"{value:+.2f}%")(delta_rmse)

    figure = plt.figure(figsize=(11.2, 7.4))
    grid = figure.add_gridspec(
        1, 4, width_ratios=[1, 0.045, 1, 0.045], wspace=0.38
    )
    r2_axis = figure.add_subplot(grid[0, 0])
    r2_colorbar = figure.add_subplot(grid[0, 1])
    rmse_axis = figure.add_subplot(grid[0, 2])
    rmse_colorbar = figure.add_subplot(grid[0, 3])

    sns.heatmap(
        delta_r2,
        ax=r2_axis,
        cmap="RdBu",
        center=0,
        vmin=-r2_limit,
        vmax=r2_limit,
        annot=r2_annotations,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        xticklabels=EVALUATIONS,
        yticklabels=row_labels,
        cbar_ax=r2_colorbar,
        cbar_kws={"label": r"$ΔR^2$ (Gain − No-Gain)"},
        annot_kws={"fontsize": 8.2},
    )
    set_annotation_colors(r2_axis, delta_r2, r2_limit)

    sns.heatmap(
        delta_rmse,
        ax=rmse_axis,
        cmap="RdBu_r",
        center=0,
        vmin=-rmse_limit,
        vmax=rmse_limit,
        annot=rmse_annotations,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        xticklabels=EVALUATIONS,
        yticklabels=False,
        cbar_ax=rmse_colorbar,
        cbar_kws={"label": r"$Δ$RMSE (%)"},
        annot_kws={"fontsize": 8.2},
    )
    set_annotation_colors(rmse_axis, delta_rmse, rmse_limit)

    for axis, title in (
        (r2_axis, r"A  $ΔR^2$ (positive is better)"),
        (rmse_axis, r"B  $Δ$RMSE (negative is better)"),
    ):
        axis.set_title(title, loc="left", fontweight="bold", pad=12)
        axis.set_xlabel("Evaluation")
        axis.tick_params(axis="x", rotation=0)
        axis.axhline(len(INTERNAL_CASES) * len(TARGETS), color="black", linewidth=1.8)
    r2_axis.set_ylabel("Case and target")
    r2_axis.tick_params(axis="y", rotation=0)
    rmse_axis.set_ylabel("")

    figure.suptitle(
        "Effect of Gain loss (weight = 0.05) on prediction accuracy",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.025,
        "Blue indicates improvement.  † External all-data evaluation; "
        "the heatmaps show changes, not absolute prediction quality.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.90, bottom=0.10, left=0.12, right=0.92)

    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"gain005_vs_no_gain_r2_rmse_heatmap.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_raw_rmse_grid(
    data: pd.DataFrame,
    cases: list[str],
    group_label: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Plot the original RMSE values for every case and target."""
    figure, axes = plt.subplots(
        len(TARGETS),
        len(cases),
        figsize=(3.25 * len(cases), 6.4),
        squeeze=False,
    )
    x_values = np.arange(len(EVALUATIONS))

    for row_index, target in enumerate(TARGETS):
        for column_index, case in enumerate(cases):
            axis = axes[row_index, column_index]
            for model in ("No-Gain", "Gain"):
                values = [
                    metric_value(data, model, case, target, evaluation, "RMSE")
                    for evaluation in EVALUATIONS
                ]
                style = MODEL_STYLES[model]
                axis.plot(
                    x_values,
                    values,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markersize=6,
                    linewidth=1.8,
                    label=model,
                )
                vertical_offset = 8 if model == "No-Gain" else -13
                for x_value, value in zip(x_values, values):
                    axis.annotate(
                        f"{value:.4g}",
                        (x_value, value),
                        xytext=(0, vertical_offset),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if model == "No-Gain" else "top",
                        color=style["color"],
                        fontsize=6.8,
                    )

            axis.set_title(
                f"{case} | {TARGET_LABELS[target]}",
                fontweight="bold",
                pad=8,
            )
            axis.set_xticks(x_values, EVALUATIONS)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75)
            axis.grid(axis="x", visible=False)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.margins(y=0.20)
            sns.despine(ax=axis)

            if column_index == 0:
                axis.set_ylabel(f"{TARGET_LABELS[target]} RMSE")
            if row_index == 1:
                axis.set_xlabel("Evaluation")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        f"Gain vs No-Gain: original RMSE ({group_label})",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        "Original RMSE values from H1, H18, and rolling time-series evaluation; "
        "no normalization and no case averaging.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.86, bottom=0.12, hspace=0.38, wspace=0.34)

    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_raw_rmse_lines(data: pd.DataFrame, output_dir: Path) -> None:
    plot_raw_rmse_grid(
        data,
        INTERNAL_CASES,
        "internal test",
        "gain005_vs_no_gain_raw_rmse_internal_test",
        output_dir,
    )
    plot_raw_rmse_grid(
        data,
        EXTERNAL_CASES,
        "external all-data",
        "gain005_vs_no_gain_raw_rmse_external_all_data",
        output_dir,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "gain_vs_no_gain_visualization_seed42"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    frames = [
        load_model_metrics(repo_root, model, result_dir)
        for model, result_dir in RESULT_DIRS.items()
    ]
    data = pd.concat(frames, ignore_index=True)
    plot_gain_heatmap(data, output_dir)
    plot_raw_rmse_lines(data, output_dir)
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
