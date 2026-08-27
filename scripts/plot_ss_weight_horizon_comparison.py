from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CASES = ["R5", "R5-1", "R5-2", "R5-6"]
EXTERNAL_CASES = ["R5-11", "R5-12"]
TARGETS = ["B35_H2S", "B35_SO2"]
TARGET_LABELS = {"B35_H2S": r"H$_2$S", "B35_SO2": r"SO$_2$"}
HORIZONS = [1, 18]
SS_WEIGHTS = [0.001, 0.01, 0.05]

RESULT_DIRS = {
    "No-Gain": "transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42",
    "Gain-only": "transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42",
    "Gain+SS=0.001": "transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss0001_seed42",
    "Gain+SS=0.01": "transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss001_seed42",
    "Gain+SS=0.05": "transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss005_seed42",
}

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]

EXTERNAL_MODELS = [
    "No-Gain",
    "Gain+SS=0.001",
    "Gain+SS=0.01",
    "Gain+SS=0.05",
]
EXTERNAL_MODEL_STYLES = {
    "No-Gain": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "Gain+SS=0.001": {"color": "#D55E00", "marker": "o", "linestyle": "-"},
    "Gain+SS=0.01": {"color": "#009E73", "marker": "^", "linestyle": "-"},
    "Gain+SS=0.05": {"color": "#CC79A7", "marker": "D", "linestyle": "-"},
}

GAIN_SS_MODELS = [
    "Gain-only",
    "Gain+SS=0.001",
    "Gain+SS=0.01",
    "Gain+SS=0.05",
]
GAIN_SS_MODEL_STYLES = {
    "No-Gain": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "Gain-only": {"color": "#000000", "marker": "P", "linestyle": "-."},
    "Gain+SS=0.001": {"color": "#D55E00", "marker": "o", "linestyle": "-"},
    "Gain+SS=0.01": {"color": "#009E73", "marker": "^", "linestyle": "-"},
    "Gain+SS=0.05": {"color": "#CC79A7", "marker": "D", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot seed-42 Horizon 1/18 RMSE comparisons across SS weights."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ss_weight_horizon_comparison_seed42"),
    )
    return parser.parse_args()


def load_metrics(repo_root: Path) -> pd.DataFrame:
    frames = []
    expected_rows = len(CASES) * len(TARGETS) * len(HORIZONS)

    for model, result_dir in RESULT_DIRS.items():
        csv_path = (
            repo_root
            / "results"
            / result_dir
            / "grouped_horizon_parity"
            / "grouped_parity_metrics.csv"
        )
        frame = pd.read_csv(csv_path)
        frame["horizon"] = pd.to_numeric(frame["horizon"])
        frame = frame[
            (frame["split"] == "test")
            & frame["case"].isin(CASES)
            & frame["variable"].isin(TARGETS)
            & frame["horizon"].isin(HORIZONS)
        ].copy()
        if len(frame) != expected_rows:
            raise ValueError(
                f"{csv_path} yielded {len(frame)} rows; expected {expected_rows}."
            )
        frame["model"] = model
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_rolling_metrics(repo_root: Path) -> pd.DataFrame:
    frames = []
    expected_rows = len(CASES) * len(TARGETS)

    for model, result_dir in RESULT_DIRS.items():
        csv_path = (
            repo_root
            / "results"
            / result_dir
            / "grouped_time_series"
            / "grouped_time_series_metrics.csv"
        )
        frame = pd.read_csv(csv_path)
        frame = frame[
            (frame["split"] == "test")
            & frame["case"].isin(CASES)
            & frame["variable"].isin(TARGETS)
        ].copy()
        if len(frame) != expected_rows:
            raise ValueError(
                f"{csv_path} yielded {len(frame)} rows; expected {expected_rows}."
            )
        frame["model"] = model
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_external_raw_metrics(repo_root: Path) -> pd.DataFrame:
    records = []
    for model in EXTERNAL_MODELS:
        result_root = repo_root / "results" / RESULT_DIRS[model]
        horizon = pd.read_csv(
            result_root / "grouped_horizon_parity" / "grouped_parity_metrics.csv"
        )
        rolling = pd.read_csv(
            result_root / "grouped_time_series" / "grouped_time_series_metrics.csv"
        )
        horizon["horizon"] = pd.to_numeric(horizon["horizon"])

        for case in EXTERNAL_CASES:
            for target in TARGETS:
                for evaluation, horizon_step in (("H1", 1), ("H18", 18)):
                    row = horizon[
                        (horizon["split"] == "all")
                        & (horizon["case"] == case)
                        & (horizon["variable"] == target)
                        & (horizon["horizon"] == horizon_step)
                    ]
                    if len(row) != 1:
                        raise ValueError(
                            f"Expected one external row for {model}, {case}, "
                            f"{target}, {evaluation}; found {len(row)}."
                        )
                    records.append(
                        {
                            "model": model,
                            "case": case,
                            "target": target,
                            "evaluation": evaluation,
                            "RMSE": float(row.iloc[0]["RMSE"]),
                        }
                    )

                row = rolling[
                    (rolling["split"] == "all")
                    & (rolling["case"] == case)
                    & (rolling["variable"] == target)
                ]
                if len(row) != 1:
                    raise ValueError(
                        f"Expected one external rolling row for {model}, {case}, "
                        f"{target}; found {len(row)}."
                    )
                records.append(
                    {
                        "model": model,
                        "case": case,
                        "target": target,
                        "evaluation": "TS",
                        "RMSE": float(row.iloc[0]["RMSE"]),
                    }
                )

    return pd.DataFrame.from_records(records)


def rmse_value(
    data: pd.DataFrame, model: str, case: str, target: str, horizon: int
) -> float:
    values = data.loc[
        (data["model"] == model)
        & (data["case"] == case)
        & (data["variable"] == target)
        & (data["horizon"] == horizon),
        "RMSE",
    ]
    if len(values) != 1:
        raise ValueError(
            f"Expected one RMSE for {model}, {case}, {target}, H{horizon}; "
            f"found {len(values)}."
        )
    return float(values.iloc[0])


def rolling_rmse_value(
    data: pd.DataFrame, model: str, case: str, target: str
) -> float:
    values = data.loc[
        (data["model"] == model)
        & (data["case"] == case)
        & (data["variable"] == target),
        "RMSE",
    ]
    if len(values) != 1:
        raise ValueError(
            f"Expected one rolling RMSE for {model}, {case}, {target}; "
            f"found {len(values)}."
        )
    return float(values.iloc[0])


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
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_relative_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    row_keys = [(case, target) for case in CASES for target in TARGETS]
    column_models = [f"Gain+SS={weight:g}" for weight in SS_WEIGHTS]
    matrices: dict[int, np.ndarray] = {}

    for horizon in HORIZONS:
        matrix = np.zeros((len(row_keys), len(column_models)))
        for row_index, (case, target) in enumerate(row_keys):
            baseline = rmse_value(data, "No-Gain", case, target, horizon)
            for column_index, model in enumerate(column_models):
                variant = rmse_value(data, model, case, target, horizon)
                matrix[row_index, column_index] = 100.0 * (variant / baseline - 1.0)
        matrices[horizon] = matrix

    max_abs = max(float(np.abs(matrix).max()) for matrix in matrices.values())
    color_limit = max(10.0, np.ceil(max_abs / 10.0) * 10.0)
    row_labels = [f"{case} | {TARGET_LABELS[target]}" for case, target in row_keys]
    column_labels = [f"SS={weight:g}" for weight in SS_WEIGHTS]

    figure = plt.figure(figsize=(10.2, 5.7))
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.32)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])

    for index, (axis, horizon) in enumerate(zip(axes, HORIZONS)):
        annotations = np.vectorize(lambda value: f"{value:+.1f}%")(matrices[horizon])
        sns.heatmap(
            matrices[horizon],
            ax=axis,
            cmap="RdBu_r",
            center=0,
            vmin=-color_limit,
            vmax=color_limit,
            annot=annotations,
            fmt="",
            linewidths=0.8,
            linecolor="white",
            xticklabels=column_labels,
            yticklabels=row_labels if index == 0 else False,
            cbar=index == 1,
            cbar_ax=colorbar_axis if index == 1 else None,
            cbar_kws={"label": "RMSE change relative to No-Gain (%)"},
            annot_kws={"fontsize": 8.5},
        )
        axis.set_title(f"{'AB'[index]}  Horizon {horizon}", loc="left", fontweight="bold")
        axis.set_xlabel("Gain fixed at 0.05")
        axis.set_ylabel("Case and target" if index == 0 else "")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)

    figure.suptitle(
        "Case-dependent effect of Gain+SS configurations on prediction RMSE",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    figure.subplots_adjust(top=0.90, bottom=0.12, left=0.13, right=0.92)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"horizon_rmse_relative_heatmap_vs_no_gain.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_ss_response(data: pd.DataFrame, output_dir: Path) -> None:
    model_sequence = [
        "Gain-only",
        "Gain+SS=0.001",
        "Gain+SS=0.01",
        "Gain+SS=0.05",
    ]
    x_positions = np.arange(len(model_sequence))
    x_labels = ["0", "0.001", "0.01", "0.05"]

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)

    for row_index, target in enumerate(TARGETS):
        target_values = []
        for horizon in HORIZONS:
            for case in CASES:
                target_values.extend(
                    rmse_value(data, model, case, target, horizon)
                    for model in model_sequence
                )
        lower = min(target_values)
        upper = max(target_values)
        padding = max(1e-8, 0.10 * (upper - lower))

        for column_index, horizon in enumerate(HORIZONS):
            axis = axes[row_index, column_index]
            for case_index, case in enumerate(CASES):
                values = [
                    rmse_value(data, model, case, target, horizon)
                    for model in model_sequence
                ]
                axis.plot(
                    x_positions,
                    values,
                    color=OKABE_ITO[case_index],
                    marker=MARKERS[case_index],
                    linewidth=1.8,
                    markersize=5.5,
                    label=case,
                )
            axis.set_ylim(lower - padding, upper + padding)
            axis.set_xticks(x_positions, x_labels)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axis.set_title(
                f"{'ABCD'[row_index * 2 + column_index]}  "
                f"{TARGET_LABELS[target]}, Horizon {horizon}",
                loc="left",
                fontweight="bold",
            )
            axis.set_ylabel("RMSE" if column_index == 0 else "")
            axis.set_xlabel("SS-loss weight (Gain fixed at 0.05)" if row_index == 1 else "")
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Test case",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "SS-loss weight response: original RMSE with Gain loss held constant",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.2, w_pad=1.6)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"ss_weight_raw_rmse_response.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_rolling_relative_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    row_keys = [(case, target) for case in CASES for target in TARGETS]
    column_models = [f"Gain+SS={weight:g}" for weight in SS_WEIGHTS]
    matrix = np.zeros((len(row_keys), len(column_models)))

    for row_index, (case, target) in enumerate(row_keys):
        baseline = rolling_rmse_value(data, "No-Gain", case, target)
        for column_index, model in enumerate(column_models):
            variant = rolling_rmse_value(data, model, case, target)
            matrix[row_index, column_index] = 100.0 * (variant / baseline - 1.0)

    max_abs = float(np.abs(matrix).max())
    color_limit = max(10.0, np.ceil(max_abs / 10.0) * 10.0)
    annotations = np.vectorize(lambda value: f"{value:+.1f}%")(matrix)
    row_labels = [
        f"{case} | {TARGET_LABELS[target]}" for case, target in row_keys
    ]
    column_labels = [f"SS={weight:g}" for weight in SS_WEIGHTS]

    figure, axis = plt.subplots(figsize=(6.8, 5.8))
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="RdBu_r",
        center=0,
        vmin=-color_limit,
        vmax=color_limit,
        annot=annotations,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        xticklabels=column_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "RMSE change relative to No-Gain (%)"},
        annot_kws={"fontsize": 9},
    )
    axis.set_title(
        "Rolling time-series RMSE change relative to No-Gain",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    axis.set_xlabel("Gain fixed at 0.05")
    axis.set_ylabel("Case and target")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()

    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"rolling_rmse_relative_heatmap_vs_no_gain.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_rolling_ss_response(data: pd.DataFrame, output_dir: Path) -> None:
    model_sequence = [
        "Gain-only",
        "Gain+SS=0.001",
        "Gain+SS=0.01",
        "Gain+SS=0.05",
    ]
    x_positions = np.arange(len(model_sequence))
    x_labels = ["0", "0.001", "0.01", "0.05"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True)

    for target_index, target in enumerate(TARGETS):
        axis = axes[target_index]
        values_by_case = {}
        for case in CASES:
            values_by_case[case] = [
                rolling_rmse_value(data, model, case, target)
                for model in model_sequence
            ]

        all_values = [value for values in values_by_case.values() for value in values]
        lower = min(all_values)
        upper = max(all_values)
        padding = max(1e-8, 0.10 * (upper - lower))

        for case_index, case in enumerate(CASES):
            axis.plot(
                x_positions,
                values_by_case[case],
                color=OKABE_ITO[case_index],
                marker=MARKERS[case_index],
                linewidth=1.8,
                markersize=5.5,
                label=case,
            )
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xticks(x_positions, x_labels)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.set_title(
            f"{'AB'[target_index]}  {TARGET_LABELS[target]}",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("SS-loss weight (Gain fixed at 0.05)")
        axis.set_ylabel("RMSE" if target_index == 0 else "")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Test case",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Rolling time-series SS-loss weight response: original RMSE",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.8)
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"rolling_ss_weight_raw_rmse_response.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_internal_rmse_normalized_to_no_gain(
    horizon_data: pd.DataFrame,
    rolling_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    evaluations = ["H1", "H18", "TS"]
    x_values = np.arange(len(evaluations))
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4), squeeze=False)
    records = []
    models = ["No-Gain", *GAIN_SS_MODELS]

    for row_index, target in enumerate(TARGETS):
        for column_index, case in enumerate(CASES):
            axis = axes[row_index, column_index]
            baseline_values = [
                rmse_value(horizon_data, "No-Gain", case, target, 1),
                rmse_value(horizon_data, "No-Gain", case, target, 18),
                rolling_rmse_value(rolling_data, "No-Gain", case, target),
            ]
            ratios_by_model = {}
            raw_by_model = {}
            for model in models:
                raw_values = [
                    rmse_value(horizon_data, model, case, target, 1),
                    rmse_value(horizon_data, model, case, target, 18),
                    rolling_rmse_value(rolling_data, model, case, target),
                ]
                raw_by_model[model] = raw_values
                ratios_by_model[model] = [
                    value / baseline
                    for value, baseline in zip(raw_values, baseline_values)
                ]

            all_ratios = [
                value for values in ratios_by_model.values() for value in values
            ]
            lower = min(all_ratios + [1.0])
            upper = max(all_ratios + [1.0])
            padding = max(0.02, 0.10 * (upper - lower))
            y_limits = (lower - padding, upper + padding)
            axis.axhspan(y_limits[0], 1.0, color="#DCEEF8", alpha=0.42, zorder=0)
            axis.axhspan(1.0, y_limits[1], color="#F8E1DE", alpha=0.35, zorder=0)

            for model in models:
                style = GAIN_SS_MODEL_STYLES[model]
                axis.plot(
                    x_values,
                    ratios_by_model[model],
                    label=model,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=5.5,
                )
                records.extend(
                    {
                        "case": case,
                        "target": target,
                        "evaluation": evaluation,
                        "model": model,
                        "RMSE": raw_value,
                        "No-Gain RMSE": baseline,
                        "RMSE / No-Gain RMSE": ratio,
                    }
                    for evaluation, raw_value, baseline, ratio in zip(
                        evaluations,
                        raw_by_model[model],
                        baseline_values,
                        ratios_by_model[model],
                    )
                )

            axis.set_title(
                f"{case} | {TARGET_LABELS[target]}",
                fontweight="bold",
                pad=8,
            )
            axis.set_xticks(x_values, evaluations)
            axis.set_ylim(*y_limits)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75)
            axis.grid(axis="x", visible=False)
            axis.margins(y=0.12)
            sns.despine(ax=axis)
            if column_index == 0:
                axis.set_ylabel("RMSE / No-Gain RMSE")
            if row_index == 1:
                axis.set_xlabel("Evaluation")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "RMSE normalized to No-Gain across H1, H18, and time series",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        "No-Gain = 1. Values below 1 indicate lower RMSE than No-Gain; "
        "Gain weight fixed at 0.05 and no case averaging.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.85, bottom=0.12, hspace=0.38, wspace=0.34)

    filename = "internal_test_gain_ss_rmse_normalized_to_no_gain"
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    pd.DataFrame.from_records(records).to_csv(
        output_dir / f"{filename}.csv",
        index=False,
    )


def plot_external_rmse_normalized_to_no_gain(
    data: pd.DataFrame, output_dir: Path
) -> None:
    evaluations = ["H1", "H18", "TS"]
    x_values = np.arange(len(evaluations))
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), squeeze=False)
    records = []

    for row_index, target in enumerate(TARGETS):
        for column_index, case in enumerate(EXTERNAL_CASES):
            axis = axes[row_index, column_index]
            raw_by_model = {}
            for model in EXTERNAL_MODELS:
                values = []
                for evaluation in evaluations:
                    rows = data[
                        (data["model"] == model)
                        & (data["case"] == case)
                        & (data["target"] == target)
                        & (data["evaluation"] == evaluation)
                    ]
                    if len(rows) != 1:
                        raise ValueError(
                            f"Expected one raw RMSE for {model}, {case}, "
                            f"{target}, {evaluation}; found {len(rows)}."
                        )
                    values.append(float(rows.iloc[0]["RMSE"]))

                raw_by_model[model] = values

            baseline_values = raw_by_model["No-Gain"]
            ratios_by_model = {
                model: [
                    value / baseline
                    for value, baseline in zip(values, baseline_values)
                ]
                for model, values in raw_by_model.items()
            }
            all_ratios = [
                value for values in ratios_by_model.values() for value in values
            ]
            lower = min(all_ratios + [1.0])
            upper = max(all_ratios + [1.0])
            padding = max(0.02, 0.10 * (upper - lower))
            y_limits = (lower - padding, upper + padding)
            axis.axhspan(y_limits[0], 1.0, color="#DCEEF8", alpha=0.42, zorder=0)
            axis.axhspan(1.0, y_limits[1], color="#F8E1DE", alpha=0.35, zorder=0)

            for model in EXTERNAL_MODELS:
                style = EXTERNAL_MODEL_STYLES[model]
                axis.plot(
                    x_values,
                    ratios_by_model[model],
                    label=model,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=6,
                )
                records.extend(
                    {
                        "case": case,
                        "target": target,
                        "evaluation": evaluation,
                        "model": model,
                        "RMSE": raw_value,
                        "No-Gain RMSE": baseline,
                        "RMSE / No-Gain RMSE": ratio,
                    }
                    for evaluation, raw_value, baseline, ratio in zip(
                        evaluations,
                        raw_by_model[model],
                        baseline_values,
                        ratios_by_model[model],
                    )
                )

            axis.set_title(
                f"{case} | {TARGET_LABELS[target]}",
                fontweight="bold",
                pad=8,
            )
            axis.set_xticks(x_values, evaluations)
            axis.set_ylim(*y_limits)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75)
            axis.grid(axis="x", visible=False)
            axis.margins(y=0.12)
            sns.despine(ax=axis)
            if column_index == 0:
                axis.set_ylabel("RMSE / No-Gain RMSE")
            if row_index == 1:
                axis.set_xlabel("Evaluation")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "RMSE normalized to No-Gain (external all-data)",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        "No-Gain = 1. Values below 1 indicate lower RMSE than No-Gain; "
        "no case averaging.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.84, bottom=0.11, hspace=0.38, wspace=0.28)

    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir
            / f"external_all_data_gain_ss_rmse_normalized_to_no_gain.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    pd.DataFrame.from_records(records).sort_values(
        ["case", "target", "evaluation", "model"]
    ).to_csv(
        output_dir / "external_all_data_gain_ss_rmse_normalized_to_no_gain.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    metrics = load_metrics(repo_root)
    rolling_metrics = load_rolling_metrics(repo_root)
    external_metrics = load_external_raw_metrics(repo_root)
    plot_relative_heatmap(metrics, output_dir)
    plot_rolling_relative_heatmap(rolling_metrics, output_dir)
    plot_internal_rmse_normalized_to_no_gain(metrics, rolling_metrics, output_dir)
    plot_external_rmse_normalized_to_no_gain(external_metrics, output_dir)

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
