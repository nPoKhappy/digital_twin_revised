from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULT_DIRS = {
    "Gain-only": "gain_in_hard_distribution_evaluation_seed42",
    "Gain+SS=0.001": "gain_in_hard_distribution_evaluation_seed42_ss0001",
    "Gain+SS=0.01": "gain_in_hard_distribution_evaluation_seed42_ss001",
    "Gain+SS=0.05": "gain_in_hard_distribution_evaluation_seed42_ss005",
}
GAIN_SS_MODELS = ["No-Gain", "Gain+SS=0.001", "Gain+SS=0.01", "Gain+SS=0.05"]
GAIN_MODELS = ["No-Gain", "Gain-only"]
DISTRIBUTIONS = ["in_range", "out_range"]
DISTRIBUTION_LABELS = {"in_range": "In-range", "out_range": "Hard-OOD"}
PAIR_KEYS = [
    ("B35_H2S", "air2_SP"),
    ("B35_H2S", "HEATER2_output_T_SP"),
    ("B35_SO2", "air2_SP"),
    ("B35_SO2", "HEATER2_output_T_SP"),
]
PAIR_LABELS = [
    r"H$_2$S × air2",
    r"H$_2$S × T2",
    r"SO$_2$ × air2",
    r"SO$_2$ × T2",
]
MODEL_STYLES = {
    "No-Gain": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "Gain-only": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "Gain+SS=0.001": {"color": "#D55E00", "marker": "o", "linestyle": "-"},
    "Gain+SS=0.01": {"color": "#009E73", "marker": "^", "linestyle": "-"},
    "Gain+SS=0.05": {"color": "#CC79A7", "marker": "D", "linestyle": "-"},
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
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_comparison(repo_root: Path, model: str) -> pd.DataFrame:
    path = (
        repo_root
        / "results"
        / RESULT_DIRS[model]
        / "gain_vs_no_gain_comparison.csv"
    )
    data = pd.read_csv(path)
    expected = len(DISTRIBUTIONS) * len(PAIR_KEYS)
    data = data[
        data["distribution"].isin(DISTRIBUTIONS)
        & data[["target", "mv"]].apply(tuple, axis=1).isin(PAIR_KEYS)
    ].copy()
    if len(data) != expected:
        raise ValueError(f"{path} yielded {len(data)} rows; expected {expected}.")
    return data


def kci_value(
    data: pd.DataFrame,
    distribution: str,
    pair: tuple[str, str],
    column: str,
) -> float:
    rows = data[
        (data["distribution"] == distribution)
        & (data["target"] == pair[0])
        & (data["mv"] == pair[1])
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one KCI row for {distribution}, {pair}; found {len(rows)}."
        )
    return float(rows.iloc[0][column])


def plot_kci_panels(
    records: pd.DataFrame,
    models: list[str],
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
    x_values = np.arange(len(PAIR_KEYS))
    minimum = float(records["Raw KCI (%)"].min())
    lower_limit = max(0.0, np.floor((minimum - 4.0) / 5.0) * 5.0)

    for panel_index, distribution in enumerate(DISTRIBUTIONS):
        axis = axes[panel_index]
        for model in models:
            values = [
                float(
                    records.loc[
                        (records["Distribution"] == distribution)
                        & (records["Gain pair"] == pair_label)
                        & (records["Model"] == model),
                        "Raw KCI (%)",
                    ].iloc[0]
                )
                for pair_label in PAIR_LABELS
            ]
            style = MODEL_STYLES[model]
            axis.plot(
                x_values,
                values,
                label=model,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                markersize=6,
            )

        axis.set_title(
            f"{'AB'[panel_index]}  {DISTRIBUTION_LABELS[distribution]}",
            loc="left",
            fontweight="bold",
        )
        axis.set_xticks(x_values, PAIR_LABELS, rotation=18, ha="right")
        axis.set_ylim(lower_limit, 102.0)
        axis.set_xlabel("Gain pair")
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75)
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)

    axes[0].set_ylabel("Raw KCI (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(models),
        frameon=False,
    )
    figure.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    figure.text(
        0.5,
        0.015,
        "Higher Raw KCI means stronger sign agreement with ANN-reference gradients; "
        "it is not direct process-physics validation.",
        ha="center",
        fontsize=8.8,
    )
    figure.subplots_adjust(top=0.78, bottom=0.25, wspace=0.16)

    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    records.to_csv(output_dir / f"{filename}.csv", index=False)


def build_records(comparisons: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gain_ss_records = []
    gain_records = []
    baseline = comparisons["Gain-only"]

    for distribution in DISTRIBUTIONS:
        for pair, pair_label in zip(PAIR_KEYS, PAIR_LABELS):
            no_gain = kci_value(
                baseline, distribution, pair, "kci_percent_no_gain"
            )
            gain_only = kci_value(
                baseline, distribution, pair, "kci_percent_gain"
            )
            gain_records.extend(
                [
                    {
                        "Distribution": distribution,
                        "Gain pair": pair_label,
                        "Model": "No-Gain",
                        "Raw KCI (%)": no_gain,
                    },
                    {
                        "Distribution": distribution,
                        "Gain pair": pair_label,
                        "Model": "Gain-only",
                        "Raw KCI (%)": gain_only,
                    },
                ]
            )
            gain_ss_records.append(
                {
                    "Distribution": distribution,
                    "Gain pair": pair_label,
                    "Model": "No-Gain",
                    "Raw KCI (%)": no_gain,
                }
            )
            for model in GAIN_SS_MODELS[1:]:
                gain_ss_records.append(
                    {
                        "Distribution": distribution,
                        "Gain pair": pair_label,
                        "Model": model,
                        "Raw KCI (%)": kci_value(
                            comparisons[model],
                            distribution,
                            pair,
                            "kci_percent_gain",
                        ),
                    }
                )

    return (
        pd.DataFrame.from_records(gain_ss_records),
        pd.DataFrame.from_records(gain_records),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "kci_pair_visualization_seed42"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    comparisons = {
        model: load_comparison(repo_root, model) for model in RESULT_DIRS
    }
    gain_ss_records, gain_records = build_records(comparisons)
    plot_kci_panels(
        gain_ss_records,
        GAIN_SS_MODELS,
        "Raw KCI across gain pairs: No-Gain vs Gain + SS loss",
        "raw_kci_no_gain_vs_gain_ss",
        output_dir,
    )
    plot_kci_panels(
        gain_records,
        GAIN_MODELS,
        "Raw KCI across gain pairs: No-Gain vs Gain-only",
        "raw_kci_no_gain_vs_gain_only",
        output_dir,
    )
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
