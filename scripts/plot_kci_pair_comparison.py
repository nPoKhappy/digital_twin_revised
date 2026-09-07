from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULT_FILES = {
    "No-Gain": (
        "gain_in_soft_ood_distribution_evaluation/gain_evaluation_detail.csv",
        "no_gain",
    ),
    "Gain": (
        "gain_in_soft_ood_distribution_evaluation/gain_evaluation_detail.csv",
        "gain",
    ),
    "Gain + SS=0.01": (
        "gain_in_soft_ood_ss_weight_evaluation/ss001/gain_evaluation_detail.csv",
        "gain",
    ),
    "Difficult-GradNorm": (
        "gain_in_soft_ood_difficult_group_gradnorm_no_ss_evaluation/gain_evaluation_detail.csv",
        "gain",
    ),
}
MODELS = list(RESULT_FILES)
DISTRIBUTIONS = ["in_range", "soft_ood"]
DISTRIBUTION_LABELS = {"in_range": "In-range", "soft_ood": "Soft-OOD"}
PAIR_KEYS = [
    ("B35_H2S", "air2_SP"),
    ("B35_H2S", "HEATER2_output_T_SP"),
    ("B35_SO2", "air2_SP"),
    ("B35_SO2", "HEATER2_output_T_SP"),
]
PAIR_LABELS = [
    r"air2 $\rightarrow$ H$_2$S",
    r"T2 $\rightarrow$ H$_2$S",
    r"air2 $\rightarrow$ SO$_2$",
    r"T2 $\rightarrow$ SO$_2$",
]
MODEL_STYLES = {
    "No-Gain": {"color": "#333333", "marker": "s", "linestyle": "--"},
    "Gain": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "Gain + SS=0.01": {"color": "#E69F00", "marker": "^", "linestyle": "-."},
    "Difficult-GradNorm": {"color": "#D81B60", "marker": "D", "linestyle": ":"},
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


def load_model_kci(repo_root: Path, model: str) -> pd.DataFrame:
    relative_path, training_type = RESULT_FILES[model]
    path = repo_root / "results" / relative_path
    data = pd.read_csv(path)
    expected = len(DISTRIBUTIONS) * len(PAIR_KEYS)
    data = data[
        data["distribution"].isin(DISTRIBUTIONS)
        & (data["training_type"] == training_type)
        & data[["target", "mv"]].apply(tuple, axis=1).isin(PAIR_KEYS)
    ].copy()
    if len(data) != expected:
        raise ValueError(f"{path} yielded {len(data)} rows; expected {expected}.")
    return data


def kci_value(
    data: pd.DataFrame,
    distribution: str,
    pair: tuple[str, str],
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
    return float(rows.iloc[0]["kci_percent"])


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


def build_records(comparisons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []

    for distribution in DISTRIBUTIONS:
        for pair, pair_label in zip(PAIR_KEYS, PAIR_LABELS):
            for model in MODELS:
                records.append(
                    {
                        "Distribution": distribution,
                        "Gain pair": pair_label,
                        "Model": model,
                        "Raw KCI (%)": kci_value(
                            comparisons[model], distribution, pair
                        ),
                    }
                )

    return pd.DataFrame.from_records(records)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        repo_root / "results" / "kci_pair_visualization_seed42_in_soft_ood"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    comparisons = {
        model: load_model_kci(repo_root, model) for model in RESULT_FILES
    }
    records = build_records(comparisons)
    plot_kci_panels(
        records,
        MODELS,
        "Four-model Raw KCI comparison across gain pairs",
        "raw_kci_four_model_comparison",
        output_dir,
    )
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
