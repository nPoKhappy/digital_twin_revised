from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_ORDER = [
    "air2 → H₂S",
    "T2 → H₂S",
    "air2 → SO₂",
    "T2 → SO₂",
]
PAIR_LABELS = {
    ("B35_H2S", "air2_SP"): "air2 → H₂S",
    ("B35_H2S", "HEATER2_output_T_SP"): "T2 → H₂S",
    ("B35_SO2", "air2_SP"): "air2 → SO₂",
    ("B35_SO2", "HEATER2_output_T_SP"): "T2 → SO₂",
}
PLOT_PAIR_LABELS = {
    "air2 → H₂S": r"air2 $\rightarrow$ H$_2$S",
    "T2 → H₂S": r"T2 $\rightarrow$ H$_2$S",
    "air2 → SO₂": r"air2 $\rightarrow$ SO$_2$",
    "T2 → SO₂": r"T2 $\rightarrow$ SO$_2$",
}
MODEL_DIRS = {
    "No-Gain": "transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42",
    "Gain": "transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42",
    "Gain + SS=0.01": "transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss001_seed42",
    "Difficult-GradNorm": (
        "transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42_"
        "difficult_group_gradnorm_no_ss"
    ),
}
MODEL_STYLES = {
    "No-Gain": {"color": "#333333", "marker": "s", "linestyle": "--"},
    "Gain": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "Gain + SS=0.01": {"color": "#E69F00", "marker": "^", "linestyle": "-."},
    "Difficult-GradNorm": {
        "color": "#D81B60",
        "marker": "D",
        "linestyle": ":",
    },
}
DISTRIBUTIONS = ["in_range", "soft_ood"]
DISTRIBUTION_LABELS = {"in_range": "In-range", "soft_ood": "Soft-OOD"}
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
SIGN_ORDER = ["Negative", "Positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze teacher-Gain landscape and conditional all18 KCI."
    )
    parser.add_argument("--input-root", default="results/gain_eval_all18")
    parser.add_argument(
        "--teacher-scale-file",
        default="results/Tabular_MLP_New/zscore_std.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/visualization/gain_pair_difficulty_seed42",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_pair_label(frame: pd.DataFrame) -> pd.DataFrame:
    keys = list(zip(frame["target"], frame["mv"]))
    frame = frame.copy()
    frame["pair"] = [PAIR_LABELS.get(key) for key in keys]
    if frame["pair"].isna().any():
        unknown = frame.loc[frame["pair"].isna(), ["target", "mv"]].drop_duplicates()
        raise ValueError(f"Unexpected gain pairs:\n{unknown.to_string(index=False)}")
    return frame


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def load_teacher_landscape(
    diagnostic_path: Path, scales: pd.Series, chunk_size: int
) -> pd.DataFrame:
    columns = [
        "distribution",
        "probe_index",
        "target",
        "mv",
        "direction",
        "teacher_gain",
        "valid_teacher_gain",
        "step_index",
    ]
    pieces = []
    for chunk in pd.read_csv(diagnostic_path, usecols=columns, chunksize=chunk_size):
        selected = chunk.loc[
            (chunk["direction"] == "plus") & (chunk["step_index"] == 1)
        ].copy()
        if not selected.empty:
            pieces.append(selected)
    teacher = pd.concat(pieces, ignore_index=True)
    teacher = add_pair_label(teacher)
    teacher["valid_teacher_gain"] = bool_series(teacher["valid_teacher_gain"])
    teacher["normalized_gain"] = [
        gain * float(scales[mv]) / float(scales[target])
        for gain, mv, target in zip(
            teacher["teacher_gain"], teacher["mv"], teacher["target"]
        )
    ]
    duplicates = teacher.duplicated(
        ["distribution", "probe_index", "target", "mv"], keep=False
    )
    if duplicates.any():
        raise ValueError("Teacher landscape contains duplicate probe/pair rows.")
    return teacher


def sign_entropy(positive_count: int, negative_count: int) -> float:
    total = positive_count + negative_count
    if total == 0:
        return float("nan")
    probabilities = [positive_count / total, negative_count / total]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def summarize_teacher(teacher: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for distribution in DISTRIBUTIONS:
        for pair in PAIR_ORDER:
            group = teacher.loc[
                (teacher["distribution"] == distribution) & (teacher["pair"] == pair)
            ]
            valid = group.loc[group["valid_teacher_gain"]]
            positive_count = int((valid["teacher_gain"] > 0).sum())
            negative_count = int((valid["teacher_gain"] < 0).sum())
            valid_count = len(valid)
            rows.append(
                {
                    "distribution": distribution,
                    "pair": pair,
                    "probe_points": len(group),
                    "valid_probe_points": valid_count,
                    "coverage_percent": 100.0 * valid_count / len(group),
                    "near_zero_or_invalid_percent": 100.0
                    * (len(group) - valid_count)
                    / len(group),
                    "positive_percent_of_valid": 100.0
                    * positive_count
                    / valid_count,
                    "negative_percent_of_valid": 100.0
                    * negative_count
                    / valid_count,
                    "sign_entropy_bits": sign_entropy(positive_count, negative_count),
                    "median_abs_raw_gain": valid["teacher_gain"].abs().median(),
                    "q25_abs_raw_gain": valid["teacher_gain"].abs().quantile(0.25),
                    "q75_abs_raw_gain": valid["teacher_gain"].abs().quantile(0.75),
                    "median_abs_normalized_gain": valid["normalized_gain"].abs().median(),
                    "q25_abs_normalized_gain": valid["normalized_gain"].abs().quantile(0.25),
                    "q75_abs_normalized_gain": valid["normalized_gain"].abs().quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def build_quartile_edges(teacher: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for distribution in DISTRIBUTIONS:
        for pair in PAIR_ORDER:
            group = teacher.loc[
                (teacher["distribution"] == distribution)
                & (teacher["pair"] == pair)
                & teacher["valid_teacher_gain"]
            ]
            raw = group["teacher_gain"].abs()
            normalized = group["normalized_gain"].abs()
            raw_edges = raw.quantile([0.25, 0.50, 0.75]).to_numpy()
            normalized_edges = normalized.quantile([0.25, 0.50, 0.75]).to_numpy()
            if np.any(np.diff(normalized_edges) <= 0):
                raise ValueError(
                    f"Non-unique Gain quartile edges for {distribution}, {pair}."
                )
            rows.append(
                {
                    "distribution": distribution,
                    "pair": pair,
                    "q25_abs_raw_gain": raw_edges[0],
                    "q50_abs_raw_gain": raw_edges[1],
                    "q75_abs_raw_gain": raw_edges[2],
                    "q25_abs_normalized_gain": normalized_edges[0],
                    "q50_abs_normalized_gain": normalized_edges[1],
                    "q75_abs_normalized_gain": normalized_edges[2],
                }
            )
    return pd.DataFrame(rows)


def assign_quartiles(frame: pd.DataFrame, edge_lookup: dict) -> pd.Series:
    labels = np.empty(len(frame), dtype=object)
    for (distribution, pair), indices in frame.groupby(
        ["distribution", "pair"], sort=False
    ).groups.items():
        edges = edge_lookup[(distribution, pair)]
        values = frame.loc[indices, "abs_normalized_gain"].to_numpy()
        labels[frame.index.get_indexer(indices)] = np.array(QUARTILES)[
            np.searchsorted(edges, values, side="right")
        ]
    return pd.Series(labels, index=frame.index)


def load_probe_accuracy(
    diagnostic_path: Path,
    scales: pd.Series,
    quartile_edges: pd.DataFrame,
    chunk_size: int,
) -> pd.DataFrame:
    columns = [
        "distribution",
        "probe_index",
        "target",
        "mv",
        "teacher_gain",
        "teacher_sign",
        "valid_teacher_gain",
        "agreement",
    ]
    edge_lookup = {
        (row.distribution, row.pair): np.array(
            [
                row.q25_abs_normalized_gain,
                row.q50_abs_normalized_gain,
                row.q75_abs_normalized_gain,
            ]
        )
        for row in quartile_edges.itertuples(index=False)
    }
    pieces = []
    for chunk in pd.read_csv(diagnostic_path, usecols=columns, chunksize=chunk_size):
        chunk = chunk.loc[bool_series(chunk["valid_teacher_gain"])].copy()
        if chunk.empty:
            continue
        chunk = add_pair_label(chunk)
        chunk["abs_normalized_gain"] = [
            abs(gain) * float(scales[mv]) / float(scales[target])
            for gain, mv, target in zip(
                chunk["teacher_gain"], chunk["mv"], chunk["target"]
            )
        ]
        chunk["magnitude_bin"] = assign_quartiles(chunk, edge_lookup)
        chunk["teacher_sign_label"] = np.where(
            chunk["teacher_sign"] > 0, "Positive", "Negative"
        )
        chunk["correct"] = (chunk["agreement"] == "correct").astype(np.int16)
        grouped = (
            chunk.groupby(
                [
                    "distribution",
                    "pair",
                    "magnitude_bin",
                    "teacher_sign_label",
                    "probe_index",
                ],
                observed=True,
                sort=False,
            )["correct"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "correct", "count": "total"})
        )
        pieces.append(grouped)
    probe = pd.concat(pieces, ignore_index=True)
    probe = (
        probe.groupby(
            [
                "distribution",
                "pair",
                "magnitude_bin",
                "teacher_sign_label",
                "probe_index",
            ],
            observed=True,
            sort=False,
        )[["correct", "total"]]
        .sum()
        .reset_index()
    )
    unexpected_totals = probe.loc[probe["total"] != 36, "total"].unique()
    if len(unexpected_totals):
        raise ValueError(
            "Expected 36 all18 comparisons per valid probe/pair; found "
            f"{sorted(unexpected_totals.tolist())}."
        )
    probe["probe_kci_percent"] = 100.0 * probe["correct"] / probe["total"]
    return probe


def bootstrap_ci(
    values: np.ndarray, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = []
    batch_size = 200
    for start in range(0, resamples, batch_size):
        count = min(batch_size, resamples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
    sampled_means = np.concatenate(means)
    return tuple(np.quantile(sampled_means, [0.025, 0.975]))


def summarize_conditional_kci(
    probe: pd.DataFrame,
    model: str,
    group_column: str,
    group_order: list[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for distribution in DISTRIBUTIONS:
        for pair in PAIR_ORDER:
            for group_value in group_order:
                group = probe.loc[
                    (probe["distribution"] == distribution)
                    & (probe["pair"] == pair)
                    & (probe[group_column] == group_value)
                ]
                values = group["probe_kci_percent"].to_numpy()
                low, high = bootstrap_ci(values, resamples, rng)
                total = int(group["total"].sum())
                correct = int(group["correct"].sum())
                rows.append(
                    {
                        "model": model,
                        "distribution": distribution,
                        "pair": pair,
                        group_column: group_value,
                        "probe_points": len(group),
                        "comparisons": total,
                        "correct": correct,
                        "kci_percent": 100.0 * correct / total if total else np.nan,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def validate_against_official(
    input_root: Path, model: str, model_dir: str, probe: pd.DataFrame
) -> None:
    official = pd.read_csv(input_root / model_dir / "gain_evaluation_detail.csv")
    official = add_pair_label(official)
    for row in official.itertuples(index=False):
        group = probe.loc[
            (probe["distribution"] == row.distribution) & (probe["pair"] == row.pair)
        ]
        calculated = 100.0 * group["correct"].sum() / group["total"].sum()
        if not np.isclose(calculated, row.kci_percent, rtol=0, atol=1e-10):
            raise AssertionError(
                f"{model}, {row.distribution}, {row.pair}: calculated KCI "
                f"{calculated} differs from official {row.kci_percent}."
            )


def plot_teacher_landscape(
    teacher: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), squeeze=False)
    magnitude_color = "#009E73"
    sign_colors = {
        "Negative": "#0072B2",
        "Near-zero / invalid": "#999999",
        "Positive": "#D55E00",
    }
    for row_index, distribution in enumerate(DISTRIBUTIONS):
        magnitude_axis = axes[row_index, 0]
        sign_axis = axes[row_index, 1]
        magnitude_data = []
        for pair in PAIR_ORDER:
            values = teacher.loc[
                (teacher["distribution"] == distribution)
                & (teacher["pair"] == pair)
                & teacher["valid_teacher_gain"],
                "normalized_gain",
            ].abs()
            magnitude_data.append(values.to_numpy())
        boxes = magnitude_axis.boxplot(
            magnitude_data,
            tick_labels=[PLOT_PAIR_LABELS[pair] for pair in PAIR_ORDER],
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(magnitude_color)
            box.set_alpha(0.75)
        magnitude_axis.set_yscale("log")
        magnitude_axis.set_ylabel(r"Absolute normalized teacher Gain, $|\tilde{g}|$")
        magnitude_axis.set_title(
            f"{chr(65 + row_index * 2)}  {DISTRIBUTION_LABELS[distribution]} magnitude",
            loc="left",
            fontweight="bold",
        )
        magnitude_axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.7)
        magnitude_axis.tick_params(axis="x", rotation=18)

        current = summary.loc[summary["distribution"] == distribution].set_index("pair")
        negative = current.loc[PAIR_ORDER, "negative_percent_of_valid"].to_numpy()
        near_zero = current.loc[PAIR_ORDER, "near_zero_or_invalid_percent"].to_numpy()
        coverage = current.loc[PAIR_ORDER, "coverage_percent"].to_numpy() / 100.0
        negative_all = negative * coverage
        positive_all = (
            current.loc[PAIR_ORDER, "positive_percent_of_valid"].to_numpy() * coverage
        )
        x_values = np.arange(len(PAIR_ORDER))
        sign_axis.bar(
            x_values,
            negative_all,
            color=sign_colors["Negative"],
            label="Negative",
        )
        sign_axis.bar(
            x_values,
            near_zero,
            bottom=negative_all,
            color=sign_colors["Near-zero / invalid"],
            label="Near-zero / invalid",
        )
        sign_axis.bar(
            x_values,
            positive_all,
            bottom=negative_all + near_zero,
            color=sign_colors["Positive"],
            label="Positive",
        )
        sign_axis.set_xticks(
            x_values,
            [PLOT_PAIR_LABELS[pair] for pair in PAIR_ORDER],
            rotation=18,
        )
        sign_axis.set_ylim(0, 100)
        sign_axis.set_ylabel("Teacher Gain composition (%)")
        sign_axis.set_title(
            f"{chr(66 + row_index * 2)}  {DISTRIBUTION_LABELS[distribution]} sign",
            loc="left",
            fontweight="bold",
        )
        sign_axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.7)
    sign_handles, sign_labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        sign_handles,
        sign_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.75, 0.91),
    )
    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Teacher Gain magnitude and sign across operating points",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        "Normalized Gain uses Tabular-MLP IQR scales. Near-zero / invalid follows "
        "the existing physical-Gain validity threshold.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"teacher_gain_landscape.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_conditional_kci(
    data: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.6), sharey=True, squeeze=False)
    x_values = np.arange(len(group_order))
    for row_index, distribution in enumerate(DISTRIBUTIONS):
        for column_index, pair in enumerate(PAIR_ORDER):
            axis = axes[row_index, column_index]
            for model, style in MODEL_STYLES.items():
                rows = data.loc[
                    (data["model"] == model)
                    & (data["distribution"] == distribution)
                    & (data["pair"] == pair)
                ].set_index(group_column).loc[group_order]
                values = rows["kci_percent"].to_numpy()
                lower = values - rows["ci95_low"].to_numpy()
                upper = rows["ci95_high"].to_numpy() - values
                axis.errorbar(
                    x_values,
                    values,
                    yerr=np.vstack([lower, upper]),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    markersize=5,
                    capsize=2,
                    label=model,
                )
            axis.set_title(PLOT_PAIR_LABELS[pair], fontweight="bold")
            axis.set_xticks(x_values, group_order)
            axis.set_ylim(0, 102)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column_index == 0:
                axis.set_ylabel(
                    f"{DISTRIBUTION_LABELS[distribution]}\nAll18 KCI (%)"
                )
            if row_index == 1:
                axis.set_xlabel(
                    "Teacher Gain magnitude quartile"
                    if group_column == "magnitude_bin"
                    else "Teacher Gain sign"
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    figure.text(
        0.5,
        0.01,
        "Points show pooled all18 KCI. Error bars are 95% bootstrap intervals "
        "resampled by operating point.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.89))
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{filename}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_root = repo_root / args.input_root
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = pd.read_csv(repo_root / args.teacher_scale_file, index_col=0).squeeze(
        "columns"
    )
    required_scales = {target for target, _ in PAIR_LABELS} | {
        mv for _, mv in PAIR_LABELS
    }
    missing_scales = sorted(required_scales - set(scales.index))
    if missing_scales:
        raise ValueError(f"Teacher scale file is missing: {missing_scales}")

    teacher_path = (
        input_root / MODEL_DIRS["No-Gain"] / "gain_probe_diagnostics.csv"
    )
    teacher = load_teacher_landscape(teacher_path, scales, args.chunk_size)
    teacher_summary = summarize_teacher(teacher)
    quartile_edges = build_quartile_edges(teacher)
    teacher_summary.to_csv(output_dir / "teacher_gain_summary.csv", index=False)
    quartile_edges.to_csv(output_dir / "gain_magnitude_quartile_edges.csv", index=False)

    magnitude_frames = []
    sign_frames = []
    for model_index, (model, model_dir) in enumerate(MODEL_DIRS.items()):
        print(f"[Info] Processing {model}", flush=True)
        diagnostic_path = input_root / model_dir / "gain_probe_diagnostics.csv"
        probe = load_probe_accuracy(
            diagnostic_path, scales, quartile_edges, args.chunk_size
        )
        validate_against_official(input_root, model, model_dir, probe)
        magnitude_frames.append(
            summarize_conditional_kci(
                probe,
                model,
                "magnitude_bin",
                QUARTILES,
                args.bootstrap_resamples,
                args.bootstrap_seed + model_index,
            )
        )
        sign_frames.append(
            summarize_conditional_kci(
                probe,
                model,
                "teacher_sign_label",
                SIGN_ORDER,
                args.bootstrap_resamples,
                args.bootstrap_seed + 100 + model_index,
            )
        )

    magnitude_kci = pd.concat(magnitude_frames, ignore_index=True)
    sign_kci = pd.concat(sign_frames, ignore_index=True)
    magnitude_kci.to_csv(output_dir / "conditional_kci_by_gain_magnitude.csv", index=False)
    sign_kci.to_csv(output_dir / "conditional_kci_by_teacher_sign.csv", index=False)

    configure_style()
    plot_teacher_landscape(teacher, teacher_summary, output_dir)
    plot_conditional_kci(
        magnitude_kci,
        "magnitude_bin",
        QUARTILES,
        "All18 KCI conditioned on teacher Gain magnitude",
        "conditional_kci_by_gain_magnitude",
        output_dir,
    )
    plot_conditional_kci(
        sign_kci,
        "teacher_sign_label",
        SIGN_ORDER,
        "All18 KCI conditioned on teacher Gain sign",
        "conditional_kci_by_teacher_sign",
        output_dir,
    )
    print(f"[Done] Saved analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
