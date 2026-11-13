from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


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
    "No-Gain": "no_gain",
    "Gain=250": "gain",
    "SS-only=2.5": "ss_only",
    "Gain+SS=2.5/2.5": "gain_ss",
}
MODEL_STYLES = {
    "No-Gain": {"color": "#333333", "marker": "s", "linestyle": "--"},
    "Gain=250": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "SS-only=2.5": {"color": "#D55E00", "marker": "^", "linestyle": "-."},
    "Gain+SS=2.5/2.5": {
        "color": "#009E73",
        "marker": "s",
        "linestyle": ":",
    },
}
DISTRIBUTIONS = ["in_range", "soft_ood"]
DISTRIBUTION_LABELS = {"in_range": "In-range", "soft_ood": "Soft-OOD"}
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
SIGN_ORDER = ["Negative", "Positive"]
FIGURE_NAMES = [
    "teacher_gain_landscape",
    "conditional_kci_by_gain_magnitude",
    "conditional_kci_by_teacher_sign",
]
ANALYSIS_NAMES = {
    "teacher_gain_summary.csv", "gain_magnitude_quartile_edges.csv",
    "teacher_gain_landscape.csv", "gain_pair_probe_accuracy.csv",
    "conditional_kci_by_gain_magnitude.csv", "conditional_kci_by_teacher_sign.csv",
    "gain_pair_difficulty_protocol.json", "gain_pair_difficulty_validation.json",
    "gain_pair_difficulty_README.md", "rebuild_gain_pair_difficulty.py",
    "gain_pair_difficulty_contact_sheet.png", "gain_pair_difficulty_contact_sheet_grayscale.png",
    *[name+suffix for name in FIGURE_NAMES for suffix in [".png", ".pdf", "_grayscale.png"]],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze teacher-Gain landscape and conditional all18 KCI."
    )
    parser.add_argument(
        "--input-root",
        default="results/evaluation/gain_eval_all18/test/best_family_models_seed42_20261106",
    )
    parser.add_argument(
        "--teacher-scale-file",
        default="results/steady_state/Tabular_MLP_New/zscore_std.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/visualization/best_gain_ss_gainss_test_seed42",
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


def model_files(directory: Path, filename: str) -> list[Path]:
    """Support both the original combined CSV and distribution-specific exports."""
    combined = directory / filename
    if combined.is_file():
        return [combined]
    paths = [directory / distribution / filename for distribution in DISTRIBUTIONS]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def diagnostic_chunks(paths: Path | list[Path], columns: list[str], chunk_size: int):
    for path in ([paths] if isinstance(paths, Path) else paths):
        yield from pd.read_csv(path, usecols=columns, chunksize=chunk_size)


def load_teacher_landscape(
    diagnostic_path: Path | list[Path], scales: pd.Series, chunk_size: int
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
    for chunk in diagnostic_chunks(diagnostic_path, columns, chunk_size):
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
    diagnostic_path: Path | list[Path],
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
    for chunk in diagnostic_chunks(diagnostic_path, columns, chunk_size):
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
    official = pd.concat(
        [pd.read_csv(path) for path in model_files(input_root / model_dir, "gain_evaluation_detail.csv")],
        ignore_index=True,
    )
    official = add_pair_label(official)
    if len(official) != 8 or official.duplicated(["distribution", "pair"]).any():
        raise ValueError(f"{model}: expected exactly eight official distribution/pair rows.")
    for row in official.itertuples(index=False):
        group = probe.loc[
            (probe["distribution"] == row.distribution) & (probe["pair"] == row.pair)
        ]
        calculated = 100.0 * group["correct"].sum() / group["total"].sum()
        if int(group["correct"].sum()) != row.tp + row.tn or int(group["total"].sum()) != row.valid_comparisons:
            raise AssertionError(f"{model}, {row.distribution}, {row.pair}: counts differ from the official result.")
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
        "Normalized Gain uses saved Tabular-MLP input/output scales. Near-zero / invalid follows "
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


def verify_partition_summaries(magnitude, sign, probes) -> None:
    keys = ["model", "distribution", "pair"]
    expected = pd.concat(probes).groupby(keys)[["correct", "total"]].sum().sort_index()
    for frame in [magnitude, sign]:
        partitioned = frame.groupby(keys)[["correct", "comparisons"]].sum().sort_index()
        np.testing.assert_array_equal(partitioned.correct, expected.correct)
        np.testing.assert_array_equal(partitioned.comparisons, expected.total)
        valid = frame[frame.probe_points > 0]
        if not np.isfinite(valid[["kci_percent", "ci95_low", "ci95_high"]]).all().all():
            raise AssertionError("Nonempty conditional groups must have finite KCI and confidence intervals.")
        if not ((valid.ci95_low >= 0) & (valid.ci95_high <= 100) & (valid.ci95_low <= valid.ci95_high)).all():
            raise AssertionError("Invalid bootstrap interval.")


def finish_analysis(output_dir, protocol, teacher_summary, magnitude, sign) -> None:
    contact = Image.new("RGB", (1500, 1650), "white")
    for index, name in enumerate(FIGURE_NAMES):
        with Image.open(output_dir / f"{name}.png") as source:
            thumbnail = source.convert("RGB")
            thumbnail.convert("L").save(output_dir / f"{name}_grayscale.png")
        thumbnail.thumbnail((1500, 550))
        contact.paste(thumbnail, ((1500-thumbnail.width)//2, index*550+(550-thumbnail.height)//2))
    contact.save(output_dir / "gain_pair_difficulty_contact_sheet.png")
    contact.convert("L").save(output_dir / "gain_pair_difficulty_contact_sheet_grayscale.png")
    lines = ["# 四模型 test：增益配對難度分析", "",
             "模型：No-Gain、Gain=250、SS-only=2.5、Gain+SS=(2.5,2.5)，seed42。使用剛才四模型 test KCI 的原始逐點明細，不重新訓練或推論。", "",
             "## 圖與數據", ""]
    for name in FIGURE_NAMES:
        lines.append(f"- [{name} PNG]({name}.png)／[PDF]({name}.pdf)／[灰階]({name}_grayscale.png)")
    lines += ["", "- [Teacher 統計](teacher_gain_summary.csv)／[逐點 teacher 增益](teacher_gain_landscape.csv)",
              "- [Q1～Q4 邊界](gain_magnitude_quartile_edges.csv)／[依增益大小的 KCI](conditional_kci_by_gain_magnitude.csv)",
              "- [依正負方向的 KCI](conditional_kci_by_teacher_sign.csv)／[每個操作點的36次比較](gain_pair_probe_accuracy.csv)",
              "- [來源與計算定義](gain_pair_difficulty_protocol.json)／[核對紀錄](gain_pair_difficulty_validation.json)", "",
              "## 定義", "",
              "Teacher 增益取同一個 probe 的 plus 方向、第1個 post-step 時刻作為唯一參考，避免重複計數；模型的KCI仍使用±方向各18個時刻，共36次比較。", "",
              "正規化增益為 teacher_gain × teacher輸入scale / teacher輸出scale。使用保存的Tabular-MLP尺度；這份程式原先圖註的IQR已修正為保存尺度，數值公式維持原樣。", "",
              "Q1～Q4在每個區域、每個增益配對的有效teacher探測點內各自計算25/50/75百分位；四個模型共用No-Gain teacher邊界。Q1為較小增益，Q4為較大增益；原本無效／近零增益不納入條件式KCI。", "",
              f"誤差棒為{protocol['bootstrap_resamples']:,}次bootstrap的95%區間。重抽樣單位是操作點，其36次方向／時刻比較保留為一整組；不是把36次比較當成獨立樣本，也不是跨訓練seed的不確定性。", "",
              "沒有有效探測點的正／負增益組別保留NaN，不畫成0%。所有四分位與正負組別相加後，correct與valid均重現原本test KCI計數。", "",
              "穩態test核心4385點、近外插2692點；KCI是ANN-reference方向一致性。Teacher大小與正負分布反映參考模型與本批探測點，不等同實際製程的物理增益真值。", "",
              "## 重跑與重畫", "", "完整分析：`python -m scripts.analysis.analyze_gain_pair_difficulty`。",
              "只用保存CSV重畫：`python results/visualization/best_gain_ss_gainss_test_seed42/rebuild_gain_pair_difficulty.py`。", ""]
    (output_dir / "gain_pair_difficulty_README.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "rebuild_gain_pair_difficulty.py").write_text('''from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.analysis.analyze_gain_pair_difficulty import configure_style, plot_teacher_landscape, plot_conditional_kci, finish_analysis, QUARTILES, SIGN_ORDER
if __name__=='__main__':
    import json
    p=Path(__file__).resolve().parent
    teacher=pd.read_csv(p/'teacher_gain_landscape.csv')
    summary=pd.read_csv(p/'teacher_gain_summary.csv')
    magnitude=pd.read_csv(p/'conditional_kci_by_gain_magnitude.csv')
    sign=pd.read_csv(p/'conditional_kci_by_teacher_sign.csv')
    configure_style()
    plot_teacher_landscape(teacher,summary,p)
    plot_conditional_kci(magnitude,'magnitude_bin',QUARTILES,'All18 KCI conditioned on teacher Gain magnitude','conditional_kci_by_gain_magnitude',p)
    plot_conditional_kci(sign,'teacher_sign_label',SIGN_ORDER,'All18 KCI conditioned on teacher Gain sign','conditional_kci_by_teacher_sign',p)
    finish_analysis(p,json.loads((p/'gain_pair_difficulty_protocol.json').read_text(encoding='utf-8')),summary,magnitude,sign)
''', encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    input_root = repo_root / args.input_root
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.chunk_size < 1 or args.bootstrap_resamples < 1:
        raise ValueError("Chunk size and bootstrap resamples must be positive.")
    digest = lambda path: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    retained = {
        str(path.relative_to(repo_root)): digest(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in ANALYSIS_NAMES
    }
    inputs = [Path(__file__).resolve(), repo_root / args.teacher_scale_file]
    for name in ["selected_models.csv", "run_protocol.json"]:
        if (output_dir / name).is_file():
            inputs.append(output_dir / name)
    for model_dir in MODEL_DIRS.values():
        inputs.extend(model_files(input_root / model_dir, "gain_probe_diagnostics.csv"))
        inputs.extend(model_files(input_root / model_dir, "gain_evaluation_detail.csv"))
        # These cached arrays prove that the bins/signs use identical teacher states.
        for distribution in DISTRIBUTIONS:
            array_path = input_root / model_dir / distribution / "gain_arrays.npz"
            if array_path.exists():
                inputs.append(array_path)
                with np.load(input_root / MODEL_DIRS["No-Gain"] / distribution / "gain_arrays.npz") as reference, np.load(array_path) as current:
                    np.testing.assert_array_equal(reference["teacher_gain"], current["teacher_gain"])
                    np.testing.assert_array_equal(reference["physical_ANN_inputs"], current["physical_ANN_inputs"])
            validation = array_path.with_name("evaluation_validation.json")
            if validation.exists():
                inputs.append(validation)
                proof = json.loads(validation.read_text(encoding="utf-8"))
                if not proof["passed"]:
                    raise AssertionError(f"Unverified KCI input: {validation}")
                for name, value in proof["output_sha256"].items():
                    if digest(validation.parent / name) != value:
                        raise AssertionError(f"KCI input changed: {validation.parent / name}")
    protected = {str(path.relative_to(repo_root)): digest(path) for path in inputs}
    protocol = dict(
        input_root=str(input_root.relative_to(repo_root)), models=MODEL_DIRS,
        teacher_normalization="teacher_gain * saved_teacher_scale[mv] / saved_teacher_scale[target]",
        quartiles="Within each distribution and pair; valid No-Gain teacher probes only; shared across all models",
        bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.bootstrap_seed,
        bootstrap_unit="Operating point; its two directions x 18 time steps stay together",
        data_role="Same steady-state test probes as the best-model comparison; not validation",
        protected_sha256=protected, retained_test_outputs_sha256=retained,
    )
    (output_dir / "gain_pair_difficulty_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("[Info] Input hashes verified; identical teacher arrays and probe inputs across four models.", flush=True)
    scales = pd.read_csv(repo_root / args.teacher_scale_file, index_col=0).squeeze(
        "columns"
    )
    required_scales = {target for target, _ in PAIR_LABELS} | {
        mv for _, mv in PAIR_LABELS
    }
    missing_scales = sorted(required_scales - set(scales.index))
    if missing_scales:
        raise ValueError(f"Teacher scale file is missing: {missing_scales}")
    if not np.isfinite(scales[list(required_scales)]).all() or not scales[list(required_scales)].gt(0).all():
        raise ValueError("Teacher normalization scales must be finite and positive.")

    teacher_path = model_files(input_root / MODEL_DIRS["No-Gain"], "gain_probe_diagnostics.csv")
    teacher = load_teacher_landscape(teacher_path, scales, args.chunk_size)
    teacher_summary = summarize_teacher(teacher)
    quartile_edges = build_quartile_edges(teacher)
    teacher_summary.to_csv(output_dir / "teacher_gain_summary.csv", index=False)
    quartile_edges.to_csv(output_dir / "gain_magnitude_quartile_edges.csv", index=False)
    teacher.to_csv(output_dir / "teacher_gain_landscape.csv", index=False)

    magnitude_frames = []
    sign_frames = []
    probe_frames = []
    for model_index, (model, model_dir) in enumerate(MODEL_DIRS.items()):
        print(f"[Info] Processing {model}", flush=True)
        diagnostic_path = model_files(input_root / model_dir, "gain_probe_diagnostics.csv")
        probe = load_probe_accuracy(
            diagnostic_path, scales, quartile_edges, args.chunk_size
        )
        validate_against_official(input_root, model, model_dir, probe)
        probe_frames.append(probe.assign(model=model))
        print(f"[Pass] {model}: all eight official KCI counts reproduced; {len(probe)} valid probe/pair clusters.", flush=True)
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
    pd.concat(probe_frames, ignore_index=True).to_csv(output_dir / "gain_pair_probe_accuracy.csv", index=False)
    assert len(magnitude_kci) == 128 and len(sign_kci) == 64
    verify_partition_summaries(magnitude_kci, sign_kci, probe_frames)

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
    finish_analysis(output_dir, protocol, teacher_summary, magnitude_kci, sign_kci)
    if any(digest(repo_root / name) != value for name, value in protected.items()):
        raise AssertionError("Analysis input changed during execution.")
    if any(digest(repo_root / name) != value for name, value in retained.items()):
        raise AssertionError("An existing test comparison output was changed.")
    validation = dict(
        passed=True, models=list(MODEL_DIRS), official_distribution_pair_counts_checked=32,
        teacher_landscape_rows=len(teacher), valid_probe_pair_clusters_per_model=[len(p) for p in probe_frames],
        conditional_magnitude_rows=128, conditional_sign_rows=64,
        shared_teacher_and_probe_arrays_verified=True, partitions_reconstruct_official_counts=True,
        bootstrap_resamples=args.bootstrap_resamples, bootstrap_unit="Operating point, 36 comparisons per cluster",
        protected_inputs_unchanged=True, retained_test_outputs_unchanged=True,
        visual_review="pending", output_sha256={name: digest(output_dir / name) for name in ANALYSIS_NAMES
                                               if (output_dir / name).is_file() and name != "gain_pair_difficulty_validation.json"},
    )
    (output_dir / "gain_pair_difficulty_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"[Done] Saved analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
