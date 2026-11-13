"""Overlay four fixed-model TS predictions against Aspen, one figure per case."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualization/best_gain_ss_gainss_test_seed42"
PREDICTIONS = RESULTS / "predictions"
OUTPUT = PREDICTIONS / "comparison_by_case"
MODELS = {
    "no_gain": {"label": "No-Gain", "color": "#0072B2", "linestyle": "--"},
    "gain": {"label": "Gain (250)", "color": "#D55E00", "linestyle": "-."},
    "ss_only": {"label": "SS-only (2.5)", "color": "#CC79A7", "linestyle": ":"},
    "gain_ss": {"label": "Gain+SS (2.5/2.5)", "color": "#009E73", "linestyle": (0, (5, 2, 1, 2))},
}
COLUMNS = ["time_min", "B35_H2S_true", "B35_SO2_true", "B35_H2S_pred", "B35_SO2_pred"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(case: str) -> tuple[int, int]:
    match = re.fullmatch(r"R5(?:-(\d+))?", case)
    if match is None:
        raise ValueError(f"Unexpected case name: {case}")
    return (0, int(match.group(1) or 0))


def load_cases():
    cases_by_model = {
        name: {p.parent.name for p in (PREDICTIONS / name).glob("*/time_series.csv")}
        for name in MODELS
    }
    reference_cases = cases_by_model["no_gain"]
    if not reference_cases or any(cases != reference_cases for cases in cases_by_model.values()):
        raise ValueError(f"The four models do not have the same cases: {cases_by_model}")
    sources = [RESULTS / "selected_models.csv", Path(__file__).resolve()]
    frames = {}
    for case in sorted(reference_cases, key=case_key):
        frames[case] = {}
        for model in MODELS:
            path = PREDICTIONS / model / case / "time_series.csv"
            sources.append(path)
            df = pd.read_csv(path)
            if list(df.columns) != COLUMNS or df.empty or not np.isfinite(df.to_numpy(dtype=float)).all():
                raise ValueError(f"Invalid time-series file: {path}")
            if not np.all(np.diff(df.time_min.to_numpy()) > 0):
                raise ValueError(f"Time must be strictly increasing: {path}")
            frames[case][model] = df
        ref = frames[case]["no_gain"]
        for model, df in frames[case].items():
            for col in ("time_min", "B35_H2S_true", "B35_SO2_true"):
                if not np.array_equal(ref[col].to_numpy(), df[col].to_numpy()):
                    raise ValueError(f"Different {col} in {case}, model={model}")
    return frames, sources


def make_figure(case: str, frames: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.5), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.80, bottom=0.105, hspace=0.22)
    fig.suptitle(f"{case}  |  Test time-series comparison", fontsize=16, fontweight="bold", y=0.965)
    truth = frames["no_gain"]
    for ax, target, title in zip(axes, ["B35_H2S", "B35_SO2"], [r"H$_2$S", r"SO$_2$"]):
        ax.plot(truth.time_min, truth[f"{target}_true"], label="Aspen", color="#171717", linewidth=2.2, zorder=5)
        for name, style in MODELS.items():
            df = frames[name]
            ax.plot(df.time_min, df[f"{target}_pred"], label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], linewidth=1.7, zorder=3)
        ax.set_ylabel(f"{title}\n(kmol/kmol)", fontsize=11)
        ax.grid(True, color="#D8DEE7", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("Time (min)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=5, frameon=False, fontsize=10.5, handlelength=2.8, columnspacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    cases, sources = load_cases()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    output.mkdir(parents=True, exist_ok=True)
    with PdfPages(output / "all_cases_TS_comparison.pdf") as pdf:
        for case, frames in cases.items():
            fig = make_figure(case, frames)
            fig.savefig(output / f"{case}_TS_comparison.png", dpi=180, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            print(f"Plotted {case}: {len(frames['no_gain'])} points", flush=True)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise RuntimeError("Source CSV, selected-model record or plotting script changed during plotting")
    provenance = {"cases": list(cases), "model_labels": {k: v["label"] for k, v in MODELS.items()},
                  "seed": 42, "prediction": "TS recursive rollout after 18 true initial history points, with known future decoder inputs",
                  "truth": "Aspen; exact per-case time and target equality across all four model CSVs checked",
                  "case_points": {case: len(frames["no_gain"]) for case, frames in cases.items()},
                  "source_sha256": before}
    (output / "plot_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(f"Wrote {len(cases)} PNGs and one PDF: {output}")


if __name__ == "__main__":
    main()
