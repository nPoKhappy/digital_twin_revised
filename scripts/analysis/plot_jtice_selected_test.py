"""Redraw the selected test evaluation for a provisional JTICE submission.

Reads the completed evaluation and never selects models or recalculates metrics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/visualization/retrain_lhs10_23_selected_test_nogain42/updated_validation_best"
OUT = SOURCE / "jtice_figures"
FAMILIES = {
    "no_gain": ("N", "No-Gain", "#333333"),
    "gain": ("G", "Gain = 25", "#0072B2"),
    "ss_only": ("S", "SS-only = 2.5", "#009E73"),
    "gain_ss": ("C", "Gain + SS = 25 / 25", "#D55E00"),
    "difficult_gradnorm": ("D", r"Difficult-GradNorm, $\rho$ = 1.5", "#CC79A7"),
}
SEEDS = {42: "o", 43: "s", 44: "^"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    coordinates = SOURCE / "regional_coordinates.csv"
    selection = SOURCE / "selected_models.json"
    validation = SOURCE / "evaluation_validation.json"
    if json.loads(validation.read_text(encoding="utf-8")).get("passed") is not True:
        raise ValueError("Evaluation validation has not passed")
    df = pd.read_csv(coordinates)
    selected = json.loads(selection.read_text(encoding="utf-8"))["selected_models"]
    aliases = {family: next(m["alias"] for m in selected if m["family"] == family) for family in FAMILIES}
    labels = {
        "no_gain": "No-Gain",
        "gain": "Gain = " + aliases["gain"].removeprefix("gain").replace("p", "."),
        "ss_only": "SS-only = " + aliases["ss_only"].removeprefix("ss").removesuffix("_only").replace("p", "."),
        "gain_ss": "Gain + SS = " + aliases["gain_ss"].removeprefix("gain").replace("_ss", " / ").replace("p", "."),
        "difficult_gradnorm": r"Difficult-GradNorm, $\rho$ = " + aliases["difficult_gradnorm"].removeprefix("dg_ratio").replace("p", "."),
    }
    families = {family: (code, labels[family], color) for family, (code, _, color) in FAMILIES.items()}
    expected = {(m["family"], m["alias"], int(m["seed"])) for m in selected}
    found = set(zip(df.family, df.alias, df.seed.astype(int)))
    if len(df) != 30 or len(selected) != 15 or len(expected) != 15 or found != expected:
        raise ValueError("Expected 15 selected models and 30 regional records")
    if set(df.family) != set(FAMILIES) or not np.isfinite(df[["KCI_percent", "AvgRelRMSE"]]).all().all():
        raise ValueError("Unexpected families or nonfinite values")
    for family, alias in aliases.items():
        subset = df[df.family == family]
        if set(subset.alias) != {alias} or set(subset.seed.astype(int)) != set(SEEDS):
            raise ValueError(f"Unexpected selection for {family}")
        for seed in SEEDS:
            one = subset[subset.seed == seed]
            if len(one) != 2 or set(one.partition) != {"internal", "external"}:
                raise ValueError(f"Missing regional observation: {family} {seed}")
    if not np.allclose(df[(df.family == "no_gain") & (df.seed == 42)].AvgRelRMSE, 1.0, atol=1e-12, rtol=0):
        raise ValueError("Incorrect fixed seed-42 baseline")
    xmin = min(85.7, float(df.KCI_percent.min()) - 0.8)
    xmax = max(100.75, float(df.KCI_percent.max()) + 0.8)
    ymin = min(0.53, float(df.AvgRelRMSE.min()) * 0.93)
    ymax = max(1.17, float(df.AvgRelRMSE.max()) * 1.07)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": 0.65})
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(190 / 25.4, 97 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.31, wspace=0.18)
    for ax, (partition, title) in zip(axs, [("internal", "(a) Interpolation"), ("external", "(b) Extrapolation")]):
        for family, (_, _, color) in families.items():
            for seed, marker in SEEDS.items():
                row = df[(df.partition == partition) & (df.family == family) & (df.seed == seed)].iloc[0]
                ax.scatter(row.KCI_percent, row.AvgRelRMSE, s=34, color=color, marker=marker,
                           edgecolor="white", linewidth=0.55, zorder=3)
        ax.axhline(1, color="#777777", linewidth=0.65, linestyle="--", zorder=0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("KCI (%)")
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(np.ceil(xmin / 2) * 2, 100.01, 2))
        ax.set_ylim(ymin, ymax)
        ax.tick_params(width=0.65, length=2.5)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("AvgRelRMSE")
    family_handles = [Line2D([], [], color=c, marker="o", linestyle="None", markersize=5,
                             label=label) for _, label, c in families.values()]
    seed_handles = [Line2D([], [], color="#333333", marker=marker, linestyle="None", markersize=5,
                           label=f"Seed {seed}") for seed, marker in SEEDS.items()]
    fig.legend(handles=family_handles, loc="lower center", bbox_to_anchor=(0.5, 0.115),
               ncol=3, frameon=False, fontsize=7, handletextpad=0.35, columnspacing=0.8)
    fig.legend(handles=seed_handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=3, frameon=False, fontsize=7, handletextpad=0.35, columnspacing=1.2)
    pdf = OUT / "selected_test_jtice.pdf"
    png = OUT / "selected_test_jtice.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=1000, facecolor="white")
    plt.close(fig)
    grayscale = OUT / "grayscale_check_preview.png"
    with Image.open(png) as preview:
        preview.convert("L").resize((2000, 1021), Image.Resampling.LANCZOS).save(grayscale)

    caption = ("Test comparison of models selected using validation data. (a) Interpolation and (b) near extrapolation. "
               "Each symbol is one model training seed (42, 43 or 44; 15 models per panel). "
               "KCI measures directional agreement with the shared artificial neural network "
               "(ANN) reference. The average relative root mean squared error (AvgRelRMSE) uses the corresponding "
               "No-Gain seed-42 test root mean squared error as denominator. KCI is evaluated on steady-state probe "
               "points; AvgRelRMSE uses dynamic test cases. Colours identify the model family; "
               "marker shapes identify the training seed. The dashed horizontal line is the No-Gain seed-42 baseline. "
               "Points are individual runs; no error bars or significance tests are shown.")
    (OUT / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    notes = ("Prepared using the JTICE guide text supplied by the user; confirm the current journal instructions "
             "before submission. The PDF is the vector master and the PNG is a preview. The caption is separate. "
             "Colour identifies the model family and marker shape identifies seed. After removal of point labels, "
             "the model families are not reliably distinguishable in grayscale. "
             "These figures reuse the completed test evaluation; they do not constitute a fresh blind test.\n")
    (OUT / "figure_notes.txt").write_text(notes, encoding="utf-8")
    report = {"status": "user-supplied JTICE guide snapshot; confirm current guide before submission",
              "num_models": 15, "regional_points": len(df), "seeds": list(SEEDS),
              "figure_width_mm": 190, "figure_height_mm": 97, "png_dpi": 1000,
              "data_source_sha256": {p.name: sha(p) for p in (coordinates, selection, validation)},
              "figure_source_sha256": sha(Path(__file__)),
              "output_sha256": {p.name: sha(p) for p in (pdf, png, grayscale, OUT / "caption.txt", OUT / "figure_notes.txt")}}
    (OUT / "plot_provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
