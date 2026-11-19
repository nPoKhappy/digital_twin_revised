"""Draw the selected test comparison, retaining all three training seeds.

Run after the evaluator has written regional_coordinates.csv and
selected_models.json.  The CSV already uses matched No-Gain seed 42 RMSE as
its denominator.  This script does not recalculate metrics or choose models.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FAMILIES = ["no_gain", "gain", "ss_only", "gain_ss", "difficult_gradnorm"]
COLORS = dict(zip(FAMILIES, ["#4d4d4d", "#0072b2", "#009e73", "#d55e00", "#cc79a7"]))
MPL_MARKERS = {42: "o", 43: "s", 44: "^"}
PLOTLY_MARKERS = {42: "circle", 43: "square", 44: "triangle-up"}
PARTITIONS = [("internal", "Interpolation"), ("external", "Extrapolation")]


def label(family: str, alias: str) -> str:
    def number(s: str) -> str:
        return s.replace("p", ".")

    if family == "no_gain":
        return "No-Gain"
    if family == "gain":
        return "Gain = " + number(alias.removeprefix("gain"))
    if family == "ss_only":
        return "SS-only = " + number(alias.removeprefix("ss").removesuffix("_only"))
    if family == "gain_ss":
        match = re.fullmatch(r"gain(\d+(?:p\d+)?)_ss(\d+(?:p\d+)?)", alias)
        if not match:
            raise ValueError(f"Unrecognized Gain+SS alias: {alias}")
        return f"Gain + SS = {number(match[1])} / {number(match[2])}"
    if family == "difficult_gradnorm":
        return "Difficult-GradNorm, rho = " + number(alias.removeprefix("dg_ratio"))
    raise ValueError(f"Unexpected family {family}")


def padded_range(values: pd.Series, fraction: float) -> tuple[float, float]:
    lower, upper = float(values.min()), float(values.max())
    padding = max((upper - lower) * fraction, 1e-4)
    return lower - padding, upper + padding


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="completed evaluation directory")
    parser.add_argument("--output", type=Path, help="defaults to INPUT/three_seed_report")
    args = parser.parse_args()
    source = args.input
    out = args.output or source / "three_seed_report"
    coordinates_path, selection_path = source / "regional_coordinates.csv", source / "selected_models.json"
    validation_path = source / "evaluation_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise ValueError("Evaluation validation has not passed")
    selected = json.loads(selection_path.read_text(encoding="utf-8"))["selected_models"]
    df = pd.read_csv(coordinates_path)
    df["seed"] = df["seed"].astype(int)
    expected = {(m["family"], m["alias"], int(m["seed"])) for m in selected}
    present = set(zip(df.family, df.alias, df.seed))
    if len(selected) != 15 or len(expected) != 15 or present != expected or len(df) != 30:
        raise ValueError("Expected exactly 15 selected models with 30 regional points")
    if set(df.family) != set(FAMILIES):
        raise ValueError("Expected four model families and No-Gain")
    for family in FAMILIES:
        subset = df[df.family == family]
        if set(subset.seed) != {42, 43, 44} or subset.alias.nunique() != 1:
            raise ValueError(f"Expected exactly three seeds and one hyperparameter for {family}")
        for seed in (42, 43, 44):
            sub = subset[subset.seed == seed]
            if set(sub.partition) != {"internal", "external"} or len(sub) != 2:
                raise ValueError(f"Missing or duplicated regional points for {family}, seed {seed}")
    if not np.isfinite(df[["KCI_percent", "AvgRelRMSE"]].to_numpy()).all():
        raise ValueError("Nonfinite coordinates")
    reference = df[(df.family == "no_gain") & (df.seed == 42)]
    if not np.allclose(reference.AvgRelRMSE, 1.0, rtol=0, atol=1e-12):
        raise ValueError("No-Gain seed 42 does not equal the fixed baseline")
    labels = {family: label(family, df.loc[df.family == family, "alias"].iloc[0]) for family in FAMILIES}
    xlim = padded_range(df.KCI_percent, 0.06)
    xlim = (xlim[0], max(xlim[1], 100.5))
    x_ticks = np.arange(np.ceil(xlim[0] / 2) * 2, 100.1, 2).tolist()
    ylim = padded_range(df.AvgRelRMSE, 0.09)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.3), sharex=True, sharey=True, layout="constrained")
    for ax, (partition, title) in zip(axs, PARTITIONS):
        for family in FAMILIES:
            for seed in (42, 43, 44):
                r = df[(df.partition == partition) & (df.family == family) & (df.seed == seed)].iloc[0]
                ax.scatter(r.KCI_percent, r.AvgRelRMSE, s=85, c=COLORS[family], marker=MPL_MARKERS[seed],
                           edgecolor="white", linewidth=0.7, zorder=3)
        ax.axhline(1, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
        ax.set(title=title, xlabel="KCI (%)", xlim=xlim, ylim=ylim)
        ax.set_xticks(x_ticks)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("AvgRelRMSE")
    family_handles = [Line2D([], [], color=COLORS[f], marker="o", linestyle="None", markersize=7, label=labels[f]) for f in FAMILIES]
    seed_handles = [Line2D([], [], color="#444444", marker=MPL_MARKERS[s], linestyle="None", markersize=7, label=f"Seed {s}") for s in (42, 43, 44)]
    fig.legend(handles=family_handles + seed_handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"three_seed_test_comparison.{suffix}", dpi=350)
    plt.close(fig)

    chart = make_subplots(rows=1, cols=2, subplot_titles=[p[1] for p in PARTITIONS], shared_yaxes=True,
                          horizontal_spacing=0.08)
    for col, (partition, _) in enumerate(PARTITIONS, 1):
        for family in FAMILIES:
            d = df[(df.partition == partition) & (df.family == family)].sort_values("seed")
            hover = [f"{html.escape(labels[family])}<br>Seed {r.seed}<br>KCI: {r.KCI_percent:.3f}%<br>AvgRelRMSE: {r.AvgRelRMSE:.4f}<extra></extra>" for r in d.itertuples()]
            chart.add_trace(go.Scatter(x=d.KCI_percent, y=d.AvgRelRMSE, mode="markers", name=labels[family],
                legendgroup=family, showlegend=col == 1, hovertemplate=hover,
                marker=dict(size=12, color=COLORS[family], symbol=[PLOTLY_MARKERS[s] for s in d.seed],
                            line=dict(color="white", width=1))), row=1, col=col)
        chart.add_hline(y=1, line_dash="dash", line_color="#999999", line_width=1, row=1, col=col)
        chart.update_xaxes(title_text="KCI (%)", range=xlim, tickmode="array", tickvals=x_ticks, row=1, col=col)
        chart.update_yaxes(range=ylim, row=1, col=col)
    chart.update_yaxes(title_text="AvgRelRMSE", row=1, col=1)
    for seed in (42, 43, 44):
        chart.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Seed {seed}",
            marker=dict(size=11, color="#444444", symbol=PLOTLY_MARKERS[seed]), hoverinfo="skip"), row=1, col=1)
    chart.update_layout(template="plotly_white", title="Test comparison: all three seeds", height=660,
        legend=dict(orientation="h", yanchor="bottom", y=-0.27, x=0, font=dict(size=12)), margin=dict(b=180))
    div = chart.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True})
    rows = []
    for family in FAMILIES:
        for seed in (42, 43, 44):
            a = df[(df.family == family) & (df.seed == seed) & (df.partition == "internal")].iloc[0]
            b = df[(df.family == family) & (df.seed == seed) & (df.partition == "external")].iloc[0]
            rows.append(f"<tr><td>{html.escape(labels[family])}</td><td>{seed}</td>"
                        f"<td>{a.AvgRelRMSE:.4f}</td><td>{a.KCI_percent:.2f}%</td>"
                        f"<td>{b.AvgRelRMSE:.4f}</td><td>{b.KCI_percent:.2f}%</td></tr>")
    document = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test comparison: three seeds</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:20px;color:#202124}p{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f2f4f7} .table-wrap{overflow-x:auto} </style></head><body><h1>Test comparison：每組超參數的三個 seed</h1><p>相同 case、輸出及預測位置的 No-Gain seed 42 test RMSE 為共同分母；圓形、方形及三角形分別為 seed 42、43、44。KCI 為相對 ANN 參考增益的方向一致率。圖上可滑鼠移至各點查看參數與 seed，點圖例可隱藏或顯示模型。</p>""" + div + "<div class='table-wrap'><table><thead><tr><th>模型／超參數</th><th>Seed</th><th>內插 AvgRelRMSE</th><th>內插 KCI</th><th>外插 AvgRelRMSE</th><th>外插 KCI</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div></body></html>"
    (out / "three_seed_test_comparison.html").write_text(document, encoding="utf-8")
    df.to_csv(out / "three_seed_coordinates.csv", index=False, float_format="%.17g")
    summary = {"passed": True, "model_count": 15, "regional_points": len(df), "models": labels,
               "source_sha256": {p.name: sha256(p) for p in (coordinates_path, selection_path, validation_path)},
               "output_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "report_validation.json"}}
    (out / "report_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(df)} regional points to {out}")


if __name__ == "__main__":
    main()
