"""Compare ANN and Aspen gain signs on paired steady-state step cases.

Only paired Aspen cases marked Run Completed at both endpoints are evaluated.
The ANN finite-step result uses the same teacher, scalers and float32 model as
the existing KCI evaluator; the ANN local derivative is reported separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.evaluation import evaluate_gain_in_out_distribution as native


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "data/Claus_steady_state/lhs_step_change_valid.xlsx"
SAMPLING = ROOT / "data/sampling/ann_gain_validation_5000_20261108_025507_315943_UTC"
MODEL = ROOT / "saved_models/Tabular_MLP_5in_2out_QV.pth"
MEAN = ROOT / "results/steady_state/Tabular_MLP_New/zscore_mean.csv"
STD = ROOT / "results/steady_state/Tabular_MLP_New/zscore_std.csv"
OUTPUT = ROOT / "results/evaluation/ann_vs_aspen_step_change_valid"
INPUT_MAP = {
    "air2_SP": "B33.SPo.SPo", "HEATER2_output_T_SP": "B20.SPo.SPo",
    "acidgas_Fm": "B34.SPo.SPo", "acidgas_P": "ACIDGAS.P.P", "acidgas_T": "ACIDGAS.T.T",
}
TARGET_MAP = {"B35_H2S": 'S33.Zn.H2S.("H2S")', "B35_SO2": 'S33.Zn.SO2.("SO2")'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gain_sign(delta_y: float, delta_mv: float) -> int:
    if not np.isfinite(delta_y) or not np.isfinite(delta_mv) or delta_mv == 0:
        raise ValueError("Gain sign requires finite changes and nonzero signed MV step")
    return int(np.sign(delta_y / delta_mv))


def classify_sign(aspen_gain: float, ann_gain: float) -> str:
    if not np.isfinite(aspen_gain) or not np.isfinite(ann_gain):
        raise ValueError("Cannot classify non-finite gains")
    if aspen_gain == 0 and ann_gain == 0:
        return "both_zero"
    if aspen_gain == 0:
        return "aspen_zero"
    if ann_gain == 0:
        return "ann_zero"
    return "match" if np.sign(aspen_gain) == np.sign(ann_gain) else "mismatch"


def pair_status(base_status: str, step_status: str) -> str:
    base = base_status == "Run Completed"
    step = step_status == "Run Completed"
    if base and step:
        return "valid"
    if not base and not step:
        return "both_failed"
    return "step_failed" if base else "base_failed"


def valid_aspen_outputs(sheet: pd.DataFrame, rows: list[int]) -> bool:
    values = sheet.loc[rows, list(TARGET_MAP.values())].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all())


def load_step_magnitudes(sampling_dir: Path) -> dict[str, float]:
    report_path = sampling_dir / "generation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hashes = report.get("output_sha256", {})
    for name in ("aspen_inputs.csv", "row_manifest.csv"):
        if name not in hashes or digest(sampling_dir / name) != hashes[name]:
            raise ValueError(f"Sampling input hash mismatch or missing: {name}")
    if "step_magnitudes" in report:
        magnitudes = report["step_magnitudes"]
    elif "step_magnitude" in report:
        magnitudes = {name: report["step_magnitude"] for name in ("air2_SP", "HEATER2_output_T_SP")}
    else:
        raise ValueError("Sampling report does not define step magnitudes")
    steps = {name: float(magnitudes[name]) for name in ("air2_SP", "HEATER2_output_T_SP")}
    if not all(np.isfinite(value) and value > 0 for value in steps.values()):
        raise ValueError("Step magnitudes must be finite and positive")
    return steps


def load_cases(workbook: Path, sampling: Path):
    step_magnitudes = load_step_magnitudes(sampling)
    expected = pd.read_csv(sampling / "aspen_inputs.csv")
    manifest = pd.read_csv(sampling / "row_manifest.csv")
    sheet = pd.read_excel(workbook, sheet_name=0, header=2, skiprows=[3])
    if len(sheet) != len(expected) or len(sheet) != len(manifest):
        raise ValueError("Workbook, input and manifest row counts differ; cannot safely pair")
    if len(expected.columns) != 17 or not set(expected.columns).issubset(sheet.columns):
        raise ValueError("Missing expected Aspen input tags")
    if not {"Scenario", "Status", *TARGET_MAP.values()}.issubset(sheet.columns):
        raise ValueError("Workbook is missing a scenario, status or quality output")
    scenario = sheet["Scenario"].astype(str).str.fullmatch(r"Case\s+(\d+)")
    if not scenario.all():
        raise ValueError("Unexpected scenario ID; refusing to infer pairing from row position")
    ids = sheet["Scenario"].astype(str).str.extract(r"Case\s+(\d+)")[0].astype(int)
    if ids.duplicated().any() or set(ids) != set(range(1, len(expected) + 1)):
        raise ValueError("Workbook scenario IDs are missing, duplicated or outside input range")
    if manifest.input_row_1based.duplicated().any() or set(manifest.input_row_1based) != set(ids):
        raise ValueError("Manifest IDs are missing or duplicated")
    sheet = sheet.assign(input_row_1based=ids).sort_values("input_row_1based").reset_index(drop=True)
    manifest = manifest.sort_values("input_row_1based").reset_index(drop=True)
    numbers = sheet[list(expected.columns)].apply(pd.to_numeric, errors="coerce").to_numpy()
    original = expected.to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or not np.allclose(numbers, original, rtol=0, atol=1e-8):
        raise ValueError("Workbook input conditions do not match the designated source CSV")
    max_difference = float(np.max(np.abs(numbers - original)))
    if manifest.case_id.duplicated().any():
        raise ValueError("Duplicate case IDs in manifest")
    for pair_id, group in manifest.groupby("base_id", sort=False):
        if len(group) != 2 or set(group.case_type) != {"base", "perturb"}:
            raise ValueError(f"Incomplete pair: {pair_id}")
        a, b = [group[group.case_type.eq(v)].iloc[0] for v in ("base", "perturb")]
        for name in ["region", "stratum", "perturbed_mv", "mv_name", "direction"]:
            if a[name] != b[name]:
                raise ValueError(f"Inconsistent pair metadata: {pair_id} {name}")
        if b.mv_name not in ("air2_SP", "HEATER2_output_T_SP") or INPUT_MAP[b.mv_name] != b.perturbed_mv:
            raise ValueError(f"MV name and Aspen tag disagree: {pair_id}")
        base, step = original[int(a.input_row_1based)-1], original[int(b.input_row_1based)-1]
        diff = step - base
        wanted = np.zeros(len(expected.columns))
        wanted[expected.columns.get_loc(b.perturbed_mv)] = float(b.delta)
        if (not np.allclose(diff, wanted, rtol=0, atol=1e-10)
                or int(b.direction) not in (-1, 1)
                or not np.isclose(float(b.delta), int(b.direction) * step_magnitudes[b.mv_name], rtol=0, atol=1e-10)
                or float(a.delta) != 0):
            raise ValueError(f"Incorrect or non-single-MV step: {pair_id}")
    return sheet, manifest, max_difference


def predict_teacher(sheet: pd.DataFrame):
    device = torch.device("cpu")
    model, mean, std, target_mean, target_std = native.load_teacher(device)
    named = sheet[list(INPUT_MAP.values())].rename(columns={v: k for k, v in INPUT_MAP.items()})
    named = named[native.TAB_INPUT_COLS].apply(pd.to_numeric, errors="raise")
    z = (named.to_numpy(dtype=np.float64) - mean[native.TAB_INPUT_COLS].to_numpy()) / std[native.TAB_INPUT_COLS].to_numpy()
    with torch.no_grad():
        # Output scaling follows predict_tabular_v2: float32 NN, float64 CSV scalers.
        p32 = model(torch.tensor(z, dtype=torch.float32)).numpy().astype(np.float64)
        p64 = copy.deepcopy(model).double()(torch.tensor(z, dtype=torch.float64)).numpy()
    y_mean = mean[native.FULL_TARGET_COLS].to_numpy(dtype=np.float64)
    y_std = std[native.FULL_TARGET_COLS].to_numpy(dtype=np.float64)
    return named, model, mean, std, target_mean, target_std, p32 * y_std + y_mean, p64 * y_std + y_mean


def summary_table(detail: pd.DataFrame, keys: list[str], classification: str):
    grouped = detail.groupby(keys, dropna=False) if keys else [((), detail)]
    rows = []
    for group_key, df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        counts = df[classification].value_counts()
        total, matches = len(df), int(counts.get("match", 0))
        row = dict(zip(keys, group_key))
        row.update(comparisons=total, matches=matches, mismatches=int(counts.get("mismatch", 0)),
                   aspen_zero=int(counts.get("aspen_zero", 0)), ann_zero=int(counts.get("ann_zero", 0)),
                   both_zero=int(counts.get("both_zero", 0)), agreement_percent=100 * matches / total if total else None)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(sheet, manifest):
    named, model, mean, std, target_mean, target_std, pred, pred64 = predict_teacher(sheet)
    base_meta = manifest[manifest.case_type.eq("base")].set_index("base_id")
    step_meta = manifest[manifest.case_type.eq("perturb")].set_index("base_id")
    if set(base_meta.index) != set(step_meta.index):
        raise ValueError("Base and step pair IDs disagree")
    step_meta = step_meta.loc[base_meta.index]
    pairs, details = [], []
    valid_base_indices = []
    for pair_id in base_meta.index:
        a, b = base_meta.loc[pair_id], step_meta.loc[pair_id]
        i, j = int(a.input_row_1based)-1, int(b.input_row_1based)-1
        status = pair_status(sheet.at[i, "Status"], sheet.at[j, "Status"])
        if status == "valid" and not valid_aspen_outputs(sheet, [i, j]):
            status = "invalid_output"
        record = {"base_id": int(pair_id), "region": a.region, "stratum": a.stratum,
                  "mv": b.mv_name, "direction": int(b.direction), "delta_mv": float(b.delta),
                  "base_case": sheet.at[i, "Scenario"], "step_case": sheet.at[j, "Scenario"],
                  "base_row_1based": i+1, "step_row_1based": j+1,
                  "base_status": sheet.at[i, "Status"], "step_status": sheet.at[j, "Status"], "pair_status": status}
        # Newer sampling files record endpoint region; grouping by `region`
        # always means the base-point region, even when a step crosses it.
        for key in ("endpoint_region", "crosses_core_boundary", "endpoint_in_base_stratum"):
            if key in b.index:
                record[key] = b[key]
        pairs.append(record)
        if status != "valid":
            continue
        valid_base_indices.append(i)
        for k, target in enumerate(native.FULL_TARGET_COLS):
            aspen_a, aspen_b = float(sheet.at[i, TARGET_MAP[target]]), float(sheet.at[j, TARGET_MAP[target]])
            delta = float(b.delta)
            asp_gain = (aspen_b-aspen_a)/delta
            ann_gain = (pred[j,k]-pred[i,k])/delta
            double_gain = (pred64[j,k]-pred64[i,k])/delta
            details.append({**record, "target": target, "aspen_base": aspen_a, "aspen_step": aspen_b,
                            "ann_base_float32": pred[i,k], "ann_step_float32": pred[j,k],
                            "aspen_delta_y": aspen_b-aspen_a, "ann_delta_y_float32": pred[j,k]-pred[i,k],
                            "aspen_gain": asp_gain, "ann_gain_float32": ann_gain, "ann_gain_float64": double_gain,
                            "aspen_sign": gain_sign(aspen_b-aspen_a, delta), "ann_sign_float32": int(np.sign(ann_gain)),
                            "ann_sign_float64": int(np.sign(double_gain)),
                            "finite_step_comparison": classify_sign(asp_gain, ann_gain),
                            "finite_step_comparison_float64": classify_sign(asp_gain, double_gain),
                            "ann_float32_vs_float64_sign_changed": bool(np.sign(ann_gain) != np.sign(double_gain))})
    pairs = pd.DataFrame(pairs)
    detail = pd.DataFrame(details)
    if detail.empty:
        raise ValueError("No completed and numerically valid pairs to compare")
    # Use the exact teacher autograd implementation used in the KCI evaluator.
    gradients = native.teacher_gain_matrix(named.iloc[valid_base_indices], model, mean, std,
        target_mean, target_std, native.FULL_TARGET_COLS, ["air2_SP", "HEATER2_output_T_SP"], torch.device("cpu")).numpy()
    local, eligible = [], []
    for i, row in detail.iterrows():
        pair_index = i // len(native.FULL_TARGET_COLS)
        target_index = native.FULL_TARGET_COLS.index(row.target)
        mv_index = ["air2_SP", "HEATER2_output_T_SP"].index(row.mv)
        g = float(gradients[pair_index, target_index, mv_index])
        local.append(g)
        eligible.append(abs(g) >= 1e-5)
    detail["ann_local_gradient_float32"] = local
    detail["local_gradient_vs_aspen"] = [classify_sign(a, b) for a, b in zip(detail.aspen_gain, local)]
    detail["kci_teacher_threshold_1e_5"] = eligible
    return pairs, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--sampling-dir", type=Path, default=SAMPLING)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    sources = [args.workbook.resolve(), args.sampling_dir.resolve()/"aspen_inputs.csv",
               args.sampling_dir.resolve()/"row_manifest.csv", args.sampling_dir.resolve()/"generation_report.json", MODEL, MEAN, STD, Path(__file__).resolve(),
               ROOT/"scripts/evaluation/evaluate_gain_in_out_distribution.py", ROOT/"scripts/training/train_pgin_stepwise_gain.py"]
    before = {str(path): digest(path) for path in sources}
    sheet, manifest, max_difference = load_cases(args.workbook, args.sampling_dir)
    pairs, detail = evaluate(sheet, manifest)
    coverage = pairs.groupby(["region", "stratum", "mv", "direction", "pair_status"], dropna=False).size().rename("pairs").reset_index()
    strata = (pairs.assign(valid=pairs.pair_status.eq("valid")).groupby(["region", "stratum"])
              .valid.agg(total_pairs="size", completed_pairs="sum").reset_index())
    strata["completion_percent"] = 100 * strata.completed_pairs / strata.total_pairs
    overall = summary_table(detail, [], "finite_step_comparison")
    by_pair = summary_table(detail, ["region", "mv", "target"], "finite_step_comparison")
    by_direction = summary_table(detail, ["region", "mv", "target", "direction"], "finite_step_comparison")
    local = summary_table(detail, ["region", "mv", "target"], "local_gradient_vs_aspen")
    kci_valid = detail[detail.kci_teacher_threshold_1e_5]
    local_kci = summary_table(kci_valid, ["region", "mv", "target"], "local_gradient_vs_aspen")
    sensitivity = []
    for threshold in (0., 1e-6, 1e-5):
        selected = detail[detail.aspen_gain.abs() >= threshold] if threshold else detail
        frame = summary_table(selected, ["region", "mv", "target"], "finite_step_comparison")
        frame.insert(0, "min_abs_aspen_gain", threshold)
        sensitivity.append(frame)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {"pair_status.csv": pairs, "pair_status_by_stratum.csv": coverage,
        "coverage_by_stratum.csv": strata,
        "gain_comparisons.csv": detail, "mismatches.csv": detail[detail.finite_step_comparison.ne("match")],
        "overall_summary.csv": overall, "summary_by_region_mv_target.csv": by_pair,
        "summary_by_direction.csv": by_direction, "local_gradient_summary.csv": local,
        "local_gradient_kci_threshold_summary.csv": local_kci,
        "aspen_gain_threshold_sensitivity.csv": pd.concat(sensitivity, ignore_index=True)}.items():
        frame.to_csv(output/name, index=False, float_format="%.17g")
    if "crosses_core_boundary" in pairs:
        cross_coverage = pairs.groupby(["region", "endpoint_region", "crosses_core_boundary", "pair_status"], dropna=False).size().rename("pairs").reset_index()
        cross_coverage.to_csv(output/"pair_status_by_region_transition.csv", index=False)
        cross_summary = summary_table(detail, ["region", "endpoint_region", "mv", "target"], "finite_step_comparison")
        cross_summary.to_csv(output/"summary_by_region_transition.csv", index=False, float_format="%.17g")
    after = {str(path): digest(path) for path in sources}
    if before != after:
        raise RuntimeError("A protected source or checkpoint changed during evaluation")
    report = {"input_rows": len(sheet), "aspen_status_counts": sheet.Status.value_counts(dropna=False).to_dict(),
              "pair_status_counts": pairs.pair_status.value_counts().to_dict(), "valid_comparisons": len(detail),
              "max_input_absolute_difference": max_difference, "input_match_tolerance_atol": 1e-8,
              "finite_step_overall": overall.to_dict(orient="records"),
              "mismatch_distinct_pairs": int(detail.loc[detail.finite_step_comparison.eq("mismatch"), "base_id"].nunique()),
              "float32_vs_float64_sign_changed": int(detail.ann_float32_vs_float64_sign_changed.sum()),
              "teacher_threshold_comparisons": len(kci_valid), "step_magnitudes": load_step_magnitudes(args.sampling_dir), "protected_sha256": before,
              "primary_rule": "same sign of (y_step - y_base)/(MV_step - MV_base), exact zero categories; no near-zero filtering",
              "precision": "same float32 network as existing KCI; float64 network sensitivity separately",
              "local_gradient": "ANN autograd at base vs Aspen finite-step; KCI threshold 1e-5 applies only to ANN gradient",
              "caveat": "Aspen solver completed status is not independent proof of convergence; no time-response or closed-loop validation"}
    (output/"evaluation_protocol.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output/"README.md").write_text(
        "# ANN 與 Aspen 穩態 step Gain 方向比較\n\n"
        "以 Case 編號連結原始輸入與配對表，逐欄核對 17 個輸入。僅兩端均為 Run Completed 且輸出有效的配對納入。\n\n"
        "主要結果是 Aspen 與 ANN 的 (擾動後濃度－基準濃度)/帶號 ΔMV 之嚴格正負號比較，未排除近零 Gain；"
        "`gain_comparisons.csv` 保留逐組實際數值及零值分類。`mismatches.csv` 也包含零值，請以分類欄辨別。\n\n"
        "`local_gradient_summary.csv` 額外比較 ANN 基準點局部導數與 Aspen 指定步長的有限差；"
        "`local_gradient_kci_threshold_summary.csv` 套用 ANN teacher |gain| >= 1e-5，"
        "是 KCI teacher 條件下的補充結果，不是 Transformer KCI。\n\n"
        "`coverage_by_stratum.csv` 記錄以基準點為分區的完成數與比例；`pair_status.csv` 保留所有失敗組。"
        "若有跨區資訊，`pair_status_by_region_transition.csv` 與 `summary_by_region_transition.csv` 分別記錄兩端分區、配對狀態與方向一致率。\n\n"
        "`aspen_gain_threshold_sensitivity.csv` 的 Aspen 增益門檻僅供敏感度比較，不宣稱是模擬器的可靠性門檻。"
        "成功狀態不保證獨立檢查過收斂，且不驗證動態過程或閉迴路。\n", encoding="utf-8")
    print(json.dumps({"pairs": report["pair_status_counts"], "overall": report["finite_step_overall"],
                      "float32_vs_float64_sign_changed": report["float32_vs_float64_sign_changed"],
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
