"""Train one isolated 5-input/2-output ANN using explicitly listed LHS files.

Unlike the legacy trainer, this does not overwrite the current ANN or scalers.
For files 10–22, all rows (including failed cases) are connected within each
source when four of five operating inputs match to eight decimal places.
For paired file 23, the verified original base/step IDs are used instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.training.train_tabular_v2 import INV_MAPPING, SimpleTabularMLP


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/Claus_steady_state"
SOURCE_IDS = (10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23)
INPUTS = ["air2_SP", "HEATER2_output_T_SP", "acidgas_Fm", "acidgas_P", "acidgas_T"]
TARGETS = ["B35_H2S", "B35_SO2"]
PAIRED_DIR = ROOT / "data/sampling/ann_gain_validation_5000_20261108_025507_315943_UTC"
EXPECTED_FILE_23_SHA256 = "5b53494e8a787b3b3581758528cb4e5da5152c9d966a721eb4eae707b0ac2b1c"
GROUP_DECIMALS = 8
MAX_USABLE_COMPONENT_SHARE = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_sources(overrides: list[Path] | None = None) -> list[Path]:
    paths = ([p.resolve() for p in overrides] if overrides else
             [(DATA_DIR / f"lhs_generated_dynamic_ss_data_{n}.xlsx").resolve() for n in SOURCE_IDS])
    if not paths or len(set(paths)) != len(paths) or len({p.name for p in paths}) != len(paths):
        raise ValueError("Specify unique input files with unique filenames")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("One or more specified ANN input workbooks do not exist")
    return paths


def load_pair_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    report = json.loads((PAIRED_DIR / "generation_report.json").read_text(encoding="utf-8"))
    for name in ("aspen_inputs.csv", "row_manifest.csv"):
        expected = report.get("output_sha256", {}).get(name)
        if not expected or sha256(PAIRED_DIR / name) != expected:
            raise ValueError(f"Pair provenance hash mismatch: {name}")
    inputs = pd.read_csv(PAIRED_DIR / "aspen_inputs.csv")
    manifest = pd.read_csv(PAIRED_DIR / "row_manifest.csv")
    ids = manifest.input_row_1based
    if len(inputs) != 5000 or len(manifest) != 5000 or ids.duplicated().any() or set(ids) != set(range(1, 5001)):
        raise ValueError("Expected 5000 unique input and manifest rows for source 23")
    groups = manifest.groupby("base_id").case_type.agg(list)
    if len(groups) != 2500 or not groups.map(lambda x: sorted(x) == ["base", "perturb"]).all():
        raise ValueError("Expected 2500 complete source-23 base/step pairs")
    return inputs, manifest.set_index("input_row_1based").sort_index()


def connected_component_ids(input_values: np.ndarray) -> tuple[np.ndarray, dict]:
    """Connect records whose other four inputs agree within the rounded key.

    Run on all source rows, including failed Aspen cases: a failed row may be
    a necessary bridge between two successful single-variable perturbations.
    """
    values = np.asarray(input_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(INPUTS) or not np.isfinite(values).all():
        raise ValueError("All raw input rows must contain five finite operating conditions")
    keys = np.round(values, GROUP_DECIMALS)
    parent = np.arange(len(keys))
    sizes = np.ones(len(keys), dtype=int)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]

    edge_counts = {}
    for axis, name in enumerate(INPUTS):
        first, matches = {}, 0
        for index, row in enumerate(keys):
            key = tuple(value for i, value in enumerate(row) if i != axis)
            if key in first:
                union(index, first[key])
                matches += 1
            else:
                first[key] = index
        edge_counts[name] = matches
    roots = [find(i) for i in range(len(keys))]
    first_index = {}
    for index, root in enumerate(roots):
        first_index.setdefault(root, index)
    return np.array([first_index[root] for root in roots], dtype=int), edge_counts


def cross_source_four_input_matches(evidence: pd.DataFrame) -> dict[str, int]:
    keys = evidence[INPUTS].round(GROUP_DECIMALS)
    keys["source_file"] = evidence.source_file.to_numpy()
    keys = keys.drop_duplicates()
    result = {}
    for varying in INPUTS:
        others = [name for name in INPUTS if name != varying]
        counts = keys.groupby(others, dropna=False).source_file.nunique()
        result[varying] = int((counts > 1).sum())
    return result


def load_training_rows(paths: list[Path]) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    has_paired_file = any(path.name == "lhs_generated_dynamic_ss_data_23.xlsx" for path in paths)
    expected_inputs, pair_manifest = load_pair_manifest() if has_paired_file else (None, None)
    required = [INV_MAPPING[c] for c in INPUTS + TARGETS]
    frames, counts, raw_evidence = [], {}, []
    for path in paths:
        raw = pd.read_excel(path, sheet_name=0, header=2)
        raw = raw.iloc[1:].dropna(how="all").copy()  # first data row is the unit row
        if "Status" not in raw or "Scenario" not in raw or not set(required).issubset(raw):
            raise ValueError(f"Missing required status, scenario or input/target columns: {path}")
        operating = raw[required[:len(INPUTS)]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        component_ids, edge_counts = connected_component_ids(operating)
        if path.name == "lhs_generated_dynamic_ss_data_23.xlsx":
            if sha256(path) != EXPECTED_FILE_23_SHA256:
                raise ValueError("Source 23 no longer matches the verified paired validation workbook")
            ids = raw.Scenario.astype(str).str.extract(r"^Case\s+(\d+)$")[0]
            if ids.isna().any() or ids.astype(int).duplicated().any() or set(ids.astype(int)) != set(range(1, 5001)):
                raise ValueError("Source 23 Scenario IDs must be the unique 5000 original input cases")
            by_id = raw.assign(_id=ids.astype(int)).sort_values("_id")
            original_columns = list(expected_inputs.columns)
            if not set(original_columns).issubset(by_id):
                raise ValueError("Source 23 is missing an Aspen input tag")
            actual = by_id[original_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            expected = expected_inputs.to_numpy(dtype=float)
            if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=1e-8):
                raise ValueError("Source 23 Aspen inputs do not match the recorded paired cases")
        if path.name == "lhs_generated_dynamic_ss_data_23.xlsx":
            ids = raw.Scenario.astype(str).str.extract(r"^Case\s+(\d+)$")[0].astype(int)
            pair_ids = np.array([pair_manifest.loc[i, "base_id"] for i in ids], dtype=int)
            comparison = pd.DataFrame({"component": component_ids, "pair": pair_ids})
            if ((comparison.groupby("component").pair.nunique() != 1).any() or
                    (comparison.groupby("pair").component.nunique() != 1).any()):
                raise ValueError("Source 23 inferred 4-of-5 components disagree with its verified pair manifest")
            group_values = [f"{path.name}:pair_{n}" for n in pair_ids]
        else:
            group_values = [f"{path.name}:component_row_{int(raw.index[n] + 4)}" for n in component_ids]
        raw_groups = pd.Series(group_values, index=raw.index)
        complete = raw[raw.Status.eq("Run Completed")].copy()
        numeric = complete[required].apply(pd.to_numeric, errors="coerce")
        keep = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        complete, numeric = complete.loc[keep], numeric.loc[keep]
        numeric.columns = INPUTS + TARGETS
        numeric["source_file"] = path.name
        numeric["scenario"] = complete.Scenario.astype(str)
        numeric["excel_row_1based"] = complete.index.to_numpy() + 4
        numeric["group_id"] = raw_groups.loc[complete.index].to_numpy()
        evidence = pd.DataFrame(operating, index=raw.index, columns=INPUTS)
        evidence["source_file"] = path.name
        evidence["scenario"] = raw.Scenario.astype(str)
        evidence["excel_row_1based"] = raw.index.to_numpy() + 4
        evidence["status"] = raw.Status
        evidence["group_id"] = raw_groups
        evidence["training_eligible"] = evidence.index.isin(numeric.index)
        raw_evidence.append(evidence)
        group_sizes = evidence.groupby("group_id").size()
        size_histogram = group_sizes.value_counts().sort_index()
        counts[path.name] = {"rows": len(raw), "completed": len(raw[raw.Status.eq("Run Completed")]),
                             "usable": len(numeric), "components_all_rows": len(group_sizes),
                             "components_with_usable_rows": int(numeric.group_id.nunique()),
                             "max_component_raw_rows": int(group_sizes.max()),
                             "component_size_histogram": {str(k): int(v) for k, v in size_histogram.items()},
                             "four_input_key_matches_by_varied_input": edge_counts,
                             "grouping": ("verified original base_id; independently checked against 4-of-5 connectivity" if path.name.endswith("_23.xlsx") else "4-of-5 input connectivity, all statuses")}
        frames.append(numeric)
    combined = pd.concat(frames, ignore_index=True)
    evidence = pd.concat(raw_evidence, ignore_index=True)
    if combined.empty:
        raise ValueError("No finite completed LHS rows")
    cross_source = cross_source_four_input_matches(evidence)
    if any(cross_source.values()):
        raise ValueError(f"Cross-source four-input matches may connect split groups: {cross_source}")
    usable_group_sizes = combined.groupby("group_id").size()
    max_share = float(usable_group_sizes.max() / len(combined))
    if max_share > MAX_USABLE_COMPONENT_SHARE:
        largest = str(usable_group_sizes.idxmax())
        raise ValueError(f"Largest inferred component {largest} contains {max_share:.2%} of usable data; inspect before training")
    # Exact and near-exact 5-input duplicates must not span distinct groups.
    # If data changes, stop rather than silently allowing cross-split leakage.
    keys = combined[INPUTS].round(8).copy()
    keys["group_id"] = combined.group_id
    duplicate_keys = keys.duplicated(INPUTS, keep=False)
    if duplicate_keys.any():
        different_groups = keys.loc[duplicate_keys].groupby(INPUTS, dropna=False).group_id.nunique()
        if (different_groups > 1).any():
            raise ValueError("Duplicate input conditions across groups; assign them the same split group before training")
    group_audit = {"input_rounding_decimals": GROUP_DECIMALS,
                   "cross_source_four_input_matching_keys": cross_source,
                   "largest_usable_component_size": int(usable_group_sizes.max()),
                   "largest_usable_component_share": max_share,
                   "max_usable_component_share_allowed": MAX_USABLE_COMPONENT_SHARE,
                   "note": "File 11 contains large but exact 4-of-5 connected components caused by clipped boundary chains; each is conservatively kept in one split. Connectivity does not prove a unique physical originating base."}
    return combined, counts, evidence, group_audit


def grouped_split(data: pd.DataFrame, split_seed: int = 42) -> pd.DataFrame:
    groups = data.group_id.drop_duplicates().to_numpy()
    if len(groups) < 10:
        raise ValueError("At least 10 groups are required to form train/valid/test")
    shuffled = np.random.default_rng(split_seed).permutation(groups)
    train_end, valid_end = int(len(groups) * .8), int(len(groups) * .9)
    mapping = {group: split for split, subset in (
        ("train", shuffled[:train_end]), ("valid", shuffled[train_end:valid_end]),
        ("test", shuffled[valid_end:])) for group in subset}
    result = data.copy()
    result["split"] = result.group_id.map(mapping)
    if result.split.isna().any() or (result.groupby("group_id").split.nunique() != 1).any():
        raise ValueError("Split failed to keep groups together")
    return result


def metrics(truth: np.ndarray, prediction: np.ndarray) -> list[dict]:
    output = []
    for i, name in enumerate(TARGETS):
        actual, predicted = truth[:, i], prediction[:, i]
        residual = predicted - actual
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        output.append({"target": name, "n": len(actual), "MAE": float(np.mean(np.abs(residual))),
                       "RMSE": float(np.sqrt(np.mean(residual ** 2))),
                       "R2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else None,
                       "MAPE": float(np.mean(np.abs(residual) / np.maximum(np.abs(actual), 1e-10)) * 100)})
    return output


def write_compatible_plots(output: Path, history: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history.epoch, history.train_mse, label="Train")
    ax.plot(history.epoch, history.valid_mse, label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE (standardized targets)", title="Tabular MLP training loss")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curve.png", dpi=180)
    plt.close(fig)
    for index, name in enumerate(TARGETS):
        actual, predicted = truth[:, index], prediction[:, index]
        low, high = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
        margin = (high - low) * .05 or 1e-6
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(actual, predicted, color="#1f77b4", alpha=.62, s=16, edgecolors="none")
        ax.plot([low-margin, high+margin], [low-margin, high+margin], color="#d62728", linestyle="--", lw=1.6)
        ax.set(xlim=(low-margin, high+margin), ylim=(low-margin, high+margin),
               xlabel="Aspen (kmol/kmol)", ylabel="ANN (kmol/kmol)", title=f"{name} test parity")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(output / f"parity_plot_{name}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="ANN initialization/training seed")
    parser.add_argument("--split-seed", type=int, default=42, help="seed for group-level data splitting")
    parser.add_argument("--data-files", type=Path, nargs="+", help="Explicit workbook paths; default: 11 root-level LHS files 10,11,15-23")
    parser.add_argument("--output-dir", type=Path, help="New output directory; never overwrites an existing path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = (args.output_dir or ROOT / f"results/steady_state/ann_retrain_lhs_seed{args.seed}").resolve()
    if output.exists():
        raise FileExistsError(f"Output path already exists; choose a new --output-dir: {output}")
    files = input_sources(args.data_files)
    file23 = next((p for p in files if p.name == "lhs_generated_dynamic_ss_data_23.xlsx"), None)
    extra = [PAIRED_DIR/name for name in ("aspen_inputs.csv", "row_manifest.csv", "generation_report.json")] if file23 else []
    protect = files + extra + [Path(__file__).resolve(), ROOT / "scripts/training/train_tabular_v2.py"]
    before = {str(p): sha256(p) for p in protect}
    if file23 and before[str(file23)] != EXPECTED_FILE_23_SHA256:
        raise ValueError("Source 23 is not the verified ±2.5 paired workbook")
    data, file_counts, evidence, group_audit = load_training_rows(files)
    data = grouped_split(data, args.split_seed)
    train = data[data.split.eq("train")]
    valid = data[data.split.eq("valid")]
    test = data[data.split.eq("test")]
    if min(map(len, (train, valid, test))) == 0:
        raise ValueError("Each split must be nonempty")
    x_train, y_train = train[INPUTS].astype("float32"), train[TARGETS].astype("float32")
    x_mean, x_std = x_train.mean(), x_train.std().replace(0, 1)
    y_mean, y_std = y_train.mean(), y_train.std().replace(0, 1)
    if not np.isfinite(pd.concat([x_mean, x_std, y_mean, y_std]).to_numpy()).all():
        raise ValueError("Non-finite train-only scaler")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    def tensors(frame):
        x = ((frame[INPUTS].astype("float32") - x_mean) / x_std).to_numpy(dtype=np.float32)
        y = ((frame[TARGETS].astype("float32") - y_mean) / y_std).to_numpy(dtype=np.float32)
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    drop_last = len(train) % 32 == 1
    train_loader = DataLoader(tensors(train), batch_size=32, shuffle=True, drop_last=drop_last)
    valid_loader = DataLoader(tensors(valid), batch_size=32, shuffle=False)
    model = SimpleTabularMLP(input_dim=len(INPUTS), output_dim=len(TARGETS)).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output / "split_manifest.csv", index=False, float_format="%.17g")
    group_to_split = data.drop_duplicates("group_id").set_index("group_id").split
    evidence["split"] = evidence.group_id.map(group_to_split).fillna("no_usable_rows")
    component_audit = evidence.groupby(["source_file", "group_id", "split"], dropna=False).agg(
        raw_rows=("scenario", "size"), completed_rows=("status", lambda x: int(x.eq("Run Completed").sum())),
        usable_rows=("training_eligible", "sum")).reset_index()
    evidence.to_csv(output / "group_evidence_all_rows.csv", index=False, float_format="%.17g")
    component_audit.to_csv(output / "group_component_summary.csv", index=False)
    test[INPUTS + TARGETS].to_csv(output / "tabular_test_dataset.csv", index=False, float_format="%.17g")
    pd.concat([x_mean, y_mean]).to_csv(output / "zscore_mean.csv")
    pd.concat([x_std, y_std]).to_csv(output / "zscore_std.csv")
    config = {"input_files": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in files], "file_counts": file_counts,
              "split_rule": "80/10/10 of connected input groups; sources 10-22 use all-row 4-of-5 connectivity; source 23 uses verified base/step pair IDs",
              "group_audit": group_audit,
              "split_seed": args.split_seed, "training_seed": args.seed, "split_rows": data.split.value_counts().to_dict(),
              "split_group_counts": data.groupby("split").group_id.nunique().to_dict(),
              "input_columns": INPUTS, "target_columns": TARGETS, "source_sha256": before,
              "architecture": "SimpleTabularMLP from train_tabular_v2: 5-64-32-2, BatchNorm, GELU, dropout=0.1",
              "batch_size": 32, "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.001,
              "epochs_max": 1000, "early_stopping_patience": 50, "device": str(device),
              "epoch_mse_aggregation": "Arithmetic mean of per-batch mean MSE, matching legacy trainer; the smaller last validation batch has equal batch weight in early stopping.",
              "drop_last_if_training_remainder_one": drop_last, "torch_version": str(torch.__version__),
              "held_out_reference": ("Source 23 was previously used for ±2.5 ANN/Aspen gain evaluation; it is now training data, not an independent new-teacher validation dataset." if file23 else "Source 23 not included in this run."),
              "MAPE_definition": "mean(abs(pred-true)/max(abs(true),1e-10))*100; unstable when concentration is close to zero"}
    (output / "training_protocol.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    best, best_epoch, stale = float("inf"), 0, 0
    log_path = output / "epoch_loss.csv"
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("epoch,train_mse,valid_mse,best\n")
        for epoch in range(1, 1001):
            model.train()
            losses = []
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))
            model.eval()
            losses = []
            with torch.no_grad():
                for bx, by in valid_loader:
                    losses.append(criterion(model(bx.to(device)), by.to(device)).item())
            valid_loss = float(np.mean(losses))
            if not np.isfinite(train_loss) or not np.isfinite(valid_loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            improved = valid_loss < best
            if improved:
                best, best_epoch, stale = valid_loss, epoch, 0
                torch.save(model.state_dict(), output / "teacher.pth")
            else:
                stale += 1
            log.write(f"{epoch},{train_loss:.17g},{valid_loss:.17g},{int(improved)}\n")
            log.flush()
            if epoch == 1 or epoch % 20 == 0 or stale >= 50:
                print(f"epoch={epoch} train_mse={train_loss:.8g} valid_mse={valid_loss:.8g} best_epoch={best_epoch} patience={stale}/50", flush=True)
            if stale >= 50:
                break
    model.load_state_dict(torch.load(output / "teacher.pth", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        predicted_z = model(tensors(test).tensors[0].to(device)).cpu().numpy()
    prediction = predicted_z.astype(np.float64) * y_std.to_numpy(dtype=np.float64) + y_mean.to_numpy(dtype=np.float64)
    truth = test[TARGETS].to_numpy(dtype=np.float64)
    predictions = test[["source_file", "scenario", "excel_row_1based", "group_id"]].reset_index(drop=True)
    for index, name in enumerate(TARGETS):
        predictions[f"{name}_true"] = truth[:, index]
        predictions[f"{name}_pred"] = prediction[:, index]
    predictions.to_csv(output / "test_predictions.csv", index=False, float_format="%.17g")
    measured = pd.DataFrame(metrics(truth, prediction))
    measured.to_csv(output / "test_metrics.csv", index=False, float_format="%.17g")
    compatible_predictions = pd.DataFrame({name: truth[:, i] for i, name in enumerate(TARGETS)})
    for i, name in enumerate(TARGETS):
        compatible_predictions[f"{name}_pred"] = prediction[:, i]
    compatible_predictions.to_csv(output / "tabular_predictions.csv", index=False, float_format="%.17g")
    measured.rename(columns={"target": "Variable"})[["Variable", "MAE", "RMSE", "R2", "MAPE"]].to_csv(
        output / "tabular_metrics.csv", index=False, float_format="%.17g")
    write_compatible_plots(output, pd.read_csv(log_path), truth, prediction)
    after = {str(p): sha256(p) for p in protect}
    if before != after:
        raise RuntimeError("Protected source files changed during training; do not use this run")
    report = {"completed": True, "best_epoch": best_epoch, "best_valid_mse": best,
              "epochs_run": epoch, "test_metrics": metrics(truth, prediction),
              "output_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "training_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
