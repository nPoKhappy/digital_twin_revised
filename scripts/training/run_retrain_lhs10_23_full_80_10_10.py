"""Run one seed of the independent full-series 80/10/10 experiment.

Existing results are never cleared or overwritten. Use --check-only to inspect
the full manifest and sources without creating files or starting training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/pgin/retrain_lhs10_23_full_80_10_10/config_manifest.json"
RESULT_ROOT = ROOT / "results/pgin/retrain_lhs10_23_full_80_10_10"
PREFIX = "lhs10_23_full_80_10_10_"
MODULES = {
    "no_gain": "scripts.training.train_pgin_stepwise_gain_rawsplit",
    "gain": "scripts.training.train_pgin_stepwise_gain_rawsplit",
    "ss_only": "scripts.training.train_pgin_stepwise_gain_ss_loss_rawsplit",
    "gain_ss": "scripts.training.train_pgin_stepwise_gain_ss_loss_rawsplit",
    "difficult_gradnorm": "scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm_rawsplit",
}
LHS_NAMES = {f"lhs_generated_dynamic_ss_data_{i}.xlsx" for i in (10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23)}
DYNAMIC_NAMES = {f"Test_dataform_change_air2_R={case}.csv" for case in ("5", "5-1", "5-2", "5-6")}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def publish_reference(path: Path, sources: dict[str, str]) -> None:
    """Publish a complete reference atomically when seed processes race."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({"created_utc": now(), "sources_sha256": sources}, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8"))["sources_sha256"] != sources:
                raise ValueError("Other seed created a different source reference")
    finally:
        temporary.unlink(missing_ok=True)


def within(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def verify(hashes: dict[str, str]) -> None:
    for name, digest in hashes.items():
        path = Path(name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Protected input changed or is missing: {path}")


def prepare(seed: int) -> tuple[list[dict], dict[str, str]]:
    jobs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if len(jobs) != 63 or any(sum(j["seed"] == s for j in jobs) != 21 for s in (42, 43, 44)):
        raise ValueError("Manifest must contain exactly 21 jobs per seed")
    if len({j["exp_name"] for j in jobs}) != 63:
        raise ValueError("Manifest experiment names must be unique")

    ann_dir = ROOT / "results/steady_state/Tabular_MLP_New"
    ann_report = ann_dir / "training_result.json"
    ann_protocol = ann_dir / "training_protocol.json"
    report = json.loads(ann_report.read_text(encoding="utf-8"))
    protocol = json.loads(ann_protocol.read_text(encoding="utf-8"))
    if not report.get("completed"):
        raise ValueError("The shared ANN training report is not complete")
    teacher = ROOT / "saved_models/Tabular_MLP_5in_2out_QV.pth"
    artifacts = {"teacher.pth": teacher, "zscore_mean.csv": ann_dir / "zscore_mean.csv",
                 "zscore_std.csv": ann_dir / "zscore_std.csv", "training_protocol.json": ann_protocol}
    for name, path in artifacts.items():
        if not path.is_file() or sha(path) != report["output_sha256"][name]:
            raise ValueError(f"ANN artifact does not match its training report: {path}")

    lhs_dir = ROOT / "data/Claus_steady_state"
    lhs_files = sorted(lhs_dir.glob("lhs_generated_dynamic_ss_data*.xlsx"))
    if {p.name for p in lhs_files} != LHS_NAMES:
        raise ValueError("The LHS files differ from the expected 11-file source set")
    if {Path(name.replace("\\", "/")).name for name in protocol["input_files"]} != LHS_NAMES:
        raise ValueError("The ANN report does not use the expected 11 LHS files")
    source_by_name = {Path(name.replace("\\", "/")).name: digest
                      for name, digest in protocol["source_sha256"].items()}
    for path in lhs_files:
        if source_by_name.get(path.name, "").lower() != sha(path):
            raise ValueError(f"LHS source differs from ANN training protocol: {path}")

    sources = [MANIFEST, Path(__file__).resolve(), ann_report, *artifacts.values(), *lhs_files]
    sources += [ROOT / "data/Claus_dynamic" / name for name in sorted(DYNAMIC_NAMES)]
    sources += sorted((ROOT / "src").rglob("*.py"))
    sources += [ROOT / (module.replace(".", "/") + ".py") for module in set(MODULES.values())]
    sources.append(ROOT / "scripts/training/dynamic_raw_split.py")
    selected = []
    for job in jobs:
        s, family, alias = job["seed"], job["family"], job["alias"]
        if s not in (42, 43, 44) or job["module"] != MODULES.get(family):
            raise ValueError(f"Unexpected model identity: {job}")
        config = ROOT / job["config"]
        if not within(config, MANIFEST.parent) or not config.is_file() or sha(config) != job["config_sha256"]:
            raise ValueError(f"Config is missing or does not match manifest: {config}")
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        data, train = cfg["data"], cfg["training"]
        exp = job["exp_name"]
        if not exp.startswith(PREFIX) or cfg["exp_name"] != exp or train["seed"] != s:
            raise ValueError(f"Unexpected experiment name or seed: {exp}")
        if (set(data["training_files"]) != DYNAMIC_NAMES or data["path"] != "./data/Claus_dynamic/"
                or not data.get("split_before_downsample")
                or not math.isclose(float(data["test_data_split"]), 0.1)
                or not math.isclose(float(data["valid_data_split"]), 1 / 9)
                or train.get("scaler_fit_scope") != "train_only" or train.get("use_pretrained", False)):
            raise ValueError(f"Unexpected full-series split or scaler configuration: {config}")
        checkpoint = ROOT / job["checkpoint"]
        result = ROOT / job["results"]
        if (checkpoint.resolve() != (ROOT / "saved_models" / (exp + ".pth")).resolve()
                or result.resolve() != (RESULT_ROOT / f"seed{s}" / exp).resolve()
                or (ROOT / cfg["output"]["results_dir"] / exp).resolve() != result.resolve()
                or not within(result, RESULT_ROOT)):
            raise ValueError(f"Unexpected artifact location: {exp}")
        args = job.get("cli_args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"Invalid CLI arguments: {exp}")
        if family == "difficult_gradnorm":
            if ("--no-ss" not in args or "--use-config-steady-state-source" not in args
                    or "--target-ratio" not in args or "--max-scale" not in args):
                raise ValueError(f"Missing explicit DG training arguments: {exp}")
            if (not math.isclose(float(args[args.index("--target-ratio") + 1]), float(train["dg_target_ratio"]))
                    or not math.isclose(float(args[args.index("--max-scale") + 1]), float(train["dg_max_scale"]))):
                raise ValueError(f"DG CLI and config disagree: {exp}")
        elif args:
            raise ValueError(f"Unexpected extra CLI arguments: {exp}")
        sources.append(config)
        if s == seed:
            if checkpoint.exists() or result.exists():
                raise FileExistsError(f"Refusing to overwrite an existing artifact: {exp}")
            selected.append(job)
    if len({j["alias"] for j in selected}) != 21:
        raise ValueError(f"Expected 21 distinct aliases for seed {seed}")
    if (RESULT_ROOT / f"seed{seed}" / "_batch_run.json").exists():
        raise FileExistsError(f"A batch record already exists for seed {seed}; inspect it before rerunning")
    hashes = {str(path.resolve()): sha(path) for path in sorted(set(sources))}
    reference = RESULT_ROOT / "_shared_source_reference.json"
    if reference.exists() and json.loads(reference.read_text(encoding="utf-8"))["sources_sha256"] != hashes:
        raise ValueError("Sources differ from the existing shared experiment reference")
    return selected, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    jobs, sources = prepare(args.seed)
    if args.check_only:
        print(f"Preflight passed: seed {args.seed}, {len(jobs)} jobs; no outputs created.")
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no training started")
    seed_dir = RESULT_ROOT / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    lock = seed_dir / "_batch.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        jobs, sources = prepare(args.seed)
        reference = RESULT_ROOT / "_shared_source_reference.json"
        publish_reference(reference, sources)
        record = {"seed": args.seed, "status": "running", "started_utc": now(),
                  "sources_sha256": sources, "completed": [], "active": None}
        record_path = seed_dir / "_batch_run.json"
        write_json(record_path, record)
        try:
            for index, job in enumerate(jobs, 1):
                verify(sources)
                checkpoint, result = ROOT / job["checkpoint"], ROOT / job["results"]
                if checkpoint.exists() or result.exists():
                    raise FileExistsError(f"Output appeared during batch: {job['exp_name']}")
                record["active"] = job["exp_name"]
                write_json(record_path, record)
                print(f"[{index}/{len(jobs)}] seed={args.seed} {job['family']} {job['alias']}", flush=True)
                command = [sys.executable, "-u", "-m", job["module"], "--config", str(ROOT / job["config"])]
                if job["family"] == "difficult_gradnorm":
                    command.extend(["--experiment-suffix", ""])
                command.extend(job.get("cli_args", []))
                subprocess.run(command, cwd=ROOT, check=True)
                required = [checkpoint, result / "zscore_mean.csv", result / "zscore_std.csv"]
                if not all(path.is_file() for path in required):
                    raise FileNotFoundError(f"Missing model or scaler output: {job['exp_name']}")
                weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if not isinstance(weights, dict) or not weights or any(
                    isinstance(v, torch.Tensor) and (v.is_floating_point() or v.is_complex())
                    and not torch.isfinite(v).all().item() for v in weights.values()
                ):
                    raise ValueError(f"Invalid or non-finite checkpoint: {checkpoint}")
                verify(sources)
                record["completed"].append({"job": job, "outputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in required}})
                record["active"] = None
                write_json(record_path, record)
            record["status"] = "complete"
        except BaseException as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_utc"] = now()
            write_json(record_path, record)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
