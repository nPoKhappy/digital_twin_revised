"""Run ordinary per-model prediction for a completed full-series split job.

This entry point checks the completed training record, checkpoint, and both
training scalers before invoking the existing predictor. It does not perform
the multi-model validation, test aggregation, or KCI evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'retrain_lhs10_23_full_80_10_10'
MANIFEST = ROOT / 'configs/pgin' / EXPERIMENT / 'config_manifest.json'
RESULT_ROOT = ROOT / 'results/pgin' / EXPERIMENT
PREFIX = 'lhs10_23_full_80_10_10_'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve_completed(seed: int, alias: str) -> Path:
    """Return the verified configuration for a single completed job."""
    jobs = json.loads(MANIFEST.read_text(encoding='utf-8'))
    matches = [job for job in jobs if job['seed'] == seed and job['alias'] == alias]
    if len(matches) != 1:
        raise ValueError(f'Expected one manifest job for seed={seed}, alias={alias}')
    job = matches[0]
    name = job['exp_name']
    if not name.startswith(PREFIX) or '/' in name or '\\' in name:
        raise ValueError(f'Unexpected experiment name: {name}')
    config = ROOT / job['config']
    config_parent = (ROOT / 'configs/pgin' / EXPERIMENT).resolve()
    if not config.resolve().is_relative_to(config_parent) or not config.is_file() or sha(config) != job['config_sha256']:
        raise ValueError(f'Configuration missing or differs from manifest: {config}')
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    if cfg['exp_name'] != name or cfg['training']['seed'] != seed:
        raise ValueError('Configuration experiment identity does not match manifest')
    checkpoint = ROOT / job['checkpoint']
    result = ROOT / job['results']
    expected_result = RESULT_ROOT / f'seed{seed}' / name
    if (checkpoint.resolve() != (ROOT / 'saved_models' / (name + '.pth')).resolve()
            or result.resolve() != expected_result.resolve()
            or (ROOT / cfg['output']['results_dir'] / name).resolve() != result.resolve()):
        raise ValueError('Checkpoint or scaler directory does not match experiment identity')

    record_path = RESULT_ROOT / f'seed{seed}' / '_batch_run.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record.get('seed') != seed:
        raise ValueError('Batch record seed does not match manifest')
    completed = [entry for entry in record.get('completed', []) if entry.get('job', {}).get('exp_name') == name]
    if len(completed) != 1 or completed[0]['job'] != job:
        raise ValueError(f'No matching completed training job: {name}')
    sources = record.get('sources_sha256', {})
    for path in (MANIFEST, config):
        if sources.get(str(path.resolve()), '').lower() != sha(path):
            raise ValueError(f'Training source hash does not match: {path}')
    outputs = completed[0].get('outputs_sha256', {})
    required = (checkpoint, result / 'zscore_mean.csv', result / 'zscore_std.csv')
    for path in required:
        key = str(path.relative_to(ROOT))
        if not path.is_file() or outputs.get(key, '').lower() != sha(path):
            raise ValueError(f'Completed checkpoint or training scaler missing or changed: {path}')
    # predict.main falls back to fitting a scaler on a full input when either
    # scaler is missing. Both must exist and match the saved training hashes.
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--alias', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = resolve_completed(args.seed, args.alias)
    from scripts.prediction import predict
    predict.main(str(config))


if __name__ == '__main__':
    main()
