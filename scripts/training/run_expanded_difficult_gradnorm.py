"""Retrain the five DG ratios directly at their existing artifact paths.

The low/high commands retain their original form. Per-ratio caps are stored in
the original DG YAMLs and manifests. Check-only does not modify artifacts.
Actual training clears the selected old checkpoint and result directory first.
Existing validation and test results still describe the previous models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
EXPANDED_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
TRAINER = ROOT / 'scripts/training/train_pgin_stepwise_gain_difficult_group_gradnorm.py'
TRAINER_MODULE = 'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm'
RATIOS = ('0.015', '0.15', '1.5', '15', '150')
GROUPS = {'low': ('0.015', '0.15', '1.5'), 'high': ('15', '150')}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(sources: dict[str, str]) -> None:
    for name, expected in sources.items():
        path = Path(name)
        if not path.is_file() or sha(path).lower() != expected.lower():
            raise ValueError(f'Protected source changed or is missing: {path}')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def load_jobs(seed: int, group: str | None) -> list[dict]:
    original = json.loads(ORIGINAL_MANIFEST.read_text(encoding='utf-8'))
    expanded = json.loads(EXPANDED_MANIFEST.read_text(encoding='utf-8'))
    baseline = [j for j in original if j['seed'] == seed and j['alias'] == 'dg_ratio0p015']
    extended = [j for j in expanded if j['seed'] == seed]
    if len(baseline) != 1 or len(extended) != 4:
        raise ValueError(f'Unexpected number of DG jobs for seed {seed}')
    baseline[0]['target_ratio'] = '0.015'
    jobs = baseline + extended
    if tuple(j['target_ratio'] for j in jobs) != RATIOS:
        raise ValueError('Unexpected target-ratio ordering')
    for job in jobs:
        cfg = yaml.safe_load((ROOT / job['config']).read_text(encoding='utf-8'))
        rho = str(cfg['training']['dg_target_ratio'])
        cap = cfg['training']['dg_max_scale']
        if rho != job['target_ratio'] or not math.isfinite(float(cap)) or float(cap) < 1:
            raise ValueError(f'Invalid DG ratio/cap in {job["config"]}')
        cap = str(cap)
        expected_cli = ['--no-ss', '--target-ratio', rho, '--max-scale', cap, '--use-config-steady-state-source']
        if job['cli_args'] != expected_cli or ('max_scale' in job and job['max_scale'] != cap):
            raise ValueError(f'DG manifest and config differ: {job["alias"]}')
        job['max_scale'] = cap
    return [j for j in jobs if not group or j['target_ratio'] in GROUPS[group]]


def safe_path(relative: str, expected_parent: Path) -> Path:
    path = ROOT / relative
    resolved = path.resolve()
    parent = expected_parent.resolve()
    if (not parent.is_relative_to(ROOT.resolve()) or not resolved.is_relative_to(parent)
            or resolved == parent):
        raise ValueError(f'Unexpected output path: {path}')
    for candidate in (path, *path.parents):
        if candidate == ROOT.parent:
            break
        if candidate.is_symlink() or (candidate.exists() and getattr(candidate.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError(f'Reparse point in output path: {candidate}')
    return path


def prepare(seed: int, group: str | None) -> dict:
    jobs = load_jobs(seed, group)
    batch = ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}/_batch_run.json'
    original = json.loads(batch.read_text(encoding='utf-8-sig'))
    if original.get('status') != 'complete' or original.get('seed') != seed:
        raise ValueError('Original seed training batch is not complete')
    sources = original['teacher_and_source_sha256']
    verify(sources)
    actual = {p.name for p in (ROOT / 'data/Claus_steady_state').glob('lhs_generated_dynamic_ss_data*.xlsx')}
    expected = {f'lhs_generated_dynamic_ss_data_{i}.xlsx' for i in (10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23)}
    if actual != expected:
        raise ValueError('Steady-state source set differs from the shared ANN training sources')
    protected = {str(p): sha(p) for p in (ORIGINAL_MANIFEST, EXPANDED_MANIFEST, TRAINER, Path(__file__).resolve(), batch)}
    for job in jobs:
        config = ROOT / job['config']
        cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
        if (sha(config) != job['config_sha256'] or cfg['training']['seed'] != seed
                or cfg['training']['scaler_fit_scope'] != 'train_only'
                or cfg['exp_name'] != job['exp_name'] or job['module'] != TRAINER_MODULE
                or cfg['training'].get('use_pretrained', False)):
            raise ValueError(f'Unexpected training configuration: {job["alias"]}')
        checkpoint = safe_path(job['checkpoint'], ROOT / 'saved_models')
        result = safe_path(job['results'], ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}')
        if (checkpoint.name != job['exp_name'] + '.pth' or result.name != job['exp_name']
                or checkpoint.parent.resolve() != (ROOT / 'saved_models').resolve()
                or result.parent.resolve() != (ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}').resolve()
                or (ROOT / cfg['output']['results_dir'] / cfg['exp_name']).resolve() != result.resolve()):
            raise ValueError(f'Unexpected artifact names: {job["alias"]}')
        if result.exists():
            tree_sha(result)
        protected[str(config)] = sha(config)
    print(f'Seed {seed}: ' + ', '.join(f'{j["target_ratio"]} -> {j["max_scale"]}' for j in jobs), flush=True)
    return {'seed': seed, 'group': group, 'jobs': jobs, 'sources': sources, 'protected': protected}


def tree_sha(folder: Path) -> dict[str, str]:
    hashes = {}
    for path in folder.rglob('*'):
        if (path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400
                or not path.resolve().is_relative_to(folder.resolve())):
            raise ValueError(f'Unexpected result path: {path}')
        if path.is_file():
            hashes[str(path.relative_to(folder))] = sha(path)
    return hashes


def run_training(command: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    nonfinite = re.compile(r'(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])', re.IGNORECASE)
    found = False
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1, env=env)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                if nonfinite.search(line):
                    found = True
            result = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    if result:
        raise subprocess.CalledProcessError(result, command)
    if found:
        raise ValueError(f'Non-finite value in training log: {log_path}')


def validate_training(checkpoint: Path, result: Path) -> None:
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Checkpoint is not a nonempty state dictionary')
    for name, value in weights.items():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all().item():
                raise ValueError(f'Non-finite checkpoint weights: {name}')
    checks = {
        'gain_group_gradnorm_history.csv': ('base_output_grad_norm', 'raw_group_output_grad_norm',
                                             'target_output_grad_norm', 'scale', 'estimated_scaled_output_grad_norm'),
        'ss_loss_history.csv': ('ss_loss', 'ss_weight', 'weighted_ss_loss'),
    }
    for name, columns in checks.items():
        with (result / name).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'Empty training history: {name}')
        for row in rows:
            for column in columns:
                if not math.isfinite(float(row[column])):
                    raise ValueError(f'Non-finite {column} in {name}')


def clear_old_artifacts(job: dict, seed: int) -> None:
    """Clear only the explicitly selected DG job, once its lock is owned."""
    if seed not in (42, 43, 44) or job['alias'] not in {'dg_ratio0p015', 'dg_ratio0p15', 'dg_ratio1p5', 'dg_ratio15', 'dg_ratio150'}:
        raise ValueError('Unexpected DG job identity')
    expected = f'lhs10_23_{job["alias"]}_seed{seed}_difficult_group_gradnorm_no_ss'
    if job['exp_name'] != expected:
        raise ValueError('Unexpected experiment name')
    result = safe_path(job['results'], ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}')
    checkpoint = safe_path(job['checkpoint'], ROOT / 'saved_models')
    if (result.name != expected or checkpoint.name != expected + '.pth'
            or result.parent.resolve() != (ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}').resolve()
            or checkpoint.parent.resolve() != (ROOT / 'saved_models').resolve()):
        raise ValueError('Unexpected artifact names')
    lock = safe_path(f'results/pgin/retrain_lhs10_23/seed{seed}/_dg_expanded_job_locks/{job["alias"]}.lock',
                     ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}')
    if not lock.is_file() or lock.read_text(encoding='utf-8').strip() != f'pid={os.getpid()}':
        raise ValueError('DG lock is not owned by this runner')
    if result.exists():
        if not result.is_dir():
            raise ValueError('DG results path must be a directory')
        tree_sha(result)
    if checkpoint.exists() and not checkpoint.is_file():
        raise ValueError('DG checkpoint path must be a file')
    if result.exists():
        shutil.rmtree(result)
    if checkpoint.exists():
        checkpoint.unlink()


def run_job(prepared: dict, job: dict, run_id: str) -> None:
    seed = prepared['seed']
    verify(prepared['sources']); verify(prepared['protected'])
    result = safe_path(job['results'], ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}')
    checkpoint = safe_path(job['checkpoint'], ROOT / 'saved_models')
    journal = ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}/_dg_direct_retrain/{run_id}'
    journal.mkdir(parents=True, exist_ok=True)
    record_path = journal / f'{job["alias"]}.json'
    if record_path.exists():
        raise FileExistsError(f'Run record exists: {record_path}')
    record = {'status': 'preparing', 'started_utc': now(), 'seed': seed, 'job': job,
              'formula': 'min(max_scale, target_output_grad_norm/raw_group_output_grad_norm)',
              'training_mode': 'from_scratch_direct_overwrite', 'source_sha256': prepared['sources'],
              'protected_sha256': prepared['protected'],
              'previous_checkpoint_sha256': sha(checkpoint) if checkpoint.is_file() else None,
              'previous_results_sha256': tree_sha(result) if result.is_dir() else None,
              'note': 'Existing validation and test outputs describe previous checkpoints. If interrupted, outputs may be incomplete; previous tracked files can be recovered from Git.'}
    save(record_path, record)
    try:
        clear_old_artifacts(job, seed)
        record['status'] = 'training'; save(record_path, record)
        print(f'[Train] seed={seed}, rho={job["target_ratio"]}, max-scale={job["max_scale"]}', flush=True)
        command = [sys.executable, '-u', '-m', TRAINER_MODULE, '--config', str(ROOT / job['config']),
                   '--experiment-suffix', '', *job['cli_args']]
        run_training(command, journal / f'{job["alias"]}_training_console.log')
        required = [checkpoint, result / 'zscore_mean.csv', result / 'zscore_std.csv',
                    result / 'gain_group_gradnorm_history.csv', result / 'ss_loss_history.csv']
        if not all(p.is_file() for p in required):
            raise FileNotFoundError(f'Training did not produce all required files: {job["alias"]}')
        validate_training(checkpoint, result)
        verify(prepared['sources']); verify(prepared['protected'])
        record['new_checkpoint_sha256'] = sha(checkpoint)
        record['new_results_sha256'] = tree_sha(result)
        record['status'] = 'complete'
    except BaseException as exc:
        record['status'] = 'failed'
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['finished_utc'] = now()
        save(record_path, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), help='One seed; omit with --group for all three')
    parser.add_argument('--group', choices=tuple(GROUPS), help='low: 0.015/0.15/1.5; high: 15/150')
    parser.add_argument('--check-only', action='store_true', help='Validate without training or overwriting files')
    args = parser.parse_args()
    if args.seed is None and args.group is None:
        parser.error('Specify --seed, --group, or both')
    os.chdir(ROOT)
    seeds = (args.seed,) if args.seed is not None else (42, 43, 44)
    prepared = [prepare(seed, args.group) for seed in seeds]
    if args.check_only:
        print('Check-only passed; no training or output started.', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    locks = []
    try:
        for batch in prepared:
            folder = ROOT / f'results/pgin/retrain_lhs10_23/seed{batch["seed"]}/_dg_expanded_job_locks'
            folder.mkdir(exist_ok=True)
            for job in batch['jobs']:
                lock = folder / (job['alias'] + '.lock')
                with lock.open('x', encoding='utf-8') as stream:
                    stream.write(f'pid={os.getpid()}\n')
                locks.append(lock)
        # Verify again under locks to close the preflight race.
        prepared = [prepare(seed, args.group) for seed in seeds]
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        for batch in prepared:
            for job in batch['jobs']:
                run_job(batch, job, run_id)
    finally:
        for lock in locks:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
