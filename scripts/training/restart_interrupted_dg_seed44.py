"""Restart exactly one interrupted seed-44 job from scratch, preserving its partial files.

The prior checkpoints contain model weights only, so this does not resume an epoch.
Two terminals may run --ratio 1.5 and --ratio 150 concurrently.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
SEED_DIR = ROOT / 'results/pgin/retrain_lhs10_23/seed44'
SELECTED = {'1.5': ('low', 'dg_ratio1p5', '0.15'), '150': ('high', 'dg_ratio150', '15')}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    os.replace(temp, path)


def require_dead(pid: int) -> None:
    """OpenProcess distinguishes a nonexistent PID from an inaccessible live PID."""
    if os.name != 'nt':
        raise RuntimeError('This restart procedure is specific to the Windows workspace.')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        kernel.CloseHandle(handle)
        raise RuntimeError(f'Previous batch PID {pid} is still present; do not restart.')
    error = ctypes.get_last_error()
    if error != 87:  # ERROR_INVALID_PARAMETER: PID does not exist
        raise RuntimeError(f'Cannot establish that previous PID {pid} stopped (Windows error {error}).')


def inside(path: Path, parent: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(parent.resolve(strict=True)):
        raise ValueError(f'Path outside expected directory: {path}')
    return resolved


def file_tree(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'Expected a regular result directory: {path}')
    result = {}
    for file in path.rglob('*'):
        if file.is_symlink():
            raise ValueError(f'Unexpected symbolic link: {file}')
        if file.is_file():
            inside(file, path)
            result[str(file.relative_to(path))] = sha(file)
    return result


def check_completed() -> dict[str, str]:
    """Verify all ten completed expanded jobs before touching either incomplete job."""
    hashes = {}
    entries = json.loads(MANIFEST.read_text(encoding='utf8'))
    by_exp = {j['exp_name']: j for j in entries}
    count = 0
    for seed in (42, 43, 44):
        for group in ('low', 'high'):
            record_path = ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}/_dg_expanded_{group}_batch_run.json'
            record = json.loads(record_path.read_text(encoding='utf8'))
            completed = record['completed']
            if (seed in (42, 43) and (record['status'] != 'complete' or len(completed) != 2)) or (seed == 44 and len(completed) != 1):
                raise ValueError(f'Unexpected completed batch state: {record_path}')
            hashes[str(record_path)] = sha(record_path)
            for item in completed:
                job = by_exp[item['exp_name']]
                result_dir = ROOT / job['results']
                pairs = [(ROOT / job['checkpoint'], item['checkpoint_sha256']),
                         (result_dir / 'zscore_mean.csv', item['zscore_mean_sha256']),
                         (result_dir / 'zscore_std.csv', item['zscore_std_sha256'])]
                for path, expected in pairs:
                    if not path.is_file() or sha(path).lower() != expected.lower():
                        raise ValueError(f'Completed experiment changed: {path}')
                    hashes[str(path)] = expected
                count += 1
    if count != 10:
        raise ValueError(f'Expected exactly ten completed models, found {count}')
    return hashes


def preflight(ratio: str) -> dict:
    group, alias, other_ratio = SELECTED[ratio]
    entries = json.loads(MANIFEST.read_text(encoding='utf8'))
    jobs = [j for j in entries if j['seed'] == 44 and j['alias'] == alias]
    if len(jobs) != 1:
        raise ValueError('Missing or duplicate original manifest entry')
    job = jobs[0]
    if job['target_ratio'] != ratio or job['max_scale'] != '10000000' or job['module'] != 'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm':
        raise ValueError('Unexpected original experiment settings')
    expected_cli = ['--no-ss', '--target-ratio', ratio, '--max-scale', '10000000', '--use-config-steady-state-source']
    if job['cli_args'] != expected_cli:
        raise ValueError('Unexpected original trainer arguments')
    config = ROOT / job['config']
    inside(config, ROOT / 'configs/pgin/retrain_lhs10_23/seed44')
    cfg = yaml.safe_load(config.read_text(encoding='utf8'))
    if sha(config) != job['config_sha256'] or cfg['exp_name'] != job['exp_name'] or cfg['training']['seed'] != 44 or cfg['training'].get('use_pretrained', False):
        raise ValueError('Original training config changed or enables pretrained initialization')
    batch_path = SEED_DIR / f'_dg_expanded_{group}_batch_run.json'
    batch = json.loads(batch_path.read_text(encoding='utf8'))
    if batch['status'] != 'running' or batch['active'] != job['exp_name'] or [x['target_ratio'] for x in batch['completed']] != [other_ratio]:
        raise ValueError('Interrupted batch state changed; inspect manually')
    for mapping in (batch['teacher_and_source_sha256'], batch['expanded_source_sha256']):
        for name, digest in mapping.items():
            path = Path(name)
            if not path.is_file() or sha(path).lower() != digest.lower():
                raise ValueError(f'Protected training input changed: {path}')
    lock_path = SEED_DIR / '_dg_expanded_job_locks' / f'{alias}.lock'
    inside(lock_path, SEED_DIR / '_dg_expanded_job_locks')
    match = re.fullmatch(r'pid=(\d+)\s*', lock_path.read_text(encoding='utf8'))
    if not match:
        raise ValueError(f'Cannot identify old batch PID: {lock_path}')
    old_pid = int(match.group(1))
    require_dead(old_pid)
    result_dir = ROOT / job['results']
    checkpoint = ROOT / job['checkpoint']
    inside(result_dir, SEED_DIR)
    inside(checkpoint, ROOT / 'saved_models')
    partial_hashes = file_tree(result_dir)
    checkpoint_hash = sha(checkpoint)
    completed_hashes = check_completed()
    archive = SEED_DIR / '_interrupted_restart_archive' / alias
    if archive.exists():
        raise FileExistsError(f'Restart archive already exists: {archive}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing a CPU training run')
    return dict(job=job, group=group, old_pid=old_pid, lock=lock_path, batch_path=batch_path,
                batch_sha256=sha(batch_path), config=config, result_dir=result_dir, checkpoint=checkpoint,
                partial_hashes=partial_hashes, checkpoint_hash=checkpoint_hash,
                completed_hashes=completed_hashes, archive=archive,
                sources=batch['teacher_and_source_sha256'], expanded_sources=batch['expanded_source_sha256'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratio', choices=tuple(SELECTED), required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    task = preflight(args.ratio)
    print(f"Seed 44, rho {args.ratio}; from scratch; previous PID {task['old_pid']} absent; ten completed models verified.", flush=True)
    if args.check_only:
        print('Check-only passed: no files changed and no training started.', flush=True)
        return
    archive = task['archive']
    archive.mkdir(parents=True, exist_ok=False)
    inside(archive, SEED_DIR)
    journal_path = archive / 'restart_record.json'
    journal = dict(status='preparing', started_utc=now(), seed=44, ratio=args.ratio,
                   training_mode='from_scratch_not_resume', previous_pid=task['old_pid'],
                   original_batch=str(task['batch_path']), original_batch_sha256=task['batch_sha256'],
                   original_checkpoint_sha256=task['checkpoint_hash'], original_result_hashes=task['partial_hashes'],
                   completed_model_sha256=task['completed_hashes'], trainer_argv=task['job']['cli_args'])
    save(journal_path, journal)
    lock_acquired = False
    try:
        require_dead(task['old_pid'])
        if sha(task['checkpoint']) != task['checkpoint_hash'] or file_tree(task['result_dir']) != task['partial_hashes'] or sha(task['batch_path']) != task['batch_sha256']:
            raise RuntimeError('Interrupted artifacts changed since preflight')
        task['lock'].rename(archive / 'previous_batch.lock')
        with task['lock'].open('x', encoding='utf8') as stream:
            stream.write(f'pid={os.getpid()}\n')
        lock_acquired = True
        task['checkpoint'].rename(archive / task['checkpoint'].name)
        task['result_dir'].rename(archive / task['result_dir'].name)
        journal['status'] = 'running'
        save(journal_path, journal)
        job = task['job']
        subprocess.run([sys.executable, '-u', '-m', job['module'], '--config', str(task['config']), *job['cli_args']], cwd=ROOT, check=True)
        for path in (task['checkpoint'], task['result_dir'] / 'zscore_mean.csv', task['result_dir'] / 'zscore_std.csv',
                     task['result_dir'] / 'gain_group_gradnorm_history.csv', task['result_dir'] / 'kci_history.csv'):
            if not path.is_file():
                raise FileNotFoundError(f'Finished training artifact missing: {path}')
        for mapping in (task['sources'], task['expanded_sources'], task['completed_hashes']):
            for name, digest in mapping.items():
                path = Path(name)
                if not path.is_file() or sha(path).lower() != digest.lower():
                    raise RuntimeError(f'Protected input or completed model changed during restart: {path}')
        journal['new_checkpoint_sha256'] = sha(task['checkpoint'])
        journal['new_result_hashes'] = file_tree(task['result_dir'])
        journal['status'] = 'complete'
    except BaseException as exc:
        journal['status'] = 'failed'
        journal['error'] = str(exc)
        raise
    finally:
        journal['finished_utc'] = now()
        save(journal_path, journal)
        if lock_acquired:
            task['lock'].unlink(missing_ok=True)
    print(f'Completed seed 44 rho {args.ratio}; restart record: {journal_path}', flush=True)


if __name__ == '__main__':
    main()
