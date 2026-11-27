"""Evaluate a frozen selected-test seed in isolated scratch and publish atomically.

The main evaluator owns aggregation. A worker never changes the frozen plan,
protocol or a destination already claimed by the main evaluator. Completed
directories include their protocol-bound marker before atomic publication.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from scripts.prediction import compare_rawsplit_selected_test as selected_test
from scripts.prediction import compare_retrained_mean_tradeoff_test as evaluation

ROOT = selected_test.ROOT
MASTER = selected_test.OUT


def publish_completed(folder: Path, master: Path) -> bool:
    if not (folder / 'complete.json').is_file():
        raise ValueError(f'Cannot publish incomplete model: {folder}')
    dest_root = master / 'model_data'
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / folder.name
    if folder.parent.name != 'model_data' or not dest.resolve().is_relative_to(master.resolve()):
        raise ValueError(f'Unexpected publication path: {folder}')
    try:
        os.rename(folder, dest)
    except OSError:
        if not dest.exists():
            raise
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = MASTER / 'validation_selection_plan.json'
    all_models = selected_test.select_models(plan)
    configs, contract, protocol = selected_test.preflight(all_models, plan)
    if not (MASTER / 'run_protocol.json').is_file():
        raise ValueError('Main evaluator must freeze the complete run protocol first')
    subset = list(reversed([r for r in all_models if r['seed'] == args.seed]))
    if len(subset) != 5:
        raise ValueError('Expected exactly five selected models for the seed')
    scratch = MASTER / f'_parallel_worker_seed{args.seed}'
    scratch.mkdir(parents=True, exist_ok=False)
    for name in ('run_protocol.json', 'selected_models.json', 'validation_selection_plan.json'):
        shutil.copyfile(MASTER/name, scratch/name)
    if selected_test.sha(scratch/'run_protocol.json') != selected_test.sha(MASTER/'run_protocol.json'):
        raise ValueError('Copied protocol differs')
    evaluation.OUT = scratch
    evaluation.selection_before_test = lambda _: subset
    evaluation.preflight = lambda _selected, _path: (configs, contract, protocol)
    evaluation.summarize = lambda _selected, _protocol: None
    original_write_json = evaluation.write_json
    def write_and_publish(path, value):
        original_write_json(path, value)
        path = Path(path)
        if path.name == 'complete.json':
            result = publish_completed(path.parent, MASTER)
            print(f'Worker seed {args.seed}: {path.parent.name} '
                  + ('published' if result else 'already claimed; preserved in scratch'), flush=True)
    evaluation.write_json = write_and_publish
    evaluation.run(scratch / 'validation_selection_plan.json')


if __name__ == '__main__':
    main()
