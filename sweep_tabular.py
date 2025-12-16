# sweep_tabular.py - Hyperparameter sweep for tabular MLP models (穩態數據) 找出最佳超參數組合
import os
import copy
import yaml
import argparse
import itertools
import json
import random

# We'll import and call main() from train_tabular to avoid code duplication
import train_tabular


def _load_done_set(out_dir: str):
    done = set()
    res_path = os.path.join(out_dir, 'sweep_results.json')
    if os.path.exists(res_path):
        try:
            with open(res_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict) and 'run' in item:
                    done.add(item['run'])
        except Exception:
            pass
    return done


def _all_combos(keys, values_list):
    for values in itertools.product(*values_list):
        yield dict(zip(keys, values))


def run_sweep(config_path: str, strategy: str = 'grid', max_runs: int = 24, seed: int = 42):
    with open(config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    # Define a compact grid suitable for ~40k rows
    grids = {
        'model.hidden_dims': [
            [64, 64],
            [128, 64],
            [256, 128],
        ],
        'model.dropout': [0.0, 0.1, 0.2],
        'model.activation': ['relu', 'gelu'],
        'training.learning_rate': [1e-3, 3e-4],
        'training.batch_size': [512, 1024, 2048],
        'training.weight_decay': [0.0, 1e-4, 1e-3],
    }

    keys = list(grids.keys())
    values_list = list(grids.values())

    sweep_id = base_cfg['exp_name'] + '_sweep'
    out_dir = os.path.join('results', sweep_id)
    os.makedirs(out_dir, exist_ok=True)

    # Resume support: load already completed runs
    done = _load_done_set(out_dir)

    # Build candidate combinations
    all_items = list(_all_combos(keys, values_list))
    total_all = len(all_items)

    if strategy == 'random':
        random.seed(seed)
        random.shuffle(all_items)
    # else grid: keep original order

    # Limit to max_runs after excluding completed ones
    selected = []
    for combo in all_items:
        # Construct run_name preview for skip check
        cfg = copy.deepcopy(base_cfg)
        label_parts = []
        for k, v in combo.items():
            d = cfg
            parts = k.split('.')
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
            label_parts.append(f"{parts[-1]}={v}")
        run_name = base_cfg['exp_name'] + '__' + '__'.join(label_parts).replace(' ', '')
        if run_name in done:
            continue
        selected.append(combo)
        if len(selected) >= max_runs:
            break

    print(f"Total combos: {total_all} | Completed: {len(done)} | To run now: {len(selected)} (strategy={strategy}, max_runs={max_runs})")

    results_path = os.path.join(out_dir, 'sweep_results.json')
    # Load existing results for appending
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            results = []
    else:
        results = []

    for combo in selected:
        cfg = copy.deepcopy(base_cfg)
        label_parts = []
        for k, v in combo.items():
            d = cfg
            parts = k.split('.')
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
            label_parts.append(f"{parts[-1]}={v}")

        run_name = base_cfg['exp_name'] + '__' + '__'.join(label_parts).replace(' ', '')
        cfg['exp_name'] = run_name

        # Shorten epochs/patience for faster sweep
        cfg['training']['epochs'] = min(cfg['training'].get('epochs', 200), 200)
        cfg['training']['patience'] = min(cfg['training'].get('patience', 20), 20)

        tmp_cfg_path = os.path.join(out_dir, run_name + '.yaml')
        with open(tmp_cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        print(f"\n=== Running: {run_name} ===")
        best_val = train_tabular.main(tmp_cfg_path)
        results.append({'run': run_name, 'best_val_l1': float(best_val)})

        # Persist after each run
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Save CSV ranking
    try:
        import pandas as pd
        pd.DataFrame(results).sort_values('best_val_l1').to_csv(os.path.join(out_dir, 'sweep_results.csv'), index=False)
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Base YAML for the tabular run (e.g., configs/tabular_mlp_claus.yaml)')
    parser.add_argument('--strategy', default='grid', choices=['grid', 'random'])
    parser.add_argument('--max-runs', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_sweep(args.config, strategy=args.strategy, max_runs=args.max_runs, seed=args.seed)
