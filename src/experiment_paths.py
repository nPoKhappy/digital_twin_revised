"""Shared locations for experiment results, matching configs/<group>/<family>."""
from pathlib import Path
from collections.abc import Mapping
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_results_base(exp_name):
    name = str(exp_name).lower()
    if 'tabular_mlp' in name:
        return PROJECT_ROOT / 'results/steady_state'
    if 'difficult' in name or 'gradnorm' in name:
        family = 'difficult_gradnorm'
    elif 'no_gain' in name:
        family = 'no_gain'
    elif re.search(r'ss[^_]*_only', name):
        family = 'ss_only'
    elif re.search(r'gain[^_]*_ss[^_]+', name):
        family = 'gain_ss'
    elif 'gain' in name:
        family = 'gain'
    else:
        return PROJECT_ROOT / 'results/general'
    return PROJECT_ROOT / 'results/pgin' / family


def results_base(config):
    configured = config.get('output', {}).get('results_dir')
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return default_results_base(config['exp_name'])


def experiment_results_dir(config_or_name):
    if isinstance(config_or_name, Mapping):
        return results_base(config_or_name) / config_or_name['exp_name']
    return default_results_base(config_or_name) / str(config_or_name)


def teacher_results_dir():
    return experiment_results_dir('Tabular_MLP_New')
