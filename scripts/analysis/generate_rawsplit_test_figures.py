"""Generate interactive and JTICE selected-test figures for rawsplit runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from scripts.prediction import plot_retrained_15_seed_test as interactive
from scripts.analysis import plot_rawsplit_selected_test as journal

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_rawsplit'


def main():
    selected = json.loads((SOURCE / 'selected_models.json').read_text(encoding='utf8'))['selected_models']
    parameters = {}
    aliases = {}
    for row in selected:
        family, alias = row['family'], row['alias']
        train = yaml.safe_load((ROOT / row['config']).read_text(encoding='utf8'))['training']
        relevant = tuple(str(train.get(k)) for k in ('gain_loss_weight', 'ss_loss_weight', 'dg_target_ratio'))
        if family in parameters and (parameters[family] != relevant or aliases[family] != alias):
            raise ValueError(f'Inconsistent selected configurations for {family}')
        parameters[family], aliases[family] = relevant, alias
    num = lambda x: f'{float(x):g}'
    labels = {'no_gain': 'No-Gain',
              'gain': 'Gain = ' + num(parameters['gain'][0]),
              'ss_only': 'SS-only = ' + num(parameters['ss_only'][1]),
              'gain_ss': 'Gain + SS = ' + num(parameters['gain_ss'][0]) + ' / ' + num(parameters['gain_ss'][1]),
              'difficult_gradnorm': 'Difficult-GradNorm, rho = ' + num(parameters['difficult_gradnorm'][2])}
    def actual_label(family, alias):
        if aliases[family] != alias:
            raise ValueError(f'Unexpected alias for {family}: {alias}')
        return labels[family]
    interactive.label = actual_label
    previous = sys.argv
    try:
        sys.argv = [previous[0], '--input', str(SOURCE)]
        interactive.main()
    finally:
        sys.argv = previous
    journal.main()


if __name__ == '__main__':
    main()
