"""Audit updated-test conditional KCI using existing independent NPZ audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import audit_three_seed_magnitude_kci as magnitude
from scripts.analysis import audit_three_seed_sign_kci as sign

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42/updated_validation_best'
PREVIOUS = ROOT / 'results/visualization/gain_pair_difficulty_seed42'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chart', choices=('sign', 'magnitude'), required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if args.chart == 'sign':
        module = sign
        stem = 'conditional_kci_by_teacher_sign'
        grouping = 'teacher_sign_label'
        module.OLD = PREVIOUS / (stem + '.csv')
    else:
        module = magnitude
        stem = 'conditional_kci_by_gain_magnitude'
        grouping = 'quartile'
    # Reuse independent raw-array audits while redirecting their historical
    # SOURCE constant to the validation-selected updated test results.
    module.SOURCE = SOURCE
    sys.argv = [str(Path(module.__file__)), '--directory', str(directory)]
    module.main()

    provenance = json.loads((directory / (stem + '_provenance.json')).read_text(encoding='utf8'))
    for key in ('input_sha256', 'inputs_sha256'):
        for path, expected in provenance.get(key, {}).items():
            if sha(ROOT / path).lower() != expected.lower():
                raise AssertionError(f'Changed source: {path}')
    for key in ('output_sha256', 'outputs_sha256'):
        for path, expected in provenance.get(key, {}).items():
            if sha(directory / path).lower() != expected.lower():
                raise AssertionError(f'Changed output: {path}')

    # The 12 reused models should retain exactly their previous conditional
    # results, although the new DG family now has rho=1.5.
    old = pd.read_csv(PREVIOUS / (stem + '_per_seed.csv'))
    new = pd.read_csv(directory / (stem + '_per_seed.csv'))
    retained = {'no_gain', 'gain', 'ss_only', 'gain_ss'}
    old = old[old.family.isin(retained)]
    new = new[new.family.isin(retained)]
    keys = ['model', 'family', 'alias', 'seed', 'distribution', 'target', 'mv', grouping]
    joined = old.merge(new, on=keys, suffixes=('_old', '_new'), validate='one_to_one', how='outer', indicator=True)
    expected_count = 12 * 2 * 4 * (2 if args.chart == 'sign' else 4)
    if len(joined) != expected_count or not joined['_merge'].eq('both').all():
        raise AssertionError('Reused-model conditional group coverage changed')
    for column in ('probe_count', 'correct', 'total', 'kci_percent'):
        if not np.allclose(joined[column + '_old'], joined[column + '_new'], rtol=0, atol=1e-12, equal_nan=True):
            raise AssertionError(f'Reused-model values changed: {column}')

    output = {
        'passed': True,
        'updated_test_source': str(SOURCE.relative_to(ROOT)),
        'independent_npz_audit_script': str(Path(module.__file__).resolve().relative_to(ROOT)),
        'independent_npz_audit_script_sha256': sha(Path(module.__file__).resolve()),
        'reused_models': 12,
        'reused_model_conditional_groups_matched': expected_count,
        'updated_provenance_inputs_outputs_checked': True,
        'audit_script_sha256': sha(Path(__file__).resolve()),
    }
    path = directory / f'independent_updated_{args.chart}_audit.json'
    path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
