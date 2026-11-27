"""Recompute sign- and magnitude-conditioned KCI for the rawsplit test run.

The selected models and their gain NPZs must already be fully evaluated and
audited. All 15 models use the same ANN reference and test probes. Statistics
are the three training-seed mean and sample SD, not bootstrap intervals.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from scripts.analysis import plot_jtice_teacher_gain_kci as journal
from scripts.analysis import plot_updated_test_teacher_sign_three_seeds as sign
from scripts.analysis import plot_updated_test_teacher_magnitude_three_seeds as magnitude

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_rawsplit'
OUT = ROOT / 'results/visualization/gain_pair_difficulty_seed42_rawsplit'


def config_label(family: str, train: dict) -> str:
    """Make legend labels from actual hyperparameters, not ambiguous aliases."""
    if family == 'no_gain':
        return 'No-Gain'
    if family == 'gain':
        return f"Gain = {float(train['gain_loss_weight']):g}"
    if family == 'ss_only':
        return f"SS-only = {float(train['ss_loss_weight']):g}"
    if family == 'gain_ss':
        return f"Gain + SS = {float(train['gain_loss_weight']):g} / {float(train['ss_loss_weight']):g}"
    if family == 'difficult_gradnorm':
        return f"Difficult-GradNorm, ρ = {float(train['dg_target_ratio']):g}"
    raise ValueError(f'Unexpected model family: {family}')


def run(source: Path = SOURCE, destination: Path = OUT) -> None:
    """Create a fresh result directory after validating current test artifacts."""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):
        raise ValueError('Source and output must be inside the project')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {destination}')
    if not source.is_dir():
        raise FileNotFoundError(f'Selected test output is not available: {source}')

    selection_path = source / 'selected_models.json'
    selection = json.loads(selection_path.read_text(encoding='utf8'))['selected_models']
    families = ('no_gain', 'gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
    aliases = {}
    labels = {}
    for family in families:
        rows = [row for row in selection if row['family'] == family]
        if len(rows) != 3 or {int(row['seed']) for row in rows} != {42, 43, 44} or len({r['alias'] for r in rows}) != 1:
            raise ValueError(f'Expected one selected setting and three seeds: {family}')
        for row in rows:
            if not str(row['config']).replace('\\', '/').startswith('configs/pgin/retrain_lhs10_23_full_80_10_10/'):
                raise ValueError(f'Selected model is not from the raw-split experiment: {row["model"]}')
            if not Path(str(row['checkpoint']).replace('\\', '/')).name.startswith('lhs10_23_full_80_10_10_'):
                raise ValueError(f'Selected checkpoint is not from the raw-split experiment: {row["model"]}')
        aliases[family] = rows[0]['alias']
        # Read actual numbers from the config: alias "005" means 0.05, not 5.
        values = []
        for row in rows:
            cfg = yaml.safe_load((ROOT / row['config']).read_text(encoding='utf8'))
            values.append(config_label(family, cfg['training']))
        if len(set(values)) != 1:
            raise ValueError(f'Hyperparameters differ across seeds for {family}: {values}')
        labels[family] = values[0]

    # The established computations check the test protocol, selection and all
    # NPZ hashes, reconstruct official KCI counts, and stratify every probe.
    # Intermediate figures/metadata are deliberately kept in a temporary
    # directory: publication files and provenance below describe this run.
    with tempfile.TemporaryDirectory(prefix='rawsplit_kci_', dir=ROOT) as temp:
        temp = Path(temp)
        sign.SOURCE, sign.OUT = source, temp / 'sign'
        magnitude.SOURCE, magnitude.OUT = source, temp / 'magnitude'
        sign.main()
        magnitude.main()
        destination.mkdir(parents=True, exist_ok=False)
        for name in ('conditional_kci_by_teacher_sign.csv', 'conditional_kci_by_teacher_sign_per_seed.csv'):
            shutil.copy2(sign.OUT / name, destination / name)
        for name in ('conditional_kci_by_gain_magnitude.csv', 'conditional_kci_by_gain_magnitude_per_seed.csv',
                     'gain_magnitude_quartile_edges.csv'):
            shutil.copy2(magnitude.OUT / name, destination / name)
        upstream = {
            'sign': json.loads((sign.OUT / 'conditional_kci_by_teacher_sign_provenance.json').read_text(encoding='utf8')),
            'magnitude': json.loads((magnitude.OUT / 'conditional_kci_by_gain_magnitude_provenance.json').read_text(encoding='utf8')),
        }

    # Use the same journal layout as the prior analysis, with aliases read from
    # the new selection instead of hard-coding last run's hyperparameters.
    styles = [
        ('no_gain', '#222222', 's', '--'),
        ('gain', '#0072B2', 'o', '-'),
        ('ss_only', '#009E73', '^', '-.'),
        ('gain_ss', '#D55E00', 'D', ':'),
        ('difficult_gradnorm', '#CC79A7', 'v', (0, (5, 1, 1, 1))),
    ]
    journal.SOURCE = destination
    journal.OUT = destination / 'jtice_figures'
    journal.OUT.mkdir()
    journal.FAMILIES = [(family, aliases[family], labels[family], color, marker, style)
                        for family, color, marker, style in styles]
    frames = {kind: journal.validated(kind) for kind in ('sign', 'magnitude')}
    bounds = pd.concat(list(frames.values()), ignore_index=True)
    lower = (bounds.mean_kci_percent - bounds.sd_kci_percentage_points).dropna()
    upper = (bounds.mean_kci_percent + bounds.sd_kci_percentage_points).dropna()
    journal.Y_LIMITS = (min(-6, float(lower.min()) - 2), max(106, float(upper.max()) + 2))
    for kind, frame in frames.items():
        journal.make_figure(kind, frame)
    journal.write_captions()

    # Convenient root-level copies, retaining the existing filename convention.
    for kind, stem in [('sign', 'teacher_sign'), ('magnitude', 'gain_magnitude')]:
        for suffix in ('png', 'pdf', 'grayscale.png'):
            src = journal.OUT / f'kci_teacher_gain_{kind}_{suffix}' if suffix == 'grayscale.png' else journal.OUT / f'kci_teacher_gain_{kind}.{suffix}'
            name = f'conditional_kci_by_{stem}_{suffix}' if suffix == 'grayscale.png' else f'conditional_kci_by_{stem}.{suffix}'
            shutil.copy2(src, destination / name)

    note = (
        '# Raw-split three-seed ANN-reference KCI\n\n'
        f'Validation-selected model settings: {aliases}. All 15 dynamic models '
        'were evaluated on the raw-split experiment; no previous model predictions '
        'were reused. Each point is the mean of seed 42, 43 and 44, and error bars '
        'show one sample standard deviation (ddof=1), not a bootstrap confidence '
        'interval. The seeds share one ANN and the same test operating points. '
        'Every valid operating point contributes 2 perturbation directions × 18 '
        'prediction steps; ANN |gain| below 1e-5 is excluded, and zero dynamic '
        'gain is incorrect. Empty groups are missing values, not zero KCI.\n\n'
        'The magnitude quartiles are calculated separately per region and gain '
        'pair from normalized ANN |gain| using shared boundaries. See the '
        'quartile-edges CSV for bounds and probe counts. KCI measures agreement '
        'with ANN directions, not independent proof of Aspen physical directions. '
        'Test probes were previously used and are not a fresh blind holdout.\n\n'
        'Rebuild after the raw-split selected test evaluation with '
        '`python -m scripts.analysis.plot_rawsplit_teacher_gain_kci`.\n'
    )
    (destination / 'README.md').write_text(note, encoding='utf8')
    inputs = {}
    for info in upstream.values():
        if not info.get('passed'):
            raise ValueError('Source conditional KCI verification failed')
        inputs.update(info['input_sha256'])
    for path in [Path(__file__), Path(journal.__file__), source / 'validation_selection_plan.json']:
        inputs[str(path.resolve().relative_to(ROOT))] = sign.sha(path)
    files = [p for p in destination.rglob('*') if p.is_file()]
    record = {
        'passed': True,
        'selected_aliases': aliases,
        'model_labels': labels,
        'seeds': [42, 43, 44],
        'source_test': str(source.relative_to(ROOT)),
        'statistic': 'arithmetic mean and sample standard deviation (ddof=1) over three training seeds',
        'reference': 'shared ANN; two perturbation directions and all 18 prediction steps',
        'threshold': sign.THRESHOLD,
        'inputs_sha256': inputs,
        'outputs_sha256': {str(p.relative_to(destination)): sign.sha(p) for p in files},
    }
    (destination / 'provenance.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print(f'Wrote conditional KCI results to {destination}')


if __name__ == '__main__':
    run()
