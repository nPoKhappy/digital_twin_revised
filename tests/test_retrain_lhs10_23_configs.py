import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

import yaml

from src.experiment_paths import experiment_results_dir


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
EXPANDED_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
RETIRED_DG = {'dg_ratio0p0025', 'dg_ratio0p005', 'dg_ratio0p01', 'dg_ratio0p02'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RetrainConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = json.loads(MANIFEST.read_text(encoding='utf8'))

    def test_family_seed_counts_and_unique_names(self):
        self.assertEqual(len(self.jobs), 63)
        expected = {(seed, family): count for seed in (42, 43, 44)
                    for family, count in [('no_gain', 1), ('gain', 5), ('ss_only', 5), ('gain_ss', 5), ('difficult_gradnorm', 5)]}
        self.assertEqual(Counter((row['seed'], row['family']) for row in self.jobs), expected)
        self.assertEqual(len({row['exp_name'] for row in self.jobs}), 63)
        self.assertEqual(len({row['checkpoint'] for row in self.jobs}), 63)
        self.assertEqual(len({row['results'] for row in self.jobs}), 63)

    def test_configs_preserve_existing_hyperparameters(self):
        for row in self.jobs:
            with self.subTest(config=row['config']):
                source = ROOT / row['source_config']
                path = ROOT / row['config']
                self.assertEqual(sha256(source), row['source_sha256'])
                # The historical manifest is hash-bound to completed batches.
                # Four old DG ratios were intentionally retired for each seed;
                # their manifest entries remain as historical provenance.
                if row['family'] == 'difficult_gradnorm' and row['alias'] in RETIRED_DG:
                    self.assertFalse(path.exists(), f'Retired config unexpectedly exists: {path}')
                    continue
                self.assertEqual(sha256(path), row['config_sha256'])
                original = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
                current = yaml.safe_load(path.read_text(encoding='utf8'))
                self.assertEqual(current['exp_name'], row['exp_name'])
                self.assertEqual(current['training']['seed'], row['seed'])
                self.assertEqual(current['training']['scaler_fit_scope'], 'train_only')
                self.assertEqual(experiment_results_dir(current), ROOT / row['results'])
                self.assertEqual(row['checkpoint'], f"saved_models/{row['exp_name']}.pth")
                current['exp_name'] = original['exp_name']
                current['training']['seed'] = original['training']['seed']
                current['training'].pop('scaler_fit_scope')
                if row['family'] == 'difficult_gradnorm':
                    self.assertEqual(current['training'].pop('steady_state_source_path'), './data/Claus_steady_state/lhs_generated_dynamic_ss_data*.xlsx')
                current['output']['results_dir'] = original['output']['results_dir']
                self.assertEqual(current, original)

    def test_ss_families_use_ss_loss_trainer(self):
        for row in self.jobs:
            with self.subTest(family=row['family']):
                expected = ('scripts.training.train_pgin_stepwise_gain_ss_loss'
                            if row['family'] in ('ss_only', 'gain_ss')
                            else 'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm'
                            if row['family'] == 'difficult_gradnorm'
                            else 'scripts.training.train_pgin_stepwise_gain')
                self.assertEqual(row['module'], expected)

    def test_difficult_gradnorm_cli_controls_the_ratio_and_data(self):
        expected = {'dg_ratio0p005': '0.005', 'dg_ratio0p01': '0.01',
                    'dg_ratio0p015': '0.015', 'dg_ratio0p02': '0.02',
                    'dg_ratio0p0025': '0.0025'}
        for row in self.jobs:
            if row['family'] == 'difficult_gradnorm':
                self.assertEqual(row['cli_args'], ['--no-ss', '--target-ratio', expected[row['alias']],
                                                   '--max-scale', '50', '--use-config-steady-state-source'])
                self.assertTrue(row['exp_name'].endswith('_difficult_group_gradnorm_no_ss'))

    def test_expanded_difficult_gradnorm_configs_and_cli(self):
        jobs = json.loads(EXPANDED_MANIFEST.read_text(encoding='utf8'))
        ratios = {'0.15', '1.5', '15', '150'}
        self.assertEqual(len(jobs), 12)
        self.assertEqual({(row['seed'], row['target_ratio']) for row in jobs},
                         {(seed, ratio) for seed in (42, 43, 44) for ratio in ratios})
        self.assertEqual(len({row['exp_name'] for row in jobs}), 12)
        self.assertEqual(len({row['checkpoint'] for row in jobs}), 12)
        self.assertEqual(len({row['results'] for row in jobs}), 12)
        for row in jobs:
            with self.subTest(config=row['config']):
                path = ROOT / row['config']
                template = ROOT / row['template_config']
                self.assertEqual(sha256(path), row['config_sha256'])
                self.assertEqual(sha256(template), row['template_sha256'])
                config = yaml.safe_load(path.read_text(encoding='utf8'))
                original = yaml.safe_load(template.read_text(encoding='utf8'))
                self.assertEqual(config['exp_name'], row['exp_name'])
                self.assertEqual(config['training']['seed'], row['seed'])
                self.assertEqual(config['training']['scaler_fit_scope'], 'train_only')
                self.assertEqual(config['training']['ss_loss_weight'], 0)
                self.assertEqual(row['family'], 'difficult_gradnorm')
                self.assertEqual(row['module'], 'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm')
                self.assertEqual(row['max_scale'], '10000000')
                self.assertEqual(row['cli_args'], ['--no-ss', '--target-ratio', row['target_ratio'],
                                                   '--max-scale', row['max_scale'],
                                                   '--use-config-steady-state-source'])
                self.assertEqual(experiment_results_dir(config), ROOT / row['results'])
                self.assertEqual(row['checkpoint'], f"saved_models/{row['exp_name']}.pth")
                config['exp_name'] = original['exp_name']
                self.assertEqual(config, original)


if __name__ == '__main__':
    unittest.main()
