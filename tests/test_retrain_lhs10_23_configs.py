import hashlib
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import yaml
import torch

from src.experiment_paths import experiment_results_dir
from scripts.training import run_expanded_difficult_gradnorm as runner


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
EXPANDED_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
RETIRED_DG = {'dg_ratio0p0025', 'dg_ratio0p005', 'dg_ratio0p01', 'dg_ratio0p02'}
DG_CAPS = {'0.015': '1000', '0.15': '20000', '1.5': '500000', '15': '10000000', '150': '500000000'}


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
                    self.assertEqual(str(current['training'].pop('dg_target_ratio')), '0.015')
                    self.assertEqual(str(current['training'].pop('dg_max_scale')), DG_CAPS['0.015'])
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
                cap = DG_CAPS['0.015'] if row['alias'] == 'dg_ratio0p015' else '50'
                self.assertEqual(row['cli_args'], ['--no-ss', '--target-ratio', expected[row['alias']],
                                                   '--max-scale', cap, '--use-config-steady-state-source'])
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
                self.assertEqual(row['max_scale'], DG_CAPS[row['target_ratio']])
                self.assertEqual(str(config['training']['dg_target_ratio']), row['target_ratio'])
                self.assertEqual(str(config['training']['dg_max_scale']), DG_CAPS[row['target_ratio']])
                self.assertEqual(row['cli_args'], ['--no-ss', '--target-ratio', row['target_ratio'],
                                                   '--max-scale', row['max_scale'],
                                                   '--use-config-steady-state-source'])
                self.assertEqual(experiment_results_dir(config), ROOT / row['results'])
                self.assertEqual(row['checkpoint'], f"saved_models/{row['exp_name']}.pth")
                config['exp_name'] = original['exp_name']
                for key in ('dg_target_ratio', 'dg_max_scale'):
                    config['training'].pop(key)
                    original['training'].pop(key)
                self.assertEqual(config, original)


class DgCapsRetrainingTests(unittest.TestCase):
    def test_groups_and_caps_keep_all_fifteen_experiments(self):
        real_root = Path(runner.__file__).resolve().parents[2]
        with patch.object(runner, 'ROOT', real_root):
            for seed in (42, 43, 44):
                low = runner.load_jobs(seed, 'low')
                high = runner.load_jobs(seed, 'high')
                self.assertEqual([x['target_ratio'] for x in low], ['0.015', '0.15', '1.5'])
                self.assertEqual([x['target_ratio'] for x in high], ['15', '150'])
                self.assertEqual([str(x['max_scale']) for x in low + high],
                                 ['1000', '20000', '500000', '10000000', '500000000'])
                for job in low + high:
                    cfg = yaml.safe_load((runner.ROOT / job['config']).read_text(encoding='utf-8'))
                    self.assertEqual(str(cfg['training']['dg_target_ratio']), job['target_ratio'])
                    self.assertEqual(str(cfg['training']['dg_max_scale']), str(job['max_scale']))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'saved_models').mkdir()
        (self.root / 'configs').mkdir()
        exp = 'lhs10_23_dg_ratio0p15_seed42_difficult_group_gradnorm_no_ss'
        self.job = {'alias': 'dg_ratio0p15', 'target_ratio': '0.15', 'max_scale': '20000',
                    'exp_name': exp, 'checkpoint': f'saved_models/{exp}.pth',
                    'results': f'results/pgin/retrain_lhs10_23/seed42/{exp}',
                    'config': 'configs/original.yaml', 'cli_args': ['--target-ratio', '0.15']}
        cfg = {'exp_name': exp, 'output': {'results_dir': str(self.root / 'results/pgin/retrain_lhs10_23/seed42')}}
        (self.root / self.job['config']).write_text(yaml.safe_dump(cfg), encoding='utf-8')
        self.ckpt = self.root / self.job['checkpoint']
        self.result = self.root / self.job['results']
        self.result.mkdir(parents=True)
        self.ckpt.write_bytes(b'old checkpoint')
        (self.result / 'old.txt').write_text('old', encoding='utf-8')
        self.unrelated = self.result.parent / 'other_model' / 'keep.txt'
        self.unrelated.parent.mkdir()
        self.unrelated.write_text('keep', encoding='utf-8')
        self.prepared = {'seed': 42, 'sources': {}, 'protected': {}}
        self.root_patch = patch.object(runner, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.lock = self.result.parent / '_dg_expanded_job_locks' / (self.job['alias'] + '.lock')
        self.lock.parent.mkdir()
        self.lock.write_text(f'pid={os.getpid()}\n', encoding='utf-8')

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            runner.safe_path('../outside.pth', self.root / 'saved_models')

    def test_symlink_to_outside_is_rejected(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        link = self.root / 'saved_models' / 'link'
        try:
            link.symlink_to(Path(other.name), target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest('Creating symlinks is unavailable on this system')
        with self.assertRaises(ValueError):
            runner.safe_path('saved_models/link/file.pth', self.root / 'saved_models')

    def test_cleanup_requires_own_lock_and_preserves_other_job(self):
        self.lock.unlink()
        with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
            runner.clear_old_artifacts(self.job, 42)
        self.assertTrue(self.ckpt.exists())
        self.assertTrue((self.result / 'old.txt').exists())
        self.assertEqual(self.unrelated.read_text(), 'keep')

    def test_cleanup_rejects_wrong_result_path_before_deleting_checkpoint(self):
        job = dict(self.job, results='../outside')
        with self.assertRaises(ValueError):
            runner.clear_old_artifacts(job, 42)
        self.assertEqual(self.ckpt.read_bytes(), b'old checkpoint')
        self.assertEqual((self.result / 'old.txt').read_text(), 'old')
        self.assertEqual(self.unrelated.read_text(), 'keep')

    def test_cleanup_rejects_foreign_lock_before_deleting_anything(self):
        self.lock.write_text('pid=999999999\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            runner.clear_old_artifacts(self.job, 42)
        self.assertEqual(self.ckpt.read_bytes(), b'old checkpoint')
        self.assertEqual((self.result / 'old.txt').read_text(), 'old')

    def test_check_only_never_clears_or_trains(self):
        args = ['runner', '--seed', '42', '--group', 'low', '--check-only']
        with patch.object(sys, 'argv', args), \
             patch.object(runner.os, 'chdir'), \
             patch.object(runner, 'prepare', return_value=self.prepared), \
             patch.object(runner, 'clear_old_artifacts') as clear, \
             patch.object(runner, 'run_job') as train:
            runner.main()
        clear.assert_not_called()
        train.assert_not_called()
        self.assertEqual(self.ckpt.read_bytes(), b'old checkpoint')
        self.assertTrue((self.result / 'old.txt').exists())

    def test_direct_success_writes_original_paths_and_journal(self):
        def simulated_training(args, log):
            self.assertEqual(Path(args[args.index('--config') + 1]), self.root / self.job['config'])
            self.assertFalse(self.ckpt.exists())
            self.assertFalse(self.result.exists())
            torch.save({'weight': torch.tensor([1.0])}, self.ckpt)
            self.result.mkdir(parents=True)
            for name in ('zscore_mean.csv', 'zscore_std.csv', 'gain_group_gradnorm_history.csv', 'ss_loss_history.csv'):
                (self.result / name).write_text('synthetic', encoding='utf-8')
        with patch.object(runner, 'run_training', side_effect=simulated_training), \
             patch.object(runner, 'validate_training'):
            runner.run_job(self.prepared, self.job, 'attempt')
        journal = self.result.parent / '_dg_direct_retrain/attempt/dg_ratio0p15.json'
        self.assertEqual(torch.load(self.ckpt, weights_only=True)['weight'].item(), 1.0)
        self.assertEqual(self.unrelated.read_text(), 'keep')
        self.assertEqual(json.loads(journal.read_text(encoding='utf-8'))['status'], 'complete')

    def test_direct_failure_records_failure_and_never_touches_other_job(self):
        with patch.object(runner, 'run_training', side_effect=RuntimeError('training failed')):
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                runner.run_job(self.prepared, self.job, 'attempt')
        journal = self.result.parent / '_dg_direct_retrain/attempt/dg_ratio0p15.json'
        self.assertEqual(json.loads(journal.read_text(encoding='utf-8'))['status'], 'failed')
        self.assertEqual(self.unrelated.read_text(), 'keep')

    def test_nonfinite_checkpoint_is_rejected(self):
        bad = self.root / 'saved_models/bad.pth'
        torch.save({'weight': torch.tensor([float('nan')])}, bad)
        with self.assertRaisesRegex(ValueError, 'Non-finite checkpoint'):
            runner.validate_training(bad, self.result)

    def test_interrupted_stream_stops_trainer_before_unlocking(self):
        class InterruptedStream:
            def __iter__(self):
                raise KeyboardInterrupt()
        class FakeProcess:
            def __init__(self):
                self.stdout = InterruptedStream()
                self.terminated = False
                self.waited = False
            def poll(self):
                return None
            def terminate(self):
                self.terminated = True
            def wait(self):
                self.waited = True
                return 1
        process = FakeProcess()
        with patch.object(runner.subprocess, 'Popen', return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_training(['ignored'], self.root / 'log.txt')
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)


if __name__ == '__main__':
    unittest.main()
