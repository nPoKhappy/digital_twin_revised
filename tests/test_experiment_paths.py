import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.experiment_paths import PROJECT_ROOT, default_results_base, experiment_results_dir, teacher_results_dir


class ExperimentPathTests(unittest.TestCase):
    def test_model_families_do_not_collide(self):
        cases = {
            'transformer_no_gain_seed42': 'no_gain',
            'transformer_gain250_seed42': 'gain',
            'transformer_gain250_ss250_seed42': 'gain_ss',
            'transformer_ss250_only_seed42': 'ss_only',
            'transformer_gain005_seed42_difficult_group_gradnorm_no_ss': 'difficult_gradnorm',
        }
        for name, family in cases.items():
            with self.subTest(name=name):
                self.assertEqual(default_results_base(name), PROJECT_ROOT/'results/pgin'/family)

    def test_explicit_output_override_is_honored_without_creating_directories(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary)/'new_outputs'
            config = {'exp_name': 'example_gain250', 'output': {'results_dir': str(target)}}
            self.assertEqual(experiment_results_dir(config), target/'example_gain250')
            self.assertFalse(target.exists())

    def test_configured_relative_output_is_project_relative(self):
        config = {'exp_name': 'run', 'output': {'results_dir': 'results/pgin/gain'}}
        self.assertEqual(experiment_results_dir(config), PROJECT_ROOT/'results/pgin/gain/run')

    def test_teacher_stays_separate_from_dynamic_models(self):
        self.assertEqual(teacher_results_dir(), PROJECT_ROOT/'results/steady_state/Tabular_MLP_New')

    def test_all_configs_follow_their_config_family(self):
        for path in (PROJECT_ROOT/'configs').rglob('*.yaml'):
            config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
            family = path.parent.relative_to(PROJECT_ROOT/'configs')
            with self.subTest(config=path.name):
                self.assertEqual(experiment_results_dir(config), PROJECT_ROOT/'results'/family/config['exp_name'])

    def test_gain_and_ss_weights_move_together(self):
        configs = [yaml.safe_load(p.read_text(encoding='utf-8-sig')) for p in (PROJECT_ROOT/'configs/pgin/gain_ss').glob('*.yaml')]
        self.assertEqual(len(configs), 5)
        self.assertEqual({c['training']['gain_loss_weight'] for c in configs}, {.05, 2.5, 25, 250, 2500})
        for config in configs:
            self.assertEqual(config['training']['gain_loss_weight'], config['training']['ss_loss_weight'])
            self.assertEqual(config['training']['ss_target_scope'], 'all_48_outputs')
            self.assertEqual(config['training']['seed'], 42)


if __name__ == '__main__':
    unittest.main()
