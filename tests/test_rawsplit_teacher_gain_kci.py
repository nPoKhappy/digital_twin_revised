import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.plot_rawsplit_teacher_gain_kci import ROOT, config_label, run


class RawsplitTeacherGainKciTests(unittest.TestCase):
    def test_labels_use_config_weights_instead_of_ambiguous_alias(self):
        weights = {'gain_loss_weight': 0.05, 'ss_loss_weight': 0.05, 'dg_target_ratio': '0.015'}
        self.assertEqual(config_label('gain', weights), 'Gain = 0.05')
        self.assertEqual(config_label('ss_only', weights), 'SS-only = 0.05')
        self.assertEqual(config_label('gain_ss', weights), 'Gain + SS = 0.05 / 0.05')
        self.assertEqual(config_label('difficult_gradnorm', weights), 'Difficult-GradNorm, ρ = 0.015')

    def test_rejects_historical_checkpoint_before_creating_output(self):
        families = ('no_gain', 'gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
        rows = [dict(family=family, alias=family, seed=seed, model=f'{family}_{seed}',
                     config=f'configs/pgin/retrain_lhs10_23/seed{seed}/{family}.yaml',
                     checkpoint=f'saved_models/old_{family}_{seed}.pth')
                for family in families for seed in (42, 43, 44)]
        with tempfile.TemporaryDirectory(prefix='test_rawsplit_selection_', dir=ROOT) as tmp:
            source = Path(tmp) / 'source'
            source.mkdir()
            (source / 'selected_models.json').write_text(json.dumps({'selected_models': rows}), encoding='utf8')
            destination = Path(tmp) / 'result'
            with self.assertRaisesRegex(ValueError, 'not from the raw-split experiment'):
                run(source, destination)
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
