import argparse
import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from scripts.training import train_pgin_stepwise_gain_difficult_group_gradnorm as dg


class DifficultGradnormRetrainTests(unittest.TestCase):
    def test_train_scaler_excludes_heldout_and_short_segments(self):
        cfg = {'test_data_split': 0.1, 'valid_data_split': 0.1}
        long = pd.DataFrame({'x': np.arange(100.0)})
        short = pd.DataFrame({'x': np.full(60, 99999.0)})
        altered = long.copy()
        altered.iloc[81:] = -100000
        mean, std = dg.calculate_dynamic_stats([long, short], cfg, 'train_only', 54)
        pd.testing.assert_series_equal(mean, long.iloc[:81].mean())
        pd.testing.assert_series_equal(std, long.iloc[:81].std())
        pd.testing.assert_series_equal(mean, dg.calculate_dynamic_stats([altered, short], cfg, 'train_only', 54)[0])
        self.assertFalse(mean.equals(dg.calculate_dynamic_stats([long, short], cfg, 'legacy')[0]))
        with self.assertRaisesRegex(ValueError, 'No dynamic data'):
            dg.calculate_dynamic_stats([short], cfg, 'train_only', 54)
        with self.assertRaisesRegex(ValueError, 'scaler_fit_scope'):
            dg.calculate_dynamic_stats([long], cfg, 'wrong')

    def test_cli_legacy_and_explicit_eleven_source_route(self):
        # Execute only the module's actual argparse block with a captured main;
        # no model is created and no training or output is started.
        source = Path(dg.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        block = next(node for node in tree.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "__name__ == '__main__'")
        code = compile(ast.fix_missing_locations(ast.Module(body=block.body, type_ignores=[])), dg.__file__, 'exec')
        # The earlier 0.01 config was retired; 0.015 is the retained historical run.
        config = 'configs/pgin/retrain_lhs10_23/seed42/dg_ratio0p015.yaml'
        def run(extra):
            captured = {}
            def fake_main(*args, **kwargs):
                captured.update(kwargs)
            ns = {'argparse': argparse, 'yaml': yaml, 'main': fake_main,
                  'BASELINE_STEADY_STATE_SOURCES': dg.BASELINE_STEADY_STATE_SOURCES,
                  'DIFFICULT_PAIRS': dg.DIFFICULT_PAIRS}
            with patch.object(sys, 'argv', ['trainer', '--config', config, *extra]):
                exec(code, ns)
            return captured
        legacy = run([])
        self.assertEqual(legacy['steady_state_source_override'], dg.BASELINE_STEADY_STATE_SOURCES)
        self.assertEqual(len(legacy['steady_state_source_override']), 8)
        new = run(['--no-ss', '--target-ratio', '0.01', '--max-scale', '50', '--use-config-steady-state-source'])
        self.assertIsNone(new['steady_state_source_override'])
        self.assertEqual(new['h2s_group_target_ratio'], 0.01)
        self.assertEqual(new['h2s_group_max_scale'], 50)
        self.assertEqual(new['ss_loss_weight_override'], 0)
        self.assertEqual(new['experiment_suffix'], '')
        with open(config, encoding='utf-8') as f:
            pattern = yaml.safe_load(f)['training']['steady_state_source_path']
        self.assertEqual(len(list(Path('.').glob(pattern))), 11)


if __name__ == '__main__':
    unittest.main()
