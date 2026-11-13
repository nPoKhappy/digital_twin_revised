import unittest

import numpy as np
import pandas as pd

from scripts.training.train_pgin_stepwise_gain import calculate_dynamic_stats, dynamic_split_points


class GainTrainingScalerScopeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'test_data_split': 0.1, 'valid_data_split': 0.1}

    def test_train_only_uses_pooled_chronological_training_partitions(self):
        a = pd.DataFrame({'mv': np.arange(100.0), 'target': np.arange(100.0) * 2})
        b = pd.DataFrame({'mv': np.arange(120.0) + 1000, 'target': np.arange(120.0) * 3})
        expected = pd.concat([a.iloc[:81], b.iloc[:97]], ignore_index=True)
        mean, std = calculate_dynamic_stats([a, b], self.cfg, 'train_only', min_train_rows=54)
        pd.testing.assert_series_equal(mean, expected.mean())
        pd.testing.assert_series_equal(std, expected.std())
        self.assertEqual(dynamic_split_points(100, self.cfg), (90, 81))
        self.assertEqual(dynamic_split_points(120, self.cfg), (108, 97))

    def test_validation_and_reserved_tail_cannot_change_train_only_stats(self):
        data = pd.DataFrame({'mv': np.arange(100.0), 'target': np.arange(100.0)})
        changed = data.copy()
        changed.iloc[81:] = -1000000.0
        before = calculate_dynamic_stats([data], self.cfg, 'train_only', min_train_rows=54)
        after = calculate_dynamic_stats([changed], self.cfg, 'train_only', min_train_rows=54)
        pd.testing.assert_series_equal(before[0], after[0])
        pd.testing.assert_series_equal(before[1], after[1])
        legacy_before = calculate_dynamic_stats([data], self.cfg, 'legacy')
        legacy_after = calculate_dynamic_stats([changed], self.cfg, 'legacy')
        self.assertFalse(legacy_before[0].equals(legacy_after[0]))

    def test_excludes_files_without_enough_training_rows(self):
        used = pd.DataFrame({'mv': np.arange(100.0)})
        too_short = pd.DataFrame({'mv': np.full(60, 100000.0)})
        mean, std = calculate_dynamic_stats([used, too_short], self.cfg, 'train_only', min_train_rows=54)
        pd.testing.assert_series_equal(mean, used.iloc[:81].mean())
        pd.testing.assert_series_equal(std, used.iloc[:81].std())
        with self.assertRaisesRegex(ValueError, 'No dynamic data'):
            calculate_dynamic_stats([too_short], self.cfg, 'train_only', min_train_rows=54)

    def test_legacy_is_the_full_original_input(self):
        frames = [pd.DataFrame({'x': np.arange(100.0)}), pd.DataFrame({'x': np.arange(80.0) + 5})]
        expected = pd.concat(frames, ignore_index=True)
        mean, std = calculate_dynamic_stats(frames, self.cfg, 'legacy', min_train_rows=54)
        pd.testing.assert_series_equal(mean, expected.mean())
        pd.testing.assert_series_equal(std, expected.std())
        with self.assertRaisesRegex(ValueError, 'scaler_fit_scope'):
            calculate_dynamic_stats(frames, self.cfg, 'unexpected')


if __name__ == '__main__':
    unittest.main()
