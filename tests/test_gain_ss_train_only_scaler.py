import unittest

import numpy as np
import pandas as pd

from scripts.training.train_pgin_stepwise_gain_ss_loss import (
    calculate_dynamic_scaler_stats,
    dynamic_split_points,
)


class GainSSTrainOnlyScalerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {"test_data_split": 0.1, "valid_data_split": 0.1}
        self.frames = [
            pd.DataFrame({"x": np.arange(n, dtype=float) + offset, "y": np.arange(n, dtype=float) * 2})
            for n, offset in [(100, 0), (35, 1000), (19, 10000)]
        ]

    def test_train_only_matches_actual_training_boundaries_and_ignores_unusable_files(self):
        mean, std, count = calculate_dynamic_scaler_stats(self.frames, self.cfg, 10, 5, "train_only")
        expected = pd.concat([self.frames[0].iloc[:81], self.frames[1].iloc[:27]], ignore_index=True)
        self.assertEqual(count, 108)
        pd.testing.assert_series_equal(mean, expected.mean())
        pd.testing.assert_series_equal(std, expected.std())

    def test_validation_and_tail_values_do_not_affect_train_only(self):
        before_mean, before_std, _ = calculate_dynamic_scaler_stats(self.frames, self.cfg, 10, 5, "train_only")
        changed = [df.copy() for df in self.frames]
        for df in changed:
            _, train_end = dynamic_split_points(len(df), 0.1, 0.1)
            df.iloc[train_end:] = 1e9
        after_mean, after_std, _ = calculate_dynamic_scaler_stats(changed, self.cfg, 10, 5, "train_only")
        pd.testing.assert_series_equal(before_mean, after_mean)
        pd.testing.assert_series_equal(before_std, after_std)

    def test_legacy_scope_reproduces_existing_full_file_scaler(self):
        mean, std, count = calculate_dynamic_scaler_stats(self.frames, self.cfg, 10, 5, "legacy")
        expected = pd.concat(self.frames, ignore_index=True)
        self.assertEqual(count, len(expected))
        pd.testing.assert_series_equal(mean, expected.mean())
        pd.testing.assert_series_equal(std, expected.std())

    def test_split_is_sequential_and_rounds_at_each_stage(self):
        self.assertEqual(dynamic_split_points(101, 0.2, 0.15), (80, 68))

    def test_unknown_scope_and_no_usable_training_windows_fail(self):
        with self.assertRaisesRegex(ValueError, "scaler_fit_scope"):
            calculate_dynamic_scaler_stats(self.frames, self.cfg, 10, 5, "unexpected")
        with self.assertRaisesRegex(ValueError, "No usable"):
            calculate_dynamic_scaler_stats([self.frames[2]], self.cfg, 10, 5, "train_only")


if __name__ == "__main__":
    unittest.main()
