import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.training.train_ann_lhs10_23_retrain import (
    INPUTS, connected_component_ids, cross_source_four_input_matches,
    grouped_split, input_sources, metrics, write_compatible_plots,
)


class IsolatedANNTests(unittest.TestCase):
    def test_grouped_split_keeps_pairs_and_is_repeatable(self):
        data = pd.DataFrame({"group_id": [f"pair_{i}" for i in range(120) for _ in range(2)] +
                                           [f"single_{i}" for i in range(300)]})
        first = grouped_split(data, 42)
        second = grouped_split(data, 42)
        self.assertTrue(first.split.equals(second.split))
        self.assertTrue((first.groupby("group_id").split.nunique() == 1).all())
        self.assertEqual(set(first.split), {"train", "valid", "test"})
        self.assertFalse(first.split.equals(grouped_split(data, 43).split))

    def test_failed_bridge_connects_successful_perturbations(self):
        # Rows 0 and 2 vary two coordinates; failed row 1 joins them via
        # separate single-variable differences and must be used in grouping.
        values = np.array([[1., 2., 3., 4., 5.], [2., 2., 3., 4., 5.],
                           [2., 3., 3., 4., 5.], [10., 20., 30., 40., 50.]])
        components, _ = connected_component_ids(values)
        self.assertEqual(components[0], components[1])
        self.assertEqual(components[1], components[2])
        self.assertNotEqual(components[2], components[3])
        with self.assertRaises(ValueError):
            connected_component_ids(np.array([[1., 2., 3., 4., np.nan]]))

    def test_cross_source_four_input_match_is_detected(self):
        evidence = pd.DataFrame([[1., 2., 3., 4., 5.], [2., 2., 3., 4., 5.]], columns=INPUTS)
        evidence["source_file"] = ["first.xlsx", "second.xlsx"]
        cross = cross_source_four_input_matches(evidence)
        self.assertEqual(cross["air2_SP"], 1)
        self.assertEqual(sum(cross.values()), 1)

    def test_source_override_rejects_ambiguous_or_missing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.xlsx"
            path.touch()
            self.assertEqual(input_sources([path]), [path.resolve()])
            with self.assertRaises(ValueError):
                input_sources([path, path])
            with self.assertRaises(FileNotFoundError):
                input_sources([Path(directory) / "missing.xlsx"])

    def test_metrics_use_each_target_in_original_units(self):
        actual = np.array([[1., 2.], [3., 4.]])
        predicted = np.array([[0., 2.], [4., 2.]])
        result = metrics(actual, predicted)
        self.assertEqual(result[0]["target"], "B35_H2S")
        self.assertAlmostEqual(result[0]["RMSE"], 1.)
        self.assertAlmostEqual(result[1]["RMSE"], np.sqrt(2.))
        self.assertEqual(result[0]["n"], 2)
        self.assertAlmostEqual(result[0]["MAPE"], 100.0 * (1.0 + 1.0 / 3.0) / 2.0)

    def test_plots_written_for_new_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            truth = np.array([[0.01, 0.002], [0.02, 0.003]])
            write_compatible_plots(output, pd.DataFrame({"epoch": [1, 2], "train_mse": [2., 1.], "valid_mse": [2.2, 1.2]}), truth, truth * .95)
            for name in ("training_curve.png", "parity_plot_B35_H2S.png", "parity_plot_B35_SO2.png"):
                self.assertGreater((output / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
