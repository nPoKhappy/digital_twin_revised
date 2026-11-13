import unittest

import numpy as np
from scripts.sampling import generate_ann_gain_validation_pairs as generator


class GainValidationPairsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.manifest, _ = generator.generate()

    def test_pair_counts_and_single_mv_step(self):
        self.assertEqual(self.inputs.shape, (5000, 17))
        base = self.inputs.iloc[::2].reset_index(drop=True)
        perturbed = self.inputs.iloc[1::2].reset_index(drop=True)
        meta = self.manifest.iloc[1::2].reset_index(drop=True)
        self.assertEqual(meta.groupby("region").size().to_dict(), {"core": 1250, "soft_ood": 1250})
        self.assertEqual(meta.groupby(["mv_name", "direction"]).size().to_dict(), {
            ("air2_SP", -1): 625, ("air2_SP", 1): 625,
            ("HEATER2_output_T_SP", -1): 625, ("HEATER2_output_T_SP", 1): 625,
        })
        delta = perturbed.to_numpy() - base.to_numpy()
        self.assertTrue(np.all(np.count_nonzero(delta, axis=1) == 1))
        steps = generator.steps_from_scaler()
        self.assertAlmostEqual(steps[generator.AIR], 22.24311225638926)
        self.assertAlmostEqual(steps[generator.T2], 10.708008063614638)
        for i, row in meta.iterrows():
            self.assertAlmostEqual(delta[i, self.inputs.columns.get_loc(row.perturbed_mv)], row.direction * steps[row.perturbed_mv])

    def test_new_execution_draws_new_samples(self):
        other_inputs, _, report = generator.generate()
        self.assertFalse(self.inputs.equals(other_inputs))
        self.assertNotIn("seed", report)

    def test_validation_rejects_non_mv_change_and_crossed_boundary(self):
        altered = self.inputs.copy()
        altered.loc[1, generator.FLOW] += 0.1
        with self.assertRaises(ValueError):
            generator.validate(altered, self.manifest)
        altered = self.inputs.copy()
        altered.loc[0, generator.T2] = 1000.0
        altered.loc[1, generator.T2] = 1000.0
        with self.assertRaises(ValueError):
            generator.validate(altered, self.manifest)

    def test_validation_rejects_incorrect_crossing_metadata(self):
        altered = self.manifest.copy()
        altered.loc[0:1, "endpoint_in_base_stratum"] = ~altered.loc[0:1, "endpoint_in_base_stratum"]
        with self.assertRaises(ValueError):
            generator.validate(self.inputs, altered)


if __name__ == "__main__":
    unittest.main()
