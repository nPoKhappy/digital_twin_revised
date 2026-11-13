import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.evaluation.validate_ann_aspen_steps import (
    classify_sign,
    gain_sign,
    load_cases,
    load_step_magnitudes,
    pair_status,
    valid_aspen_outputs,
)


class GainSignTests(unittest.TestCase):
    def test_gain_sign_includes_signed_mv_step(self):
        self.assertEqual(gain_sign(0.01, 2.5), 1)
        self.assertEqual(gain_sign(-0.01, 2.5), -1)
        self.assertEqual(gain_sign(0.01, -2.5), -1)
        self.assertEqual(gain_sign(-0.01, -2.5), 1)
        self.assertEqual(gain_sign(0.0, -2.5), 0)

    def test_sign_agreement_and_exact_zeros(self):
        for aspen, ann in [(1.0, 2.0), (-1.0, -2.0), (1e-30, 1e-30)]:
            with self.subTest(aspen=aspen, ann=ann):
                self.assertEqual(classify_sign(aspen, ann), "match")
        for aspen, ann in [(1.0, -2.0), (-1.0, 2.0), (1e-30, -1e-30)]:
            with self.subTest(aspen=aspen, ann=ann):
                self.assertEqual(classify_sign(aspen, ann), "mismatch")
        self.assertEqual(classify_sign(0.0, 1.0), "aspen_zero")
        self.assertEqual(classify_sign(1.0, 0.0), "ann_zero")
        self.assertEqual(classify_sign(0.0, 0.0), "both_zero")

    def test_aspens_failed_pairs_cannot_be_valid(self):
        for base, step, expected in [
            ("Run Completed", "Run Completed", "valid"),
            ("Run failed", "Run Completed", "base_failed"),
            ("Run Completed", "Run failed", "step_failed"),
            ("Run failed", "Run failed", "both_failed"),
        ]:
            with self.subTest(base=base, step=step):
                self.assertEqual(pair_status(base, step), expected)

    def test_missing_and_negative_aspen_outputs_rejected(self):
        values = pd.DataFrame({
            'S33.Zn.H2S.("H2S")': [0.01, 0.02],
            'S33.Zn.SO2.("SO2")': [0.03, 0.04],
        })
        self.assertTrue(valid_aspen_outputs(values, [0, 1]))
        for value in (None, np.nan, -0.01, 1.01):
            with self.subTest(value=value):
                altered = values.copy()
                altered.iloc[1, 0] = value
                self.assertFalse(valid_aspen_outputs(altered, [0, 1]))


class StepProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aspen_step_test_")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.inputs = self.folder / "aspen_inputs.csv"
        self.manifest = self.folder / "row_manifest.csv"
        self.inputs.write_text("example\n1\n", encoding="utf-8")
        self.manifest.write_text("example\n1\n", encoding="utf-8")

    def write_report(self, step_fields):
        report = {
            **step_fields,
            "output_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (self.inputs, self.manifest)
            },
        }
        (self.folder / "generation_report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_old_scalar_and_large_per_mv_reports(self):
        self.write_report({"step_magnitude": 2.5})
        self.assertEqual(load_step_magnitudes(self.folder), {
            "air2_SP": 2.5, "HEATER2_output_T_SP": 2.5,
        })
        self.write_report({"step_magnitudes": {"air2_SP": 22.24311225638926, "HEATER2_output_T_SP": 10.70800806361464}})
        self.assertEqual(load_step_magnitudes(self.folder), {
            "air2_SP": 22.24311225638926, "HEATER2_output_T_SP": 10.70800806361464,
        })

    def test_rejects_missing_nonfinite_and_nonpositive_steps(self):
        for fields in (
            {},
            {"step_magnitudes": {"air2_SP": 22.2}},
            {"step_magnitude": 0},
            {"step_magnitude": -1},
            {"step_magnitude": float("nan")},
            {"step_magnitudes": {"air2_SP": float("inf"), "HEATER2_output_T_SP": 10.7}},
        ):
            with self.subTest(fields=fields):
                self.write_report(fields)
                with self.assertRaises((ValueError, KeyError)):
                    load_step_magnitudes(self.folder)

    def test_rejects_mutated_input_or_manifest(self):
        for target in (self.inputs, self.manifest):
            with self.subTest(target=target.name):
                self.write_report({"step_magnitude": 2.5})
                target.write_text("modified\n", encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    load_step_magnitudes(self.folder)

    def test_load_cases_rejects_wrong_signed_manifest_delta(self):
        tags = ["B33.SPo.SPo", "B20.SPo.SPo", "B34.SPo.SPo", "ACIDGAS.P.P", "ACIDGAS.T.T"]
        tags += [f"other_{i}" for i in range(12)]
        base = [200.0, 180.0, 140.0, 1.67, 83.6] + [1.0] * 12
        step = base.copy()
        step[0] -= 22.24311225638926
        inputs = pd.DataFrame([base, step], columns=tags)
        inputs.to_csv(self.inputs, index=False)
        metadata = pd.DataFrame([
            {"input_row_1based": 1, "case_id": "base", "base_id": 1, "case_type": "base", "region": "core", "stratum": "core", "perturbed_mv": tags[0], "mv_name": "air2_SP", "direction": -1, "delta": 0.0},
            {"input_row_1based": 2, "case_id": "step", "base_id": 1, "case_type": "perturb", "region": "core", "stratum": "core", "perturbed_mv": tags[0], "mv_name": "air2_SP", "direction": -1, "delta": -22.24311225638926},
        ])
        metadata.to_csv(self.manifest, index=False)
        steps = {"air2_SP": 22.24311225638926, "HEATER2_output_T_SP": 10.70800806361464}
        self.write_report({"step_magnitudes": steps})
        workbook = inputs.copy()
        workbook["Scenario"] = ["Case 1", "Case 2"]
        workbook["Status"] = "Run Completed"
        workbook['S33.Zn.H2S.("H2S")'] = [0.01, 0.02]
        workbook['S33.Zn.SO2.("SO2")'] = [0.02, 0.03]
        with patch("scripts.evaluation.validate_ann_aspen_steps.pd.read_excel", return_value=workbook):
            sheet, manifest, max_diff = load_cases(self.folder / "mock.xlsx", self.folder)
            self.assertEqual(len(sheet), 2)
            self.assertEqual(len(manifest), 2)
            self.assertLessEqual(max_diff, 1e-10)
            metadata.loc[1, "delta"] *= -1
            metadata.to_csv(self.manifest, index=False)
            self.write_report({"step_magnitudes": steps})
            with self.assertRaises(ValueError):
                load_cases(self.folder / "mock.xlsx", self.folder)


if __name__ == "__main__":
    unittest.main()
