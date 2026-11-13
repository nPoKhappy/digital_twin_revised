import unittest

import numpy as np

from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts


class KCIRecountTests(unittest.TestCase):
    def test_two_directions_all18_include_zero_and_exclude_invalid_teacher(self):
        teacher = np.array([[[2e-5]], [[-2e-5]], [[0.0]]])
        dynamic = np.ones((2, 18, 3, 1, 1))
        dynamic[:, :, 1] = -1
        dynamic[0, 0, 0] = 0  # Valid teacher, zero dynamic gain: incorrect.
        dynamic[1, 0, 1] = 1  # Valid negative teacher, positive dynamic gain: FP.
        row = numpy_counts(teacher, dynamic, 1e-5, ['H2S'], ['air2']).iloc[0]
        self.assertEqual(row.possible_comparisons, 108)
        self.assertEqual(row.valid_comparisons, 72)
        self.assertEqual((row.tp, row.tn, row.fp, row.fn, row.zero_gain), (35, 35, 1, 0, 1))
        self.assertEqual(row.tp+row.tn+row.fp+row.fn+row.zero_gain, row.valid_comparisons)


if __name__ == '__main__':
    unittest.main()
