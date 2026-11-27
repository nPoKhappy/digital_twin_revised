import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.training.dynamic_raw_split import split_raw_then_downsample, load_raw_training_partitions
from src import utils as data_utils


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/Claus_dynamic'


class RawDynamicSplitTests(unittest.TestCase):
    def test_raw_split_matches_existing_split_files_after_downsampling(self):
        for case in ('5', '5-1', '5-2', '5-6'):
            with self.subTest(case=case):
                base = DATA / f'Test_dataform_change_air2_R={case}'
                full = pd.read_csv(str(base) + '.csv')
                result = split_raw_then_downsample(full, 10, True)
                for actual, kind in zip(result, ('train', 'valid', 'test')):
                    old = pd.read_csv(str(base) + f'_{kind}_split.csv')
                    expected = old.rolling(10, min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
                    assert_frame_equal(actual, expected)

    def test_loader_uses_only_raw_train_for_scaler_and_keeps_valid_distinct(self):
        cfg = {'path': str(DATA), 'training_files': ['Test_dataform_change_air2_R=5-6.csv'],
               'test_data_split': 0.1, 'valid_data_split': 1 / 9}
        train, valid = load_raw_training_partitions(cfg, 10, True)
        self.assertEqual((len(train[0]), len(valid[0])), (1267, 158))
        expected = pd.read_csv(DATA / 'Test_dataform_change_air2_R=5-6_train_split.csv')
        expected = expected.rolling(10, min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        expected = data_utils.apply_log_transform(expected, ['B35_H2S', 'B35_SO2'])
        assert_frame_equal(train[0], expected)
        self.assertFalse(train[0].equals(valid[0]))

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            load_raw_training_partitions({'test_data_split': .1, 'valid_data_split': .1}, 10, True)


if __name__ == '__main__':
    unittest.main()
