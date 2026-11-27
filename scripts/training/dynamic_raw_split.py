"""Chronological 80/10/10 raw split before segment-wise downsampling."""
from pathlib import Path
import math

import pandas as pd

from src import utils as data_utils


def split_raw_then_downsample(frame, interval, use_median):
    """Return train, validation, test segments without crossing raw boundaries."""
    if interval < 1:
        raise ValueError('sampling interval must be positive')
    n = len(frame)
    train_end = int(n * 0.8)
    valid_end = train_end + int(n * 0.1)
    outputs = []
    for segment in (frame.iloc[:train_end], frame.iloc[train_end:valid_end], frame.iloc[valid_end:]):
        if interval > 1:
            if use_median:
                segment = segment.rolling(window=interval, min_periods=interval).median(numeric_only=True)
                segment = segment.iloc[interval-1::interval].reset_index(drop=True)
            else:
                segment = segment.iloc[::interval].reset_index(drop=True)
        outputs.append(segment.dropna().copy())
    return tuple(outputs)


def load_raw_training_partitions(cfg_data, interval, use_median):
    """Load raw files; test segments are never fed to the trainer or scaler."""
    if not (math.isclose(cfg_data['test_data_split'], 0.1) and
            math.isclose(cfg_data['valid_data_split'], 1 / 9)):
        raise ValueError('Raw 80/10/10 mode requires test_data_split=0.1 and valid_data_split=1/9')
    if not cfg_data.get('training_files'):
        raise ValueError('Raw split mode requires training_files')
    train, valid = [], []
    for name in cfg_data['training_files']:
        path = Path(cfg_data['path']) / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing raw training source: {path}')
        train_df, valid_df, _ = split_raw_then_downsample(pd.read_csv(path), interval, use_median)
        train.append(data_utils.apply_log_transform(train_df, ['B35_H2S', 'B35_SO2']))
        valid.append(data_utils.apply_log_transform(valid_df, ['B35_H2S', 'B35_SO2']))
    return train, valid
