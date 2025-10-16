#!/usr/bin/env python3
"""Tests for hfinference DataProcessor feature handling to avoid drift and handle edge cases."""

import os
import sys
import numpy as np
import pandas as pd

# Ensure repo root on path
TEST_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from hfinference.hf_trading_engine import DataProcessor


def make_df(n=12, with_volume=False):
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    data = {
        'Open': np.linspace(100, 110, n),
        'High': np.linspace(101, 112, n),
        'Low': np.linspace(99, 109, n),
        'Close': np.linspace(100.5, 111, n),
    }
    if with_volume:
        data['Volume'] = np.linspace(1e6, 2e6, n)
    df = pd.DataFrame(data, index=idx)
    return df


def test_prepare_features_ohlc_missing_volume_pct_change():
    cfg = {'sequence_length': 10, 'feature_mode': 'auto', 'use_pct_change': True}
    dp = DataProcessor(cfg)
    df = make_df(n=12, with_volume=False)
    feats = dp.prepare_features(df)
    # expect last 10 rows, 4 features (OHLC only)
    assert feats.shape == (10, 4)


def test_prepare_features_force_ohlcv_when_no_volume():
    cfg = {'sequence_length': 10, 'feature_mode': 'ohlcv', 'use_pct_change': False}
    dp = DataProcessor(cfg)
    df = make_df(n=12, with_volume=False)
    feats = dp.prepare_features(df)
    # expect synthetic zero volume column included
    assert feats.shape == (10, 5)

