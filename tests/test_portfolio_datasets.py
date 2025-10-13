#!/usr/bin/env python3
"""Unit tests for portfolio dataset helpers."""

import numpy as np
import pytest
import torch

from hftraining.data_utils import MultiAssetPortfolioDataset, PairStockDataset


def _make_feature_matrix(close_prices: np.ndarray) -> np.ndarray:
    """Construct synthetic feature matrix with close price at index 3."""
    open_prices = close_prices * 0.99
    high_prices = close_prices * 1.01
    low_prices = close_prices * 0.98
    volume = np.linspace(10_000, 12_000, len(close_prices), dtype=np.float32)
    base = np.stack([open_prices, high_prices, low_prices, close_prices, volume], axis=1)
    spread = (high_prices - low_prices).reshape(-1, 1)
    return np.concatenate([base, spread], axis=1).astype(np.float32)


def _zscore(features: np.ndarray) -> np.ndarray:
    mu = features.mean(axis=0, keepdims=True)
    sigma = features.std(axis=0, keepdims=True) + 1e-8
    return ((features - mu) / sigma).astype(np.float32)


def test_multi_asset_future_returns_use_raw_prices():
    close_a = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32)
    close_b = np.array([50.0, 49.5, 49.0, 50.0, 51.5], dtype=np.float32)
    features_a = _make_feature_matrix(close_a)
    features_b = _make_feature_matrix(close_b)
    normalized_a = _zscore(features_a)
    normalized_b = _zscore(features_b)

    dataset = MultiAssetPortfolioDataset(
        asset_arrays=[normalized_a, normalized_b],
        asset_names=['A', 'B'],
        asset_close_prices=[close_a, close_b],
        sequence_length=3,
        prediction_horizon=1,
        close_feature_index=3,
    )

    sample = dataset[0]
    expected_return_a = (close_a[3] - close_a[2]) / close_a[2]
    expected_return_b = (close_b[3] - close_b[2]) / close_b[2]

    assert torch.isclose(
        sample['future_returns'][0],
        torch.tensor(expected_return_a, dtype=torch.float32),
        atol=1e-6,
    ).item()
    assert torch.isclose(
        sample['future_returns'][1],
        torch.tensor(expected_return_b, dtype=torch.float32),
        atol=1e-6,
    ).item()

    assert torch.isclose(
        sample['labels'][0, 0],
        torch.tensor(normalized_a[3, 3], dtype=torch.float32),
        atol=1e-6,
    ).item()
    assert torch.isclose(
        sample['labels'][1, 0],
        torch.tensor(normalized_b[3, 3], dtype=torch.float32),
        atol=1e-6,
    ).item()
    assert sample['input_ids'].shape == (3, normalized_a.shape[1] + normalized_b.shape[1])
    assert sample['attention_mask'].shape == (3,)


def test_pair_stock_dataset_future_returns_and_labels():
    close_a = np.array([100.0, 100.0, 100.0, 103.0], dtype=np.float32)
    close_b = np.array([100.0, 101.0, 102.0, 100.0], dtype=np.float32)
    features_a = _make_feature_matrix(close_a)
    features_b = _make_feature_matrix(close_b)
    normalized_a = _zscore(features_a)
    normalized_b = _zscore(features_b)

    dataset = PairStockDataset(
        stock_a=normalized_a,
        stock_b=normalized_b,
        sequence_length=3,
        prediction_horizon=1,
        name_a='A',
        name_b='B',
        raw_close_a=close_a,
        raw_close_b=close_b,
        close_feature_index=3,
    )

    sample = dataset[0]
    expected_return_a = (close_a[3] - close_a[2]) / close_a[2]
    expected_return_b = (close_b[3] - close_b[2]) / close_b[2]

    assert torch.isclose(
        sample['future_returns'][0],
        torch.tensor(expected_return_a, dtype=torch.float32),
        atol=1e-6,
    ).item()
    assert torch.isclose(
        sample['future_returns'][1],
        torch.tensor(expected_return_b, dtype=torch.float32),
        atol=1e-6,
    ).item()

    assert sample['action_labels'].tolist() == [0, 2]
    assert torch.isclose(
        sample['labels'][0, 0],
        torch.tensor(normalized_a[3, 3], dtype=torch.float32),
        atol=1e-6,
    ).item()
    assert torch.isclose(
        sample['labels'][1, 0],
        torch.tensor(normalized_b[3, 3], dtype=torch.float32),
        atol=1e-6,
    ).item()


def test_pair_stock_dataset_requires_raw_prices():
    arr = _zscore(_make_feature_matrix(np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)))
    with pytest.raises(ValueError, match="Raw close price arrays are required"):
        PairStockDataset(
            stock_a=arr,
            stock_b=arr,
            sequence_length=3,
            prediction_horizon=1,
            name_a='A',
            name_b='B',
        )
