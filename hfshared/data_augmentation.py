#!/usr/bin/env python3
"""Shared data augmentation utilities for time-series.

Lightweight helpers that preserve price relationships when desired.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


def gaussian_noise(data: np.ndarray, std: float = 0.01, preserve_ohlc: bool = True) -> np.ndarray:
    """Add Gaussian noise. Optionally preserve OHLC ordering by applying same noise to 4 price columns."""
    out = data.copy()
    if preserve_ohlc and out.shape[1] >= 4:
        noise_vec = np.random.normal(0.0, std, size=(out.shape[0], 1))
        out[:, :4] = out[:, :4] * (1.0 + noise_vec)
        if out.shape[1] > 4:
            out[:, 4:] = out[:, 4:] + np.random.normal(0.0, std, size=(out.shape[0], out.shape[1] - 4))
    else:
        out = out + np.random.normal(0.0, std, size=out.shape)
    return out


def random_scaling(data: np.ndarray, std: float = 0.05, preserve_ohlc: bool = True) -> np.ndarray:
    """Random multiplicative scaling over time. If preserve_ohlc, apply the same scale to OHLC columns."""
    out = data.copy()
    scale = np.random.normal(1.0, std, size=(out.shape[0], 1))
    if preserve_ohlc and out.shape[1] >= 4:
        out[:, :4] = out[:, :4] * scale
        if out.shape[1] > 4:
            out[:, 4:] = out[:, 4:] * np.random.normal(1.0, std, size=(out.shape[0], out.shape[1] - 4))
    else:
        out = out * np.random.normal(1.0, std, size=out.shape)
    return out


def pct_change_transform(data: np.ndarray) -> np.ndarray:
    """Convert each feature to one-step percent change; first row becomes zeros."""
    out = np.zeros_like(data)
    out[1:] = (data[1:] - data[:-1]) / (data[:-1] + 1e-12)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

