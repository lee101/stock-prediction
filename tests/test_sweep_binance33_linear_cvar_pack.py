from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, P_HIGH
from scripts.sweep_binance33_linear_cvar_pack import (
    _apply_short_binary_fills,
    _normalise_alloc,
    _risk_adjusted_raw,
)


def test_normalise_alloc_respects_gross_and_cap() -> None:
    alloc = _normalise_alloc(np.asarray([10.0, 1.0, 1.0]), gross=1.0, max_weight=0.5)

    assert np.isclose(float(alloc.sum()), 1.0)
    assert float(alloc.max()) <= 0.5 + 1e-12
    assert alloc[0] == 0.5


def test_cvar_raw_downweights_higher_tail_risk() -> None:
    raw = _risk_adjusted_raw(
        mode="cvar",
        edge=np.asarray([1.0, 1.0]),
        risk=np.asarray([0.02, 0.20]),
        mean=np.asarray([0.01, 0.01]),
        vol=np.asarray([0.02, 0.02]),
        edge_power=1.0,
        risk_power=1.0,
        edge_return_scale=0.005,
        score_temp=1.0,
    )

    assert raw[0] > raw[1]


def test_mean_cvar_ignores_negative_realized_edge() -> None:
    raw = _risk_adjusted_raw(
        mode="mean_cvar",
        edge=np.asarray([1.0, 1.0]),
        risk=np.asarray([0.03, 0.03]),
        mean=np.asarray([0.02, -0.02]),
        vol=np.asarray([0.02, 0.02]),
        edge_power=1.0,
        risk_power=1.0,
        edge_return_scale=0.005,
        score_temp=1.0,
    )

    assert raw[0] > 0.0
    assert raw[1] == 0.0


def test_short_binary_fill_requires_buffer_for_new_short() -> None:
    features = np.zeros((2, 1, 16), dtype=np.float32)
    prices = np.ones((2, 1, 5), dtype=np.float32) * 100.0
    prices[:, :, P_CLOSE] = 100.0
    prices[:, :, P_HIGH] = 100.01
    data = MktdData(version=2, symbols=["BTCUSD"], features=features, prices=prices, tradable=None)

    current = np.asarray([0.0])
    desired = np.asarray([-1.0])
    no_fill = _apply_short_binary_fills(data, current, desired, t=0, fill_buffer_bps=5.0)

    prices[:, :, P_HIGH] = 100.10
    fill = _apply_short_binary_fills(data, current, desired, t=0, fill_buffer_bps=5.0)

    assert no_fill[0] == 0.0
    assert fill[0] == -1.0


def test_short_binary_fill_reduces_existing_short_without_buffer_cross() -> None:
    features = np.zeros((2, 1, 16), dtype=np.float32)
    prices = np.ones((2, 1, 5), dtype=np.float32) * 100.0
    prices[:, :, P_CLOSE] = 100.0
    prices[:, :, P_HIGH] = 100.01
    data = MktdData(version=2, symbols=["BTCUSD"], features=features, prices=prices, tradable=None)

    current = np.asarray([-1.0])
    desired = np.asarray([-0.25])
    reduced = _apply_short_binary_fills(data, current, desired, t=0, fill_buffer_bps=5.0)

    assert reduced[0] == -0.25
