from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, P_HIGH, P_LOW
from scripts.search_binance33_meta_anneal import (
    Candidate,
    ScoreBank,
    _apply_binary_fills,
    _apply_short_binary_fills,
    _combine_scores,
    _desired_weights,
    _normalize_score_matrix,
    _normalise_alloc,
)


def test_normalize_score_matrix_zscores_each_day() -> None:
    scores = _normalize_score_matrix(np.asarray([[1.0, 2.0, 3.0], [2.0, np.nan, 4.0]]))

    assert np.isclose(float(np.nanmean(scores[0])), 0.0)
    assert np.isclose(float(np.nanstd(scores[0])), 1.0)
    assert np.isnan(scores[1, 1])


def test_combine_scores_uses_softmax_logits() -> None:
    bank = ScoreBank(
        names=["a", "b"],
        scores=np.asarray(
            [
                [[1.0, 2.0]],
                [[10.0, 20.0]],
            ],
            dtype=np.float64,
        ),
    )
    cand = Candidate(
        candidate_id="x",
        logits=np.asarray([10.0, -10.0]),
        threshold=0.0,
        max_gross=1.0,
        max_weight=1.0,
        top_k=1,
        book_mode="single",
        score_temp=1.0,
        btc_gate=-99.0,
        market_gate=-99.0,
        rebalance_days=1,
    )

    combined = _combine_scores(bank, cand)

    assert np.allclose(combined, [[1.0, 2.0]], atol=1e-6)


def test_normalise_alloc_respects_cap() -> None:
    alloc = _normalise_alloc(np.asarray([10.0, 1.0, 1.0]), gross=1.0, max_weight=0.5)

    assert np.isclose(float(alloc.sum()), 1.0)
    assert float(alloc.max()) <= 0.5 + 1e-12


def test_binary_short_fill_blocks_new_short_without_high_cross() -> None:
    features = np.zeros((2, 1, 16), dtype=np.float32)
    prices = np.ones((2, 1, 5), dtype=np.float32) * 100.0
    prices[:, :, P_CLOSE] = 100.0
    prices[:, :, P_HIGH] = 100.01
    data = MktdData(version=2, symbols=["BTCUSD"], features=features, prices=prices, tradable=None)

    blocked = _apply_short_binary_fills(
        data,
        np.asarray([0.0]),
        np.asarray([-1.0]),
        t=0,
        fill_buffer_bps=5.0,
    )
    prices[:, :, P_HIGH] = 100.10
    filled = _apply_short_binary_fills(
        data,
        np.asarray([0.0]),
        np.asarray([-1.0]),
        t=0,
        fill_buffer_bps=5.0,
    )

    assert blocked[0] == 0.0
    assert filled[0] == -1.0


def test_binary_long_fill_blocks_new_long_without_low_cross() -> None:
    features = np.zeros((2, 1, 16), dtype=np.float32)
    prices = np.ones((2, 1, 5), dtype=np.float32) * 100.0
    prices[:, :, P_CLOSE] = 100.0
    prices[:, :, P_LOW] = 99.99
    data = MktdData(version=2, symbols=["BTCUSD"], features=features, prices=prices, tradable=None)

    blocked = _apply_binary_fills(
        data,
        np.asarray([0.0]),
        np.asarray([1.0]),
        t=0,
        fill_buffer_bps=5.0,
    )
    prices[:, :, P_LOW] = 99.90
    filled = _apply_binary_fills(
        data,
        np.asarray([0.0]),
        np.asarray([1.0]),
        t=0,
        fill_buffer_bps=5.0,
    )

    assert blocked[0] == 0.0
    assert filled[0] == 1.0


def test_longshort_portfolio_splits_long_and_short_with_short_risk_modifier() -> None:
    features = np.zeros((1, 4, 16), dtype=np.float32)
    prices = np.ones((1, 4, 5), dtype=np.float32) * 100.0
    data = MktdData(version=2, symbols=["A", "B", "C", "D"], features=features, prices=prices, tradable=None)
    scores_by_t = np.asarray([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float64)
    candidate = Candidate(
        candidate_id="ls",
        logits=np.zeros(1, dtype=np.float64),
        threshold=0.5,
        max_gross=2.0,
        max_weight=1.0,
        top_k=4,
        book_mode="longshort_portfolio",
        score_temp=1.0,
        btc_gate=-99.0,
        market_gate=-99.0,
        rebalance_days=1,
        long_fraction=0.5,
        short_risk_mult=1.1,
        always_trade=True,
    )

    weights = _desired_weights(data, scores_by_t, candidate, t=0, btc_idx=0)

    assert np.all(weights[:2] < 0.0)
    assert np.all(weights[2:] > 0.0)
    assert np.isclose(float(weights[weights > 0.0].sum()), 1.0)
    assert np.isclose(float(-weights[weights < 0.0].sum()), 1.0 / 1.1)


def test_longshort_portfolio_top_one_keeps_single_slot() -> None:
    features = np.zeros((1, 4, 16), dtype=np.float32)
    prices = np.ones((1, 4, 5), dtype=np.float32) * 100.0
    data = MktdData(version=2, symbols=["A", "B", "C", "D"], features=features, prices=prices, tradable=None)
    scores_by_t = np.asarray([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float64)
    candidate = Candidate(
        candidate_id="ls1",
        logits=np.zeros(1, dtype=np.float64),
        threshold=0.5,
        max_gross=1.0,
        max_weight=1.0,
        top_k=1,
        book_mode="longshort_portfolio",
        score_temp=1.0,
        btc_gate=-99.0,
        market_gate=-99.0,
        rebalance_days=1,
        long_fraction=0.5,
        short_risk_mult=1.1,
        always_trade=True,
    )

    weights = _desired_weights(data, scores_by_t, candidate, t=0, btc_idx=0)

    assert np.count_nonzero(np.abs(weights) > 1e-12) == 1
