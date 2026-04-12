from __future__ import annotations

import pandas as pd

from trade_stock_wide.selection import (
    WideSelectionConfig,
    build_symbol_rl_prior,
    estimate_candidate_daily_return,
    rank_candidates,
    resolve_torch_device,
    rerank_candidate_days,
)
from trade_stock_wide.sweep import run_parameter_sweep
from trade_stock_wide.types import WideCandidate


def _candidate(
    symbol: str,
    *,
    strategy: str = "maxdiff",
    forecasted_pnl: float = 0.04,
    avg_return: float = 0.02,
    last_close: float = 100.0,
    entry_price: float = 99.0,
    take_profit_price: float = 103.0,
    realized_close: float = 100.0,
    realized_high: float = 101.0,
    realized_low: float = 99.0,
    score: float | None = None,
    session_date: str | None = "2024-03-04",
) -> WideCandidate:
    return WideCandidate(
        symbol=symbol,
        strategy=strategy,
        forecasted_pnl=forecasted_pnl,
        avg_return=avg_return,
        last_close=last_close,
        entry_price=entry_price,
        take_profit_price=take_profit_price,
        predicted_high=take_profit_price,
        predicted_low=entry_price,
        realized_close=realized_close,
        realized_high=realized_high,
        realized_low=realized_low,
        score=forecasted_pnl if score is None else score,
        day_index=0,
        session_date=session_date,
        dollar_vol_20d=20_000_000.0,
        spread_bps_estimate=5.0,
    )


def test_estimate_candidate_daily_return_matches_filled_take_profit():
    candidate = _candidate(
        "AAPL",
        entry_price=99.0,
        take_profit_price=103.0,
        realized_high=104.0,
        realized_low=98.5,
        realized_close=101.0,
    )

    realized = estimate_candidate_daily_return(candidate, fee_bps=0.0, fill_buffer_bps=0.0)

    assert realized == ((103.0 - 99.0) / 99.0)


def test_rank_candidates_sortino_prefers_symbol_with_stronger_history():
    history_days = [
        [_candidate("AAA", realized_close=101.0, realized_high=104.0, realized_low=98.0)],
        [_candidate("AAA", realized_close=100.5, realized_high=104.0, realized_low=98.5)],
        [_candidate("BBB", realized_close=95.0, realized_high=100.5, realized_low=98.0)],
        [_candidate("BBB", realized_close=94.0, realized_high=100.5, realized_low=98.0)],
    ]
    today = [
        _candidate("AAA", forecasted_pnl=0.045, score=0.045),
        _candidate("BBB", forecasted_pnl=0.060, score=0.060),
    ]

    ranked = rank_candidates(
        today,
        history_days=history_days,
        config=WideSelectionConfig(objective="sortino", lookback_days=10),
        fee_bps=0.0,
        fill_buffer_bps=0.0,
    )

    assert ranked[0].symbol == "AAA"


def test_rank_candidates_tiny_net_prefers_profitable_feature_regime():
    history_days = []
    for day in range(6):
        history_days.append(
            [
                _candidate(
                    "AAA",
                    forecasted_pnl=0.06 + (day * 0.002),
                    avg_return=0.03,
                    entry_price=98.5,
                    take_profit_price=104.0,
                    realized_close=103.0,
                    realized_high=104.5,
                    realized_low=98.0,
                ),
                _candidate(
                    "BBB",
                    forecasted_pnl=0.01 + (day * 0.001),
                    avg_return=0.005,
                    entry_price=99.5,
                    take_profit_price=101.0,
                    realized_close=97.0,
                    realized_high=100.0,
                    realized_low=99.0,
                ),
            ]
        )
    today = [
        _candidate("AAA", forecasted_pnl=0.05, avg_return=0.025, entry_price=98.7, take_profit_price=103.5),
        _candidate("BBB", forecasted_pnl=0.055, avg_return=0.010, entry_price=99.6, take_profit_price=101.0),
    ]

    ranked = rank_candidates(
        today,
        history_days=history_days,
        config=WideSelectionConfig(
            objective="tiny_net",
            lookback_days=20,
            tiny_net_epochs=200,
            tiny_net_hidden_dim=6,
            tiny_net_learning_rate=0.04,
            tiny_net_augment_copies=2,
            tiny_net_noise_scale=0.02,
            tiny_net_min_train_samples=4,
            seed=7,
        ),
        fee_bps=0.0,
        fill_buffer_bps=0.0,
    )

    assert ranked[0].symbol == "AAA"


def test_rank_candidates_torch_mlp_prefers_profitable_feature_regime():
    history_days = []
    for day in range(8):
        history_days.append(
            [
                _candidate(
                    "AAA",
                    forecasted_pnl=0.055 + (day * 0.001),
                    avg_return=0.028,
                    entry_price=98.4,
                    take_profit_price=104.2,
                    realized_close=103.1,
                    realized_high=104.6,
                    realized_low=98.0,
                ),
                _candidate(
                    "BBB",
                    forecasted_pnl=0.018 + (day * 0.0005),
                    avg_return=0.006,
                    entry_price=99.7,
                    take_profit_price=101.0,
                    realized_close=96.5,
                    realized_high=100.1,
                    realized_low=99.2,
                ),
            ]
        )
    today = [
        _candidate("AAA", forecasted_pnl=0.05, avg_return=0.025, entry_price=98.8, take_profit_price=103.7),
        _candidate("BBB", forecasted_pnl=0.052, avg_return=0.008, entry_price=99.6, take_profit_price=101.1),
    ]

    ranked = rank_candidates(
        today,
        history_days=history_days,
        config=WideSelectionConfig(
            objective="torch_mlp",
            lookback_days=20,
            tiny_net_epochs=80,
            tiny_net_hidden_dim=8,
            tiny_net_learning_rate=0.03,
            tiny_net_augment_copies=1,
            tiny_net_noise_scale=0.01,
            tiny_net_min_train_samples=4,
            torch_device="cpu",
            seed=11,
        ),
        fee_bps=0.0,
        fill_buffer_bps=0.0,
    )

    assert ranked[0].symbol == "AAA"


def test_build_symbol_rl_prior_aggregates_positive_group_scores():
    rows = pd.DataFrame(
        [
            {"symbols": "AAA,BBB", "robust_score": 3.0},
            {"symbols": "AAA,CCC", "robust_score": 1.0},
            {"symbols": "BBB,CCC", "robust_score": -2.0},
        ]
    )

    prior = build_symbol_rl_prior(rows, min_score=0.0, scale=2.0)

    assert prior["AAA"] > prior["CCC"]
    assert prior["BBB"] == prior["AAA"]


def test_resolve_torch_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("trade_stock_wide.selection.torch.cuda.is_available", lambda: False)

    assert resolve_torch_device("auto") == "cpu"
    assert resolve_torch_device("cuda") == "cpu"


def test_rank_candidates_rl_prior_can_override_close_base_scores():
    history_days = []
    today = [
        _candidate("AAA", forecasted_pnl=0.041, score=0.041),
        _candidate("BBB", forecasted_pnl=0.042, score=0.042),
    ]

    ranked = rank_candidates(
        today,
        history_days=history_days,
        config=WideSelectionConfig(
            objective="pnl",
            rl_prior_weight=0.05,
            rl_prior_by_symbol={"AAA": 1.0, "BBB": 0.0},
        ),
        fee_bps=0.0,
        fill_buffer_bps=0.0,
    )

    assert ranked[0].symbol == "AAA"
    assert ranked[0].rl_prior_score == 1.0


def test_rerank_candidate_days_uses_only_prior_history():
    candidate_days = [
        [
            _candidate("AAA", forecasted_pnl=0.03, score=0.03, realized_close=103.0, realized_high=104.0, realized_low=98.0),
            _candidate("BBB", forecasted_pnl=0.05, score=0.05, realized_close=95.0, realized_high=100.5, realized_low=98.0),
        ],
        [
            _candidate("AAA", forecasted_pnl=0.04, score=0.04, realized_close=103.0, realized_high=104.0, realized_low=98.0),
            _candidate("BBB", forecasted_pnl=0.05, score=0.05, realized_close=95.0, realized_high=100.5, realized_low=98.0),
        ],
    ]

    reranked = rerank_candidate_days(
        candidate_days,
        config=WideSelectionConfig(objective="sortino", lookback_days=10),
        fee_bps=0.0,
        fill_buffer_bps=0.0,
    )

    assert reranked[0][0].symbol == "BBB"
    assert reranked[1][0].symbol == "AAA"


def test_run_parameter_sweep_produces_sorted_leaderboard():
    candidate_days = [
        [
            _candidate("AAA", forecasted_pnl=0.04, score=0.04, realized_close=103.0, realized_high=104.0, realized_low=98.0),
            _candidate("BBB", forecasted_pnl=0.05, score=0.05, realized_close=95.0, realized_high=100.5, realized_low=98.0),
        ],
        [
            _candidate("AAA", forecasted_pnl=0.05, score=0.05, realized_close=104.0, realized_high=104.5, realized_low=98.0),
            _candidate("BBB", forecasted_pnl=0.06, score=0.06, realized_close=94.0, realized_high=100.0, realized_low=98.0),
        ],
    ]

    leaderboard = run_parameter_sweep(
        candidate_days,
        starting_equity=10_000.0,
        selection_objectives=["pnl", "sortino"],
        top_ks=[1, 2],
        watch_activation_pcts=[0.005],
        steal_protection_pcts=[0.004],
        fee_bps=0.0,
        fill_buffer_bps=0.0,
        daily_only=True,
        hourly_by_symbol={"AAA": pd.DataFrame(), "BBB": pd.DataFrame()},
    )

    assert not leaderboard.empty
    assert leaderboard.iloc[0]["monthly_return"] >= leaderboard.iloc[-1]["monthly_return"]
    assert set(["selection_objective", "top_k", "monthly_return", "max_drawdown"]).issubset(leaderboard.columns)
