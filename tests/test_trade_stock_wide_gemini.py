from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trade_stock_wide.gemini_overlay import (
    GeminiWidePlan,
    build_wide_gemini_prompt,
    call_gemini_wide,
    refine_daily_candidates_with_gemini,
)
from trade_stock_wide.planner import WidePlannerConfig
from trade_stock_wide.types import WideCandidate
import trade_stock_wide.run as run_mod


def _candidate(
    symbol: str,
    *,
    forecasted_pnl: float = 0.05,
    avg_return: float = 0.02,
    entry_price: float = 99.0,
    take_profit_price: float = 103.0,
    score: float = 0.05,
    day_index: int = 0,
    session_date: str = "2026-04-10",
    allocation_fraction_of_equity: float | None = None,
) -> WideCandidate:
    return WideCandidate(
        symbol=symbol,
        strategy="maxdiff",
        forecasted_pnl=forecasted_pnl,
        avg_return=avg_return,
        last_close=100.0,
        entry_price=entry_price,
        take_profit_price=take_profit_price,
        predicted_high=104.0,
        predicted_low=98.0,
        realized_close=101.0,
        realized_high=104.0,
        realized_low=98.0,
        score=score,
        day_index=day_index,
        session_date=session_date,
        allocation_fraction_of_equity=allocation_fraction_of_equity,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-04-08T00:00:00Z",
                "open": 99.0,
                "high": 101.0,
                "low": 97.0,
                "close": 100.0,
                "volume": 1000.0,
                "predicted_close_p50": 101.0,
                "predicted_high": 104.0,
                "predicted_low": 98.0,
                "maxdiffprofit_profit": 0.03,
                "maxdiff_avg_daily_return": 0.01,
                "pctdiff_profit": 0.01,
                "pctdiff_avg_daily_return": 0.005,
            },
            {
                "timestamp": "2026-04-09T00:00:00Z",
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 1100.0,
                "predicted_close_p50": 102.0,
                "predicted_high": 105.0,
                "predicted_low": 99.0,
                "maxdiffprofit_profit": 0.04,
                "maxdiff_avg_daily_return": 0.015,
                "pctdiff_profit": 0.02,
                "pctdiff_avg_daily_return": 0.01,
            },
            {
                "timestamp": "2026-04-10T00:00:00Z",
                "open": 101.0,
                "high": 103.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1200.0,
                "predicted_close_p50": 102.5,
                "predicted_high": 104.0,
                "predicted_low": 98.0,
                "maxdiffprofit_profit": 0.05,
                "maxdiff_avg_daily_return": 0.02,
                "pctdiff_profit": 0.03,
                "pctdiff_avg_daily_return": 0.015,
            },
        ]
    )


def test_build_wide_gemini_prompt_contains_chronos_and_strategy_context():
    candidate = _candidate("AAPL")
    frame = _frame()

    prompt = build_wide_gemini_prompt(
        candidate=candidate,
        history=frame,
        current_row=frame.iloc[-1].to_dict(),
        rank=1,
        top_k=6,
    )

    assert "chronos_entry_low=99.00" in prompt
    assert "CURRENT DAY STRATEGY SNAPSHOT" in prompt
    assert "RECENT SELECTED STRATEGY HISTORY" in prompt
    assert "maxdiff" in prompt
    assert "allocation_pct must be from 0 to 50" in prompt


def test_refine_daily_candidates_with_gemini_adjusts_and_skips(monkeypatch, tmp_path: Path):
    planner = WidePlannerConfig(top_k=2)
    candidates = [
        _candidate("AAPL", score=0.05),
        _candidate("MSFT", score=0.04),
    ]
    frames = {"AAPL": _frame(), "MSFT": _frame()}
    plans = [
        GeminiWidePlan(action="buy", buy_price=97.5, sell_price=103.5, allocation_pct=25.0, confidence=0.8, reasoning="better entry"),
        GeminiWidePlan(action="skip", buy_price=0.0, sell_price=0.0, allocation_pct=0.0, confidence=0.9, reasoning="skip"),
    ]

    def _fake_call(prompt: str, *, model: str, api_key=None, fallback_plan=None, temperature=0.2):
        return plans.pop(0)

    monkeypatch.setattr("trade_stock_wide.gemini_overlay.call_gemini_wide", _fake_call)

    refined, stats = refine_daily_candidates_with_gemini(
        candidates,
        account_equity=10_000.0,
        planner=planner,
        backtests_by_symbol=frames,
        cache_root=tmp_path,
        model="gemini-3.1-flash-lite-preview",
        min_confidence=0.35,
    )

    assert len(refined) == 1
    assert refined[0].symbol == "AAPL"
    assert refined[0].entry_price == pytest.approx(97.5)
    assert refined[0].take_profit_price == pytest.approx(103.5)
    assert refined[0].allocation_fraction_of_equity == pytest.approx(0.25)
    assert stats.adjusted_count == 1
    assert stats.skipped_count == 1


def test_call_gemini_wide_http_fallback(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"action":"buy","buy_price":"97.25","sell_price":"103.75","allocation_pct":"22.5","confidence":"0.81","reasoning":"fallback ok"}'
                                }
                            ]
                        }
                    }
                ]
            }

    monkeypatch.setattr("trade_stock_wide.gemini_overlay.genai", None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("trade_stock_wide.gemini_overlay.requests.post", lambda *args, **kwargs: _FakeResponse())

    plan = call_gemini_wide("prompt", model="gemini-3.1-flash-lite-preview")

    assert plan == GeminiWidePlan(
        action="buy",
        buy_price=97.25,
        sell_price=103.75,
        allocation_pct=22.5,
        confidence=0.81,
        reasoning="fallback ok",
    )


def test_run_main_compare_submits_gemini_plan_when_overlay_wins(monkeypatch, capsys):
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-10T00:00:00Z",
                "open": 100.0,
                "high": 103.0,
                "low": 98.0,
                "close": 100.0,
                "predicted_high": 104.0,
                "predicted_low": 98.0,
                "maxdiffprofit_profit": 0.05,
                "maxdiff_avg_daily_return": 0.02,
                "maxdiffprofit_low_price": 99.0,
                "maxdiffprofit_high_price": 103.0,
            }
        ]
    )

    monkeypatch.setattr(run_mod, "load_backtests", lambda *args, **kwargs: {"AAPL": frame})
    monkeypatch.setattr(run_mod, "release_model_resources", lambda force=True: None)

    base_candidate = _candidate("AAPL", entry_price=99.0, take_profit_price=103.0)
    gemini_candidate = _candidate("AAPL", entry_price=95.0, take_profit_price=104.0)

    monkeypatch.setattr(run_mod, "build_daily_candidates", lambda *args, **kwargs: [base_candidate])
    monkeypatch.setattr(
        run_mod,
        "refine_daily_candidates_with_gemini",
        lambda *args, **kwargs: ([gemini_candidate], SimpleNamespace(prompt_count=1, cache_hits=0, adjusted_count=1, skipped_count=0, invalid_count=0)),
    )
    monkeypatch.setattr(
        run_mod,
        "refine_candidate_days_with_gemini",
        lambda *args, **kwargs: ([[gemini_candidate]], SimpleNamespace(prompt_count=1, cache_hits=0, adjusted_count=1, skipped_count=0, invalid_count=0)),
    )

    def _fake_simulate(days, *, starting_equity, config):
        first = days[0][0]
        monthly = 0.30 if first.entry_price < 97.0 else 0.10
        return SimpleNamespace(
            total_pnl=100.0,
            total_return=0.10,
            monthly_return=monthly,
            max_drawdown=-0.01,
            filled_count=1,
            trade_count=1,
            day_results=tuple(),
        )

    monkeypatch.setattr(run_mod, "simulate_wide_strategy", _fake_simulate)

    submitted_payloads: list[dict[str, object]] = []

    class _FakeTradingServerClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def claim_writer(self, *, ttl_seconds=None):
            return {"account": "test-paper", "session_id": "s", "expires_at": "later"}

        def heartbeat_writer(self, *, ttl_seconds=None):
            return {"account": "test-paper", "session_id": "s", "expires_at": "later"}

        def submit_limit_order(self, **kwargs):
            submitted_payloads.append(kwargs)
            return {"order": kwargs, "quote": None, "filled": False}

    monkeypatch.setattr(run_mod, "TradingServerClient", _FakeTradingServerClient)

    exit_code = run_mod.main(
        [
            "--symbols",
            "AAPL",
            "--account-equity",
            "10000",
            "--backtest-days",
            "1",
            "--daily-only-replay",
            "--submit-plan",
            "--gemini-overlay-mode",
            "compare",
            "--trading-execution-mode",
            "paper",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay compare base_monthly=+10.00% gemini_monthly=+30.00%" in captured.out
    assert submitted_payloads[0]["limit_price"] == pytest.approx(95.0)
