from __future__ import annotations

import json
import marketsimulator
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pufferlib_market.inference import TradingSignal
from pufferlib_market import validate_marketsim as validate


def _make_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    rows: list[dict[str, object]] = []
    for symbol, base in (("AAPL", 100.0), ("MSFT", 200.0)):
        for idx, ts in enumerate(timestamps):
            close = base + idx
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000.0 + idx,
                }
            )
    return pd.DataFrame(rows)


def _make_daily_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="D")
    rows: list[dict[str, object]] = []
    for symbol, base in (("AAPL", 100.0), ("MSFT", 200.0)):
        for idx, ts in enumerate(timestamps):
            close = base + idx
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": close - 1.0,
                    "high": close + 1.5,
                    "low": close - 1.5,
                    "close": close,
                    "volume": 10_000.0 + idx,
                }
            )
    return pd.DataFrame(rows)


def test_marketsimulator_package_exports_legacy_shared_cash_api() -> None:
    assert marketsimulator.SimulationConfig.__name__ == "SimulationConfig"
    assert callable(marketsimulator.run_shared_cash_simulation)


def test_load_trader_for_timeframe_selects_daily_and_forces_long_only(monkeypatch) -> None:
    seen: list[tuple[str, object, object, object, object]] = []

    def _fake_hourly(checkpoint, device, long_only, symbols):  # noqa: ANN001
        seen.append(("hourly", checkpoint, device, long_only, tuple(symbols)))
        return object()

    def _fake_daily(checkpoint, device, long_only, symbols):  # noqa: ANN001
        seen.append(("daily", checkpoint, device, long_only, tuple(symbols)))
        return object()

    monkeypatch.setattr(validate, "PPOTrader", _fake_hourly)
    monkeypatch.setattr(validate, "DailyPPOTrader", _fake_daily)

    validate._load_trader_for_timeframe(
        checkpoint="daily.pt",
        symbols=["AAPL", "MSFT"],
        timeframe="daily",
        device="cpu",
        long_only=False,
    )
    validate._load_trader_for_timeframe(
        checkpoint="hourly.pt",
        symbols=["AAPL"],
        timeframe="hourly",
        device="cuda",
        long_only=True,
    )

    assert seen == [
        ("daily", "daily.pt", "cpu", True, ("AAPL", "MSFT")),
        ("hourly", "hourly.pt", "cuda", True, ("AAPL",)),
    ]


def test_compute_hourly_feature_history_matches_legacy_snapshots() -> None:
    bars = _make_bars()
    extra_timestamps = pd.date_range("2026-01-01T04:00:00Z", periods=116, freq="h")
    extra_rows: list[dict[str, object]] = []
    for symbol, base in (("AAPL", 104.0), ("MSFT", 204.0)):
        for idx, ts in enumerate(extra_timestamps, start=4):
            close = base + idx + np.sin(idx / 3.0)
            extra_rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 0.75,
                    "low": close - 0.75,
                    "close": close,
                    "volume": 1_250.0 + idx * 2.0,
                }
            )
    frame = (
        pd.concat([bars, pd.DataFrame(extra_rows)], ignore_index=True)
        .query("symbol == 'AAPL'")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    feature_history = validate._compute_hourly_feature_history(frame[["open", "high", "low", "close", "volume"]])

    for idx in (0, 1, 4, 24, 72, 75, 95, len(frame) - 1):
        expected = validate.compute_hourly_feature_snapshot(frame.iloc[: idx + 1])
        actual = feature_history.iloc[idx].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6, equal_nan=True)


def test_align_symbol_frames_rejects_disjoint_timestamps() -> None:
    bars = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-02T00:00:00Z"),
                "symbol": "MSFT",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.5,
                "volume": 2_000.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="No common timestamps"):
        validate._align_symbol_frames(bars, symbols=["AAPL", "MSFT"])


def test_generate_policy_actions_rotates_single_position(monkeypatch) -> None:
    class _FakeTrader:
        def __init__(self) -> None:
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.cash = 0.0
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.current_position = None
            self.step = 0
            self.hold_hours = 0

            self._signals = iter(
                [
                    TradingSignal("long_AAPL", "AAPL", "long", 0.9, 1.0, 1.0, 0.0),
                    TradingSignal("long_AAPL", "AAPL", "long", 0.8, 1.0, 1.0, 0.0),
                    TradingSignal("long_MSFT", "MSFT", "long", 0.7, 1.0, 0.5, 0.0),
                ]
            )

        def get_signal(self, features, prices):  # noqa: ANN001, ARG002
            return next(self._signals)

    monkeypatch.setattr(
        validate,
        "_load_trader_for_timeframe",
        lambda **kwargs: _FakeTrader(),
    )
    monkeypatch.setattr(
        validate,
        "_build_feature_history",
        lambda aligned_frames, timeframe: {
            sym: pd.DataFrame(np.zeros((len(frame), 16), dtype=np.float32))
            for sym, frame in aligned_frames.items()
        },
    )

    actions, stats = validate._generate_policy_actions(
        bars=_make_bars(),
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        timeframe="hourly",
        device="cpu",
        long_only=False,
        min_history_bars=1,
    )

    assert stats["action_timestamps"] == 3
    assert stats["buy_rows"] == 2
    assert stats["sell_rows"] == 1
    assert stats["short_signals_flattened"] == 0

    first = actions[actions["timestamp"] == pd.Timestamp("2026-01-01T01:00:00Z")].set_index("symbol")
    assert first.loc["AAPL", "buy_amount"] == pytest.approx(1.0)
    assert first.loc["MSFT", "buy_amount"] == pytest.approx(0.0)

    second = actions[actions["timestamp"] == pd.Timestamp("2026-01-01T02:00:00Z")].set_index("symbol")
    assert second.loc["AAPL", "buy_amount"] == pytest.approx(0.0)
    assert second.loc["AAPL", "sell_amount"] == pytest.approx(0.0)

    third = actions[actions["timestamp"] == pd.Timestamp("2026-01-01T03:00:00Z")].set_index("symbol")
    assert third.loc["AAPL", "sell_amount"] == pytest.approx(1.0)
    assert third.loc["MSFT", "buy_amount"] == pytest.approx(0.5)


def test_generate_policy_actions_flattens_short_signals(monkeypatch) -> None:
    class _FakeTrader:
        def __init__(self) -> None:
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.cash = 0.0
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.current_position = None
            self.step = 0
            self.hold_hours = 0

            self._signals = iter(
                [
                    TradingSignal("long_AAPL", "AAPL", "long", 0.9, 1.0, 1.0, 0.0),
                    TradingSignal("short_AAPL", "AAPL", "short", 0.8, 1.0, 1.0, 0.0),
                    TradingSignal("flat", None, None, 0.7, 1.0, 0.0, 0.0),
                ]
            )

        def get_signal(self, features, prices):  # noqa: ANN001, ARG002
            return next(self._signals)

    monkeypatch.setattr(
        validate,
        "_load_trader_for_timeframe",
        lambda **kwargs: _FakeTrader(),
    )
    monkeypatch.setattr(
        validate,
        "_build_feature_history",
        lambda aligned_frames, timeframe: {
            sym: pd.DataFrame(np.zeros((len(frame), 16), dtype=np.float32))
            for sym, frame in aligned_frames.items()
        },
    )

    actions, stats = validate._generate_policy_actions(
        bars=_make_bars(),
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        timeframe="hourly",
        device="cpu",
        long_only=False,
        min_history_bars=1,
    )

    assert stats["short_signals_flattened"] == 1
    flattened = actions[actions["timestamp"] == pd.Timestamp("2026-01-01T02:00:00Z")].set_index("symbol")
    assert flattened.loc["AAPL", "sell_amount"] == pytest.approx(1.0)
    assert flattened.loc["AAPL", "buy_amount"] == pytest.approx(0.0)
    assert flattened.loc["MSFT", "buy_amount"] == pytest.approx(0.0)


def test_generate_policy_actions_daily_tracks_hold_days(monkeypatch) -> None:
    class _FakeDailyTrader:
        def __init__(self) -> None:
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.cash = 0.0
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.current_position = None
            self.step = 0
            self.hold_hours = 0
            self.hold_days = 0
            self.hold_days_seen: list[int] = []
            self._signals = iter(
                [
                    TradingSignal("long_AAPL", "AAPL", "long", 0.9, 1.0, 1.0, 0.0),
                    TradingSignal("long_AAPL", "AAPL", "long", 0.8, 1.0, 1.0, 0.0),
                    TradingSignal("flat", None, None, 0.7, 1.0, 0.0, 0.0),
                ]
            )

        def get_signal(self, features, prices):  # noqa: ANN001, ARG002
            self.hold_days_seen.append(self.hold_days)
            return next(self._signals)

    trader = _FakeDailyTrader()
    monkeypatch.setattr(
        validate,
        "_load_trader_for_timeframe",
        lambda **kwargs: trader,
    )
    monkeypatch.setattr(
        validate,
        "_build_feature_history",
        lambda aligned_frames, timeframe: {
            sym: pd.DataFrame(np.zeros((len(frame), 16), dtype=np.float32))
            for sym, frame in aligned_frames.items()
        },
    )

    actions, stats = validate._generate_policy_actions(
        bars=_make_daily_bars(),
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        timeframe="daily",
        device="cpu",
        long_only=True,
        min_history_bars=1,
    )

    # C env hold_hours convention: open_long() resets hold_hours=0, and the
    # first obs AFTER buying still sees hold_hours=0 (obs built before the HOLD
    # action that would increment it).  So:
    #   step 1 (buy): hold_days=0 (flat before signal)
    #   step 2 (hold): hold_days=0 (just bought on step 1; hold_hours still 0)
    #   step 3 (sell): hold_days=1 (held once on step 2)
    assert trader.hold_days_seen == [0, 0, 1]
    assert stats["buy_rows"] == 1
    assert stats["sell_rows"] == 1
    first = actions[actions["timestamp"] == pd.Timestamp("2026-01-02T00:00:00Z")].set_index("symbol")
    assert first.loc["AAPL", "buy_amount"] == pytest.approx(1.0)
    third = actions[actions["timestamp"] == pd.Timestamp("2026-01-04T00:00:00Z")].set_index("symbol")
    assert third.loc["AAPL", "sell_amount"] == pytest.approx(1.0)


def test_generate_policy_actions_rejects_insufficient_history(monkeypatch) -> None:
    monkeypatch.setattr(validate, "_load_trader_for_timeframe", lambda **kwargs: object())
    monkeypatch.setattr(
        validate,
        "_build_feature_history",
        lambda aligned_frames, timeframe: {
            sym: pd.DataFrame(np.zeros((len(frame), 16), dtype=np.float32))
            for sym, frame in aligned_frames.items()
        },
    )

    with pytest.raises(ValueError, match="Need more aligned history"):
        validate._generate_policy_actions(
            bars=_make_bars(),
            checkpoint="unused.pt",
            symbols=["AAPL", "MSFT"],
            timeframe="hourly",
            device="cpu",
            long_only=False,
            min_history_bars=4,
        )


def test_prepare_action_inputs_stacks_feature_and_price_matrices() -> None:
    bars = _make_bars()
    aligned_frames = validate._align_symbol_frames(bars, symbols=["AAPL", "MSFT"])
    feature_history = {
        symbol: pd.DataFrame(
            np.full((len(frame), 16), 1.0 if symbol == "AAPL" else 2.0, dtype=np.float32)
        )
        for symbol, frame in aligned_frames.items()
    }

    feature_cube, close_matrix, timestamps = validate._prepare_action_inputs(
        aligned_frames,
        feature_history,
        symbols=["AAPL", "MSFT"],
    )

    assert feature_cube.shape == (4, 2, 16)
    assert np.all(feature_cube[:, 0, :] == 1.0)
    assert np.all(feature_cube[:, 1, :] == 2.0)
    assert close_matrix.shape == (4, 2)
    assert close_matrix[0].tolist() == pytest.approx([100.0, 200.0])
    assert timestamps[0] == pd.Timestamp("2026-01-01T00:00:00Z")


def test_generate_policy_actions_uses_precomputed_arrays(monkeypatch) -> None:
    class _ExplodingFrame:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, key):  # noqa: ANN001
            raise AssertionError(f"aligned frame access should be avoided after precompute: {key}")

    seen_prices: list[dict[str, float]] = []

    class _FakeTrader:
        def __init__(self) -> None:
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.cash = 0.0
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.current_position = None
            self.step = 0
            self.hold_hours = 0

        def get_signal(self, features, prices):  # noqa: ANN001
            assert features.shape == (2, 16)
            seen_prices.append(dict(prices))
            return TradingSignal("flat", None, None, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(validate, "_align_symbol_frames", lambda bars, symbols: {"AAPL": _ExplodingFrame(), "MSFT": _ExplodingFrame()})
    monkeypatch.setattr(validate, "_load_trader_for_timeframe", lambda **kwargs: _FakeTrader())
    monkeypatch.setattr(validate, "_build_feature_history", lambda aligned_frames, timeframe: {})
    monkeypatch.setattr(
        validate,
        "_prepare_action_inputs",
        lambda aligned_frames, feature_history, symbols: (
            np.zeros((4, 2, 16), dtype=np.float32),
            np.asarray(
                [
                    [100.0, 200.0],
                    [101.0, 201.0],
                    [102.0, 202.0],
                    [103.0, 203.0],
                ],
                dtype=np.float64,
            ),
            tuple(pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")),
        ),
    )

    actions, stats = validate._generate_policy_actions(
        bars=_make_bars(),
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        timeframe="hourly",
        device="cpu",
        long_only=False,
        min_history_bars=1,
    )

    assert stats["action_timestamps"] == 3
    assert len(actions) == 6
    assert seen_prices == [
        {"AAPL": 101.0, "MSFT": 201.0},
        {"AAPL": 102.0, "MSFT": 202.0},
        {"AAPL": 103.0, "MSFT": 203.0},
    ]


def test_parse_args_supports_json_only_and_output_json() -> None:
    args = validate.parse_args(
        [
            "--checkpoint",
            "demo.pt",
            "--symbols",
            "AAPL,MSFT",
            "--json-only",
            "--output-json",
            "summary.json",
        ]
    )

    assert args.checkpoint == "demo.pt"
    assert args.symbols == "AAPL,MSFT"
    assert args.json_only is True
    assert args.output_json == "summary.json"


def test_main_json_only_emits_only_summary_json(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(validate, "load_hourly_bars", lambda symbol: _make_bars().query("symbol == @symbol").copy())
    monkeypatch.setattr(
        validate,
        "_generate_policy_actions",
        lambda **kwargs: (
            pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp("2026-01-01T01:00:00Z"),
                        "symbol": "AAPL",
                        "buy_price": 101.0,
                        "sell_price": 0.0,
                        "buy_amount": 1.0,
                        "sell_amount": 0.0,
                    }
                ]
            ),
            {
                "history_bars": 1,
                "aligned_timestamps": 4,
                "action_timestamps": 1,
                "buy_rows": 1,
                "sell_rows": 0,
                "short_signals_flattened": 0,
            },
        ),
    )
    monkeypatch.setattr(
        validate,
        "_load_marketsimulator_api",
        lambda: (
            marketsimulator.SimulationConfig,
            lambda bars, actions, config: type(
                "_Result",
                (),
                {
                    "metrics": {"total_return": 0.02, "sortino": 1.5},
                    "per_symbol": {"AAPL": type("_SymbolResult", (), {"trades": [object(), object()]})()},
                    "combined_equity": [1.0, 1.02],
                },
            )(),
        ),
    )

    output_path = tmp_path / "summary.json"
    exit_code = validate.main(
        [
            "--checkpoint",
            "demo.pt",
            "--symbols",
            "AAPL",
            "--json-only",
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"] == "demo.pt"
    assert payload["metrics"]["total_return"] == pytest.approx(0.02)
    assert payload["metrics"]["trades"] == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_set_trader_shadow_state_uses_hold_bars_not_step_index() -> None:
    """trader.step must be clamped hold_bars (not the overall bar index).

    Bug: line 165 used `step_index` (e.g. 100) for obs[base+4]=step/max_steps.
    Production sets trader.step = min(hold_bars, max_steps).  A freshly-opened
    position at overall bar 100 with hold_bars=0 must show step=0, not 100.
    """
    from pufferlib_market.validate_marketsim import _set_trader_shadow_state

    class _FakeTrader:
        cash = 0.0
        position_qty = 0.0
        entry_price = 0.0
        current_position = None
        step = 0
        hold_hours = 0
        hold_days = 0
        max_steps = 252

    trader = _FakeTrader()
    symbol_to_index = {"AAPL": 0}

    # Simulate: holding AAPL, hold_bars=3, but overall step_index=100
    _set_trader_shadow_state(
        trader,
        symbol_to_index=symbol_to_index,
        held_symbol="AAPL",
        hold_bars=3,
        entry_price=150.0,
        step_index=100,
    )

    # step must match hold_bars (3), NOT the overall bar index (100)
    assert trader.step == 3
    assert trader.hold_hours == 3

    # Flat: hold_bars=0, step_index=200 — step must be 0
    _set_trader_shadow_state(
        trader,
        symbol_to_index=symbol_to_index,
        held_symbol=None,
        hold_bars=0,
        entry_price=0.0,
        step_index=200,
    )
    assert trader.step == 0

    # Clamping: hold_bars > max_steps must be clamped
    _set_trader_shadow_state(
        trader,
        symbol_to_index=symbol_to_index,
        held_symbol="AAPL",
        hold_bars=300,
        entry_price=150.0,
        step_index=999,
    )
    assert trader.step == 252  # clamped to max_steps
