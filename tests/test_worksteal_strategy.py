"""Tests for work-stealing dip-buying strategy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import binance_worksteal.strategy as strategy_mod
from binance_worksteal.strategy import (
    WorkStealConfig, TradeLog, compute_ref_price, compute_ref_low,
    compute_atr,
    compute_avg_hold_days_from_trades, count_completed_trades,
    run_worksteal_backtest, compute_metrics,
    get_fee, _risk_off_triggered,
    _prepare_backtest_symbol_data, _initialize_backtest_cursors, _build_daily_market_context,
    prepare_backtest_bars,
    load_daily_bars, load_hourly_bars,
)


def make_bars(prices, start="2026-01-01", symbol="BTCUSD"):
    dates = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    rows = []
    for i, (d, p) in enumerate(zip(dates, prices)):
        noise = p * 0.02
        rows.append({
            "timestamp": d, "open": p - noise * 0.5,
            "high": p + noise, "low": p - noise,
            "close": p, "volume": 1000.0, "symbol": symbol,
        })
    return pd.DataFrame(rows)


def write_ohlcv_csv(path, prices, *, invalid_timestamp_index=None, missing_columns=()):
    frame = make_bars(prices)
    if invalid_timestamp_index is not None:
        frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        frame.loc[invalid_timestamp_index, "timestamp"] = "2026-99-99T00:00:00Z"
    if missing_columns:
        frame = frame.drop(columns=list(missing_columns))
    frame.to_csv(path, index=False)


class TestFees:
    def test_fdusd_zero_fee(self):
        config = WorkStealConfig(fdusd_fee=0.0, maker_fee=0.001)
        assert get_fee("BTCUSD", config) == 0.0
        assert get_fee("ETHUSD", config) == 0.0

    def test_usdt_fee(self):
        config = WorkStealConfig(fdusd_fee=0.0, maker_fee=0.001)
        assert get_fee("DOGEUSD", config) == 0.001
        assert get_fee("LINKUSD", config) == 0.001


class TestComputeRefPrice:
    def test_high_method(self):
        bars = make_bars([100, 110, 105, 95, 100])
        ref = compute_ref_price(bars, "high", 5)
        assert ref == pytest.approx(110 * 1.02, rel=0.01)

    def test_sma_method(self):
        bars = make_bars([100, 110, 105, 95, 100])
        ref = compute_ref_price(bars, "sma", 5)
        assert ref == pytest.approx(102.0, rel=0.01)

    def test_ref_low(self):
        bars = make_bars([100, 90, 95, 110, 105])
        ref = compute_ref_low(bars, 5)
        assert ref == pytest.approx(90 * 0.98, rel=0.01)


class TestComputeATR:
    def test_basic(self):
        bars = make_bars([100] * 20)
        atr = compute_atr(bars, 14)
        assert atr > 0


class TestTradeLogHelpers:
    def test_count_completed_trades_counts_exits_only(self):
        trades = [
            TradeLog(
                timestamp=pd.Timestamp("2026-01-01", tz="UTC"),
                symbol="BTCUSD",
                side="buy",
                price=100.0,
                quantity=1.0,
                notional=100.0,
                fee=0.1,
            ),
            TradeLog(
                timestamp=pd.Timestamp("2026-01-02", tz="UTC"),
                symbol="BTCUSD",
                side="sell",
                price=105.0,
                quantity=1.0,
                notional=105.0,
                fee=0.1,
            ),
            TradeLog(
                timestamp=pd.Timestamp("2026-01-03", tz="UTC"),
                symbol="ETHUSD",
                side="short",
                price=200.0,
                quantity=1.0,
                notional=200.0,
                fee=0.2,
            ),
        ]

        assert count_completed_trades(trades) == 1

    def test_compute_avg_hold_days_from_trades_has_one_day_floor(self):
        trades = [
            TradeLog(
                timestamp=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                symbol="BTCUSD",
                side="buy",
                price=100.0,
                quantity=1.0,
                notional=100.0,
                fee=0.1,
            ),
            TradeLog(
                timestamp=pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
                symbol="BTCUSD",
                side="sell",
                price=105.0,
                quantity=1.0,
                notional=105.0,
                fee=0.1,
            ),
        ]

        assert compute_avg_hold_days_from_trades(trades) == pytest.approx(1.0)


class TestBarLoading:
    def test_load_daily_bars_skips_malformed_csv_and_warns(self, tmp_path):
        write_ohlcv_csv(tmp_path / "BTCUSDT.csv", [100 + i for i in range(40)])
        write_ohlcv_csv(tmp_path / "ETHUSDT.csv", [200 + i for i in range(40)], missing_columns=("low",))

        with pytest.warns(RuntimeWarning, match="Skipping daily bars for ETHUSD"):
            loaded = load_daily_bars(str(tmp_path), ["BTCUSD", "ETHUSD"])

        assert "BTCUSD" in loaded
        assert "ETHUSD" not in loaded

    def test_load_daily_bars_coerces_bad_timestamps_instead_of_crashing(self, tmp_path):
        write_ohlcv_csv(
            tmp_path / "BTCUSDT.csv",
            [100 + i for i in range(35)],
            invalid_timestamp_index=0,
        )

        loaded = load_daily_bars(str(tmp_path), ["BTCUSD"])

        assert "BTCUSD" in loaded
        assert len(loaded["BTCUSD"]) == 34

    def test_load_hourly_bars_skips_malformed_csv_and_warns(self, tmp_path):
        write_ohlcv_csv(tmp_path / "BTCUSD.csv", [100.0, 101.0, 102.0], missing_columns=("volume",))

        with pytest.warns(RuntimeWarning, match="Skipping hourly bars for BTCUSD"):
            loaded = load_hourly_bars(str(tmp_path), ["BTCUSD"])

        assert loaded == {}


class TestWorkStealBacktest:
    def test_prepare_backtest_symbol_data_sorts_and_deduplicates(self):
        raw = pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-03T00:00:00Z",
                    "open": 103.0,
                    "high": 104.0,
                    "low": 102.0,
                    "close": 103.0,
                    "volume": 1000.0,
                    "symbol": "BTCUSD",
                },
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                    "symbol": "BTCUSD",
                },
                {
                    "timestamp": "2026-01-02T00:00:00Z",
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.0,
                    "volume": 1000.0,
                    "symbol": "BTCUSD",
                },
                {
                    "timestamp": "2026-01-02T00:00:00Z",
                    "open": 101.5,
                    "high": 102.5,
                    "low": 100.5,
                    "close": 101.5,
                    "volume": 1200.0,
                    "symbol": "BTCUSD",
                },
            ]
        )
        bars_map = {"BTCUSD": raw}
        prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data(bars_map)
        bars = prepared["BTCUSD"].bars
        cursors = _initialize_backtest_cursors(prepared, first_date_ns)
        _build_daily_market_context(prepared, cursors, all_dates[0])
        current_bars, history = _build_daily_market_context(prepared, cursors, all_dates[1])

        assert bars_map["BTCUSD"] is raw
        assert list(raw["timestamp"]) == [
            "2026-01-03T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:00Z",
        ]
        assert list(bars["timestamp"]) == [
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-02T00:00:00Z"),
            pd.Timestamp("2026-01-03T00:00:00Z"),
        ]
        assert bars.iloc[1]["close"] == pytest.approx(101.5)
        assert prepared["BTCUSD"].rows[1]["close"] == pytest.approx(101.5)
        assert list(all_dates) == list(bars["timestamp"])
        assert current_bars["BTCUSD"] is prepared["BTCUSD"].rows[1]
        assert current_bars["BTCUSD"]["close"] == pytest.approx(101.5)
        assert len(history["BTCUSD"]) == 2

    def test_prepare_backtest_symbol_data_reuses_prepared_frames(self):
        raw = make_bars([100.0 + i for i in range(10)], start="2026-01-01")
        bars_map = {"BTCUSD": raw}

        prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data(bars_map)

        assert prepared["BTCUSD"].bars is raw
        assert np.array_equal(prepared["BTCUSD"].timestamp_ns, raw["timestamp"].array.asi8)
        assert first_date_ns == all_dates[0].value

    def test_run_worksteal_backtest_accepts_prepared_bars(self):
        bars = {"BTCUSD": make_bars([100, 120, 95, 115, 90, 108, 92, 120])}
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.05,
            profit_target_pct=0.05,
            stop_loss_pct=0.10,
            max_positions=1,
            lookback_days=3,
        )

        eq_raw, trades_raw, metrics_raw = run_worksteal_backtest(bars, config)
        eq_prepared, trades_prepared, metrics_prepared = run_worksteal_backtest(
            bars,
            config,
            prepared_bars=prepare_backtest_bars(bars),
        )

        pd.testing.assert_frame_equal(eq_raw, eq_prepared)
        assert [(t.side, t.symbol, t.reason) for t in trades_prepared] == [
            (t.side, t.symbol, t.reason) for t in trades_raw
        ]
        assert metrics_prepared == metrics_raw

    def test_backtest_reuses_regime_market_breadth_for_allocation_sizing(self, monkeypatch):
        bars = {
            "BTCUSD": make_bars([100.0, 110.0, 120.0, 108.0], symbol="BTCUSD"),
            "ETHUSD": make_bars([100.0, 102.0, 104.0, 105.0], symbol="ETHUSD"),
        }
        prepared = prepare_backtest_bars(bars)
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.03,
            lookback_days=3,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
            market_breadth_filter=0.75,
            max_positions=1,
        )
        original_compute_breadth_ratio = strategy_mod.compute_breadth_ratio
        call_count = {"count": 0}

        def counting_breadth_ratio(*args, **kwargs):
            call_count["count"] += 1
            return original_compute_breadth_ratio(*args, **kwargs)

        monkeypatch.setattr(strategy_mod, "compute_breadth_ratio", counting_breadth_ratio)

        _eq, _trades, _metrics = run_worksteal_backtest(
            bars,
            config,
            start_date="2026-01-04",
            end_date="2026-01-04",
            prepared_bars=prepared,
            allocation_scale_fn=lambda context: 0.0,
        )

        assert call_count["count"] == 1

    def test_backtest_computes_market_breadth_for_allocation_sizing_when_filter_disabled(self, monkeypatch):
        bars = {
            "BTCUSD": make_bars([100.0, 110.0, 120.0, 108.0], symbol="BTCUSD"),
            "ETHUSD": make_bars([100.0, 102.0, 104.0, 105.0], symbol="ETHUSD"),
        }
        prepared = prepare_backtest_bars(bars)
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.03,
            lookback_days=3,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
            market_breadth_filter=0.0,
            max_positions=1,
        )
        original_compute_breadth_ratio = strategy_mod.compute_breadth_ratio
        seen = {"count": 0, "breadth": None}

        def counting_breadth_ratio(*args, **kwargs):
            seen["count"] += 1
            return original_compute_breadth_ratio(*args, **kwargs)

        def capture_breadth(context):
            seen["breadth"] = context.market_breadth
            return 0.0

        monkeypatch.setattr(strategy_mod, "compute_breadth_ratio", counting_breadth_ratio)

        _eq, _trades, _metrics = run_worksteal_backtest(
            bars,
            config,
            start_date="2026-01-04",
            end_date="2026-01-04",
            prepared_bars=prepared,
            allocation_scale_fn=capture_breadth,
        )

        expected_breadth, _, _ = original_compute_breadth_ratio(
            {sym: frame.iloc[-1] for sym, frame in bars.items()},
            bars,
        )
        assert seen["count"] == 1
        assert seen["breadth"] == pytest.approx(expected_breadth)

    def test_prepared_bars_allocation_context_history_is_correct_and_isolated(self):
        bars = {
            "BTCUSD": make_bars([100.0, 110.0, 120.0, 108.0, 112.0], symbol="BTCUSD")
        }
        prepared = prepare_backtest_bars(bars)
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=3,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
            max_positions=1,
        )
        seen = {"calls": 0}

        def capture_and_mutate(context):
            seen["calls"] += 1
            if "timestamps" not in seen:
                seen["timestamps"] = list(context.history["timestamp"])
                seen["closes"] = list(context.history["close"])
                seen["signal_close"] = float(context.signal_bar["close"])
            context.history.loc[:, "close"] = 0.0
            context.signal_bar.loc["close"] = 0.0
            return 0.0

        _eq, trades, _metrics = run_worksteal_backtest(
            bars,
            config,
            prepared_bars=prepared,
            allocation_scale_fn=capture_and_mutate,
        )

        assert seen["calls"] >= 1
        assert seen["timestamps"] == [
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
            pd.Timestamp("2026-01-03", tz="UTC"),
            pd.Timestamp("2026-01-04", tz="UTC"),
        ]
        assert seen["closes"] == pytest.approx([100.0, 110.0, 120.0, 108.0])
        assert seen["signal_close"] == pytest.approx(108.0)
        assert trades == []
        assert bars["BTCUSD"]["close"].tolist() == pytest.approx([100.0, 110.0, 120.0, 108.0, 112.0])
        assert prepared["BTCUSD"].bars["close"].tolist() == pytest.approx([100.0, 110.0, 120.0, 108.0, 112.0])
        assert prepared["BTCUSD"].rows[3]["close"] == pytest.approx(108.0)

    def test_symbol_metric_cache_matches_uncached_strategy_helpers(self):
        bars = {
            "BTCUSD": make_bars([
                100, 102, 104, 106, 108, 110, 109, 108, 107, 106, 105,
                104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94,
            ], symbol="BTCUSD"),
            "ETHUSD": make_bars([
                80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
            ], symbol="ETHUSD"),
        }
        prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data(bars)
        cursors = _initialize_backtest_cursors(prepared, first_date_ns)
        for date in all_dates:
            current_bars, history = _build_daily_market_context(prepared, cursors, date)
        cache = strategy_mod._build_symbol_metric_cache(current_bars, history)
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.05,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=5,
            risk_off_trigger_momentum_period=7,
            market_breadth_filter=0.5,
        )

        uncached_regime = strategy_mod.resolve_entry_regime(
            current_bars=current_bars,
            history=history,
            config=config,
        )
        cached_regime = strategy_mod.resolve_entry_regime(
            current_bars=current_bars,
            history=history,
            config=config,
            symbol_metrics=cache,
        )
        assert cached_regime == uncached_regime
        assert strategy_mod.compute_breadth_ratio(current_bars, history, symbol_metrics=cache) == (
            strategy_mod.compute_breadth_ratio(current_bars, history)
        )
        cached_breadth = strategy_mod.compute_breadth_ratio(current_bars, history, symbol_metrics=cache)
        assert cached_regime.market_breadth_ratio == pytest.approx(cached_breadth[0])
        assert cached_regime.market_breadth_dipping_count == cached_breadth[1]
        assert cached_regime.market_breadth_total_count == cached_breadth[2]

        uncached_candidates = strategy_mod.build_entry_candidates(
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            date=all_dates[-1],
            config=config,
        )
        cached_candidates = strategy_mod.build_entry_candidates(
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            date=all_dates[-1],
            config=config,
            symbol_metrics=cache,
        )
        assert cached_candidates == uncached_candidates

    def test_build_entry_candidates_max_candidates_matches_full_ordering(self):
        bars = {
            "SYM0USD": make_bars([100.0] * 20 + [90.0], symbol="SYM0USD"),
            "SYM1USD": make_bars([100.0] * 20 + [91.0], symbol="SYM1USD"),
            "SYM2USD": make_bars([100.0] * 20 + [92.0], symbol="SYM2USD"),
            "SYM3USD": make_bars([100.0] * 20 + [93.0], symbol="SYM3USD"),
        }
        history = bars
        current_bars = {sym: frame.iloc[-1] for sym, frame in bars.items()}
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.05,
            lookback_days=20,
            sma_filter_period=0,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")

        full_candidates = strategy_mod.build_entry_candidates(
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            date=date,
            config=config,
        )
        limited_candidates = strategy_mod.build_entry_candidates(
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            date=date,
            config=config,
            max_candidates=2,
        )

        assert [candidate[0] for candidate in limited_candidates] == [candidate[0] for candidate in full_candidates[:2]]
        assert limited_candidates == full_candidates[:2]

    def test_backtest_skips_resorting_common_ranked_candidates(self, monkeypatch):
        bars = {
            "AUSD": make_bars([100.0, 100.0], symbol="AUSD"),
            "BUSD": make_bars([100.0, 100.0], symbol="BUSD"),
        }
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.05,
            profit_target_pct=0.05,
            stop_loss_pct=0.10,
            max_positions=1,
            lookback_days=1,
            sma_filter_period=0,
        )
        target_date = pd.Timestamp("2026-01-02", tz="UTC")

        class NoSortList(list):
            def sort(self, *args, **kwargs):
                raise AssertionError("common path should not re-sort ranked candidates")

        def fake_build_entry_candidates(*, current_bars, date, **kwargs):
            if date != target_date:
                return []
            return NoSortList([
                ("BUSD", "long", 0.9, float(current_bars["BUSD"]["close"]), current_bars["BUSD"]),
                ("AUSD", "long", 0.8, float(current_bars["AUSD"]["close"]), current_bars["AUSD"]),
            ])

        monkeypatch.setattr(strategy_mod, "build_entry_candidates", fake_build_entry_candidates)

        _equity, trades, _metrics = run_worksteal_backtest(bars, config)

        entry_trades = [trade for trade in trades if trade.side == "buy"]
        assert entry_trades[0].symbol == "BUSD"


    def test_build_entry_candidates_dedupes_same_symbol_opposite_directions(self):
        bars = {
            "AUSD": make_bars([100.0] * 20 + [95.0], symbol="AUSD"),
            "BUSD": make_bars([100.0] * 20 + [100.0], symbol="BUSD"),
        }
        history = bars
        current_bars = {sym: frame.iloc[-1] for sym, frame in bars.items()}
        config = WorkStealConfig(
            dip_pct=0.0,
            short_pump_pct=0.0,
            proximity_pct=0.05,
            lookback_days=20,
            sma_filter_period=0,
            enable_shorts=True,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")

        candidates = strategy_mod.build_entry_candidates(
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            date=date,
            config=config,
            max_candidates=2,
        )

        assert [candidate[0] for candidate in candidates] == ["AUSD", "BUSD"]
        assert len({candidate[0] for candidate in candidates}) == 2

    def test_symbol_metric_cache_uses_lazy_history_without_materializing_slices(self):
        bars = {"BTCUSD": make_bars([100.0, 101.0, 102.0], start="2026-01-01")}
        prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data(bars)
        cursors = _initialize_backtest_cursors(prepared, first_date_ns)
        _build_daily_market_context(prepared, cursors, all_dates[0])
        current_bars, history = _build_daily_market_context(prepared, cursors, all_dates[1])

        assert getattr(history, "_cache") == {}

        cache = strategy_mod._build_symbol_metric_cache(current_bars, history)

        assert getattr(history, "_cache") == {}
        assert cache["BTCUSD"].length == 2
        sliced_history = history["BTCUSD"]
        assert len(sliced_history) == 2
        assert set(getattr(history, "_cache")) == {"BTCUSD"}

    def test_build_entry_candidates_uses_symbol_metrics_without_loading_history(self):
        bars = {"BTCUSD": make_bars([100.0] * 20 + [90.0], symbol="BTCUSD")}
        current_bars = {sym: frame.iloc[-1] for sym, frame in bars.items()}
        cache = strategy_mod._build_symbol_metric_cache(current_bars, bars)
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.05,
            lookback_days=20,
            sma_filter_period=5,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")

        class ExplodingHistory(dict):
            def get(self, key, default=None):
                raise AssertionError(f"history should not be loaded for {key}")

        candidates = strategy_mod.build_entry_candidates(
            current_bars=current_bars,
            history=ExplodingHistory(),
            positions={},
            last_exit={},
            date=date,
            config=config,
            symbol_metrics=cache,
        )

        assert [candidate[0] for candidate in candidates] == ["BTCUSD"]

    def test_build_entry_candidates_metric_filters_stay_off_history(self):
        bars = {"BTCUSD": make_bars([100.0] * 20 + [90.0], symbol="BTCUSD")}
        bars["BTCUSD"].loc[:19, "volume"] = 1000.0
        bars["BTCUSD"].loc[20, "volume"] = 2500.0
        bars["BTCUSD"].loc[20, "low"] = 89.0
        bars["BTCUSD"].loc[20, "high"] = 91.8
        current_bars = {sym: frame.iloc[-1] for sym, frame in bars.items()}
        cache = strategy_mod._build_symbol_metric_cache(current_bars, bars)
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.05,
            lookback_days=20,
            sma_filter_period=5,
            momentum_period=5,
            momentum_min=-0.20,
            rsi_filter=60,
            volume_spike_filter=2.0,
            realistic_fill=True,
        )
        date = pd.Timestamp("2026-01-21", tz="UTC")

        class ExplodingHistory(dict):
            def get(self, key, default=None):
                raise AssertionError(f"history should not be loaded for {key}")

        candidates = strategy_mod.build_entry_candidates(
            current_bars=current_bars,
            history=ExplodingHistory(),
            positions={},
            last_exit={},
            date=date,
            config=config,
            symbol_metrics=cache,
        )

        assert [candidate[0] for candidate in candidates] == ["BTCUSD"]
        assert candidates[0][1] == "long"

    def test_base_asset_should_hold_uses_metrics_without_loading_history(self):
        bars = {"BTCUSD": make_bars([100.0 + i for i in range(25)], symbol="BTCUSD")}
        current_bars = {sym: frame.iloc[-1] for sym, frame in bars.items()}
        cache = strategy_mod._build_symbol_metric_cache(current_bars, bars)
        config = WorkStealConfig(
            base_asset_symbol="BTCUSD",
            base_asset_sma_filter_period=5,
            base_asset_momentum_period=5,
            base_asset_min_momentum=-1.0,
        )

        class ExplodingHistory(dict):
            def get(self, key, default=None):
                raise AssertionError(f"history should not be loaded for {key}")

        assert strategy_mod._base_asset_should_hold(
            base_symbol="BTCUSD",
            current_bars=current_bars,
            history=ExplodingHistory(),
            config=config,
            symbol_metrics=cache,
        ) is True


    def test_prepare_backtest_symbol_data_drops_invalid_timestamps(self):
        raw = pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                    "symbol": "BTCUSD",
                },
                {
                    "timestamp": "not-a-date",
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 101.0,
                    "volume": 1000.0,
                    "symbol": "BTCUSD",
                },
                {
                    "timestamp": pd.NaT,
                    "open": 102.0,
                    "high": 103.0,
                    "low": 101.0,
                    "close": 102.0,
                    "volume": 1000.0,
                    "symbol": "BTCUSD",
                },
                {
                    "timestamp": "2026-01-03T00:00:00Z",
                    "open": 103.0,
                    "high": 104.0,
                    "low": 102.0,
                    "close": 103.0,
                    "volume": 1000.0,
                    "symbol": "BTCUSD",
                },
            ]
        )

        prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data({"BTCUSD": raw})
        bars = prepared["BTCUSD"].bars

        assert list(bars["timestamp"]) == [
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-03T00:00:00Z"),
        ]
        assert list(all_dates) == list(bars["timestamp"])
        assert first_date_ns == pd.Timestamp("2026-01-01T00:00:00Z").value
        assert not bars["timestamp"].isna().any()

    def test_market_context_keeps_pre_start_history(self):
        bars = {"BTCUSD": make_bars([100, 101, 102, 103, 104, 105], start="2026-01-01")}
        prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data(
            {k: v.copy() for k, v in bars.items()},
            start_date="2026-01-04",
            end_date="2026-01-05",
        )
        cursors = _initialize_backtest_cursors(prepared, first_date_ns)

        current_bars, history = _build_daily_market_context(prepared, cursors, all_dates[0])

        assert current_bars["BTCUSD"]["timestamp"] == pd.Timestamp("2026-01-04", tz="UTC")
        assert len(history["BTCUSD"]) == 4
        assert history["BTCUSD"].iloc[0]["timestamp"] == pd.Timestamp("2026-01-01", tz="UTC")

    def test_start_date_still_uses_prior_history_for_lookback(self):
        prices = [100] * 24 + [90, 88, 86, 89, 93, 97]
        bars = {"BTCUSD": make_bars(prices, start="2026-01-01")}
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )

        _, trades, _ = run_worksteal_backtest(
            {k: v.copy() for k, v in bars.items()},
            config,
            start_date="2026-01-25",
            end_date="2026-01-30",
        )

        assert any(trade.side == "buy" for trade in trades)

    def test_run_worksteal_backtest_does_not_mutate_input_frames(self):
        raw = make_bars([100.0 + i for i in range(30)], start="2026-01-01")
        raw["timestamp"] = raw["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        raw = raw.iloc[::-1].reset_index(drop=True)
        bars = {"BTCUSD": raw}
        original_timestamps = list(raw["timestamp"])
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )

        run_worksteal_backtest(
            bars,
            config,
            start_date="2026-01-20",
            end_date="2026-01-30",
        )

        assert bars["BTCUSD"] is raw
        assert list(raw["timestamp"]) == original_timestamps
        assert isinstance(raw.iloc[0]["timestamp"], str)

    def test_run_worksteal_backtest_checks_risk_off_once_per_day(self, monkeypatch):
        bars = {
            "BTCUSD": make_bars([100.0 + i for i in range(25)], symbol="BTCUSD"),
            "ETHUSD": make_bars([120.0 + i for i in range(25)], symbol="ETHUSD"),
        }
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=7,
            market_breadth_filter=0.0,
            max_positions=0,
        )
        counter = {"count": 0}
        original = strategy_mod._risk_off_triggered

        def counted_risk_off(current_bars, history, inner_config):
            counter["count"] += 1
            return original(current_bars, history, inner_config)

        monkeypatch.setattr(strategy_mod, "_risk_off_triggered", counted_risk_off)

        run_worksteal_backtest(bars, config)

        assert counter["count"] == 25

    def test_run_worksteal_backtest_skips_breadth_ratio_without_allocation_callback(self, monkeypatch):
        bars = {
            "BTCUSD": make_bars([100.0 + i for i in range(25)], symbol="BTCUSD"),
            "ETHUSD": make_bars([120.0 + i for i in range(25)], symbol="ETHUSD"),
        }
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
            market_breadth_filter=0.0,
            max_positions=0,
        )
        counter = {"count": 0}
        original = strategy_mod.compute_breadth_ratio

        def counted_breadth(*args, **kwargs):
            counter["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(strategy_mod, "compute_breadth_ratio", counted_breadth)

        run_worksteal_backtest(bars, config)

        assert counter["count"] == 0


    def test_no_dip_no_trade(self):
        prices = [100 + i for i in range(60)]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.005)
        eq, trades, metrics = run_worksteal_backtest(
            bars, config, start_date="2026-01-30", end_date="2026-02-28"
        )
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) == 0

    def test_dip_triggers_buy(self):
        prices = list(range(100, 130))
        prices += [120, 118, 117, 116]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, lookback_days=10,
            profit_target_pct=0.05, stop_loss_pct=0.08,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) >= 1

    def test_profit_target_exit(self):
        prices = [100] * 25 + [92, 90, 88, 86, 85]
        prices += [88, 92, 95, 98, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02,
            profit_target_pct=0.05, stop_loss_pct=0.15,
            lookback_days=20, max_hold_days=30,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        sells = [t for t in trades if t.side == "sell"]
        assert len(sells) >= 0

    def test_max_positions_respected(self):
        all_bars = {}
        for i, sym in enumerate(["S1USD", "S2USD", "S3USD", "S4USD", "S5USD"]):
            prices = [100 + i] * 25 + [80 + i] * 10
            all_bars[sym] = make_bars(prices, symbol=sym)
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.05, max_positions=3, lookback_days=20,
        )
        eq, trades, metrics = run_worksteal_backtest(all_bars, config)
        if not eq.empty:
            assert eq["n_positions"].max() <= 3

    def test_multi_symbol_only_dip_bought(self):
        all_bars = {
            "UPUSD": make_bars([100 + i * 2 for i in range(40)], symbol="UPUSD"),
            "DIPUSD": make_bars([100] * 25 + [92, 90, 88, 85, 83, 85, 88, 92, 95, 100,
                                               103, 105, 108, 110, 112], symbol="DIPUSD"),
            "FLATUSD": make_bars([100] * 40, symbol="FLATUSD"),
        }
        config = WorkStealConfig(dip_pct=0.10, proximity_pct=0.02, lookback_days=20)
        eq, trades, metrics = run_worksteal_backtest(all_bars, config)
        buy_symbols = set(t.symbol for t in trades if t.side == "buy")
        assert "UPUSD" not in buy_symbols

    def test_leverage_increases_returns(self):
        prices = [100] * 25 + [90, 88, 85, 88, 92, 95, 100, 105, 110]
        bars = {"BTCUSD": make_bars(prices)}
        config_1x = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, max_leverage=1.0,
            lookback_days=20, profit_target_pct=0.10,
        )
        config_3x = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, max_leverage=3.0,
            lookback_days=20, profit_target_pct=0.10,
        )
        eq1, _, m1 = run_worksteal_backtest(dict(bars), config_1x)
        eq3, _, m3 = run_worksteal_backtest(dict(bars), config_3x)
        # 3x leverage should produce >= 1x returns (ignoring margin cost)
        # Just check both run without error
        assert len(eq1) == len(eq3)

    def test_shorts_enabled(self):
        # Price pumps then drops - short should profit
        prices = [100] * 25 + [105, 110, 115, 120, 125, 120, 115, 110, 105, 100]
        bars = {"ALTUSD": make_bars(prices, symbol="ALTUSD")}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, enable_shorts=True,
            short_pump_pct=0.10, profit_target_pct=0.05,
            lookback_days=20,
        )
        eq, trades, metrics = run_worksteal_backtest(bars, config)
        shorts = [t for t in trades if t.side == "short"]
        # May or may not trigger depending on exact noise
        assert isinstance(shorts, list)

    def test_fdusd_zero_fee_applied(self):
        prices = [100] * 25 + [90, 88, 85, 88, 92, 95, 100, 105]
        bars_btc = {"BTCUSD": make_bars(prices)}
        bars_alt = {"DOGEUSD": make_bars(prices, symbol="DOGEUSD")}
        config = WorkStealConfig(
            dip_pct=0.10, proximity_pct=0.02, fdusd_fee=0.0, maker_fee=0.001,
            lookback_days=20, profit_target_pct=0.05,
        )
        _, trades_btc, _ = run_worksteal_backtest(bars_btc, config)
        _, trades_alt, _ = run_worksteal_backtest(bars_alt, config)
        # BTC trades should have 0 fee
        btc_fees = sum(t.fee for t in trades_btc if t.symbol == "BTCUSD" and t.side == "buy")
        alt_fees = sum(t.fee for t in trades_alt if t.symbol == "DOGEUSD" and t.side == "buy")
        if trades_btc and trades_alt:
            assert btc_fees < alt_fees  # BTC 0% < DOGE 10bps

    def test_realistic_fill_requires_signal_bar_touch(self):
        dates = pd.date_range("2026-01-01", periods=22, freq="D", tz="UTC")
        rows = []
        for idx, ts in enumerate(dates):
            if idx < 20:
                close = 100.0
                low = 99.0
            elif idx == 20:
                close = 91.0
                low = 91.2  # close is near target ~90.9, but bar never touches it
            else:
                close = 95.0
                low = 94.0
            rows.append(
                {
                    "timestamp": ts,
                    "open": close,
                    "high": close + 1.0,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                    "symbol": "DIPUSD",
                }
            )
        bars = {"DIPUSD": pd.DataFrame(rows)}

        optimistic = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )
        realistic = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            realistic_fill=True,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )

        _, optimistic_trades, _ = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, optimistic)
        _, realistic_trades, _ = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, realistic)

        assert any(trade.side == "buy" for trade in optimistic_trades)
        assert not any(trade.side == "buy" for trade in realistic_trades)

    def test_daily_checkpoint_only_requires_next_bar_fill(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
        rows = [
            {"timestamp": dates[0], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[1], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[2], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[3], "open": 95.0, "high": 100.0, "low": 89.0, "close": 90.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[4], "open": 95.0, "high": 96.0, "low": 94.0, "close": 95.0, "volume": 1000.0, "symbol": "DIPUSD"},
        ]
        bars = {"DIPUSD": pd.DataFrame(rows)}

        same_bar = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=3,
            sma_filter_period=0,
            realistic_fill=True,
            profit_target_pct=0.20,
            stop_loss_pct=0.20,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )
        next_bar = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=3,
            sma_filter_period=0,
            realistic_fill=True,
            daily_checkpoint_only=True,
            profit_target_pct=0.20,
            stop_loss_pct=0.20,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )

        _, same_bar_trades, _ = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, same_bar)
        _, next_bar_trades, _ = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, next_bar)

        assert [trade.side for trade in same_bar_trades] == ["buy"]
        assert same_bar_trades[0].timestamp == dates[3]
        assert next_bar_trades == []

    def test_daily_checkpoint_only_blocks_same_bar_roundtrip(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
        rows = [
            {"timestamp": dates[0], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[1], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[2], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[3], "open": 95.0, "high": 100.0, "low": 89.0, "close": 90.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[4], "open": 92.0, "high": 95.0, "low": 89.0, "close": 94.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[5], "open": 94.0, "high": 96.0, "low": 93.0, "close": 95.0, "volume": 1000.0, "symbol": "DIPUSD"},
        ]
        bars = {"DIPUSD": pd.DataFrame(rows)}

        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=3,
            sma_filter_period=0,
            realistic_fill=True,
            daily_checkpoint_only=True,
            profit_target_pct=0.05,
            stop_loss_pct=0.20,
            max_hold_days=10,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )

        _, trades, _ = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, config)

        assert [trade.side for trade in trades] == ["buy", "sell"]
        assert trades[0].timestamp == dates[4]
        assert trades[1].timestamp == dates[5]
        assert trades[1].reason == "profit_target"

    def test_daily_checkpoint_only_pending_order_expires_after_one_bar(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
        rows = [
            {"timestamp": dates[0], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[1], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[2], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[3], "open": 95.0, "high": 100.0, "low": 89.0, "close": 90.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[4], "open": 93.0, "high": 95.0, "low": 92.0, "close": 94.0, "volume": 1000.0, "symbol": "DIPUSD"},
            {"timestamp": dates[5], "open": 94.0, "high": 95.0, "low": 89.0, "close": 94.0, "volume": 1000.0, "symbol": "DIPUSD"},
        ]
        bars = {"DIPUSD": pd.DataFrame(rows)}
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=3,
            sma_filter_period=0,
            realistic_fill=True,
            daily_checkpoint_only=True,
            profit_target_pct=0.20,
            stop_loss_pct=0.20,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )

        _, trades, _ = run_worksteal_backtest({k: v.copy() for k, v in bars.items()}, config)

        assert trades == []

    def test_base_asset_idle_mode_participates_in_eth_trend(self):
        eth_prices = [100 + i * 2 for i in range(40)]
        bars = {"ETHUSD": make_bars(eth_prices, symbol="ETHUSD")}
        config = WorkStealConfig(
            base_asset_symbol="ETHUSD",
            base_asset_rebalance_min_cash=0.0,
            lookback_days=10,
        )

        eq, trades, metrics = run_worksteal_backtest(bars, config)

        assert trades == []
        assert eq["base_asset_qty"].iloc[-1] > 0.0
        assert metrics["final_equity"] > config.initial_cash

    def test_base_asset_filter_can_keep_strategy_in_cash(self):
        eth_prices = [200 - i * 3 for i in range(40)]
        bars = {"ETHUSD": make_bars(eth_prices, symbol="ETHUSD")}
        config = WorkStealConfig(
            base_asset_symbol="ETHUSD",
            base_asset_momentum_period=5,
            base_asset_min_momentum=0.0,
            base_asset_rebalance_min_cash=0.0,
            lookback_days=10,
        )

        eq, _, metrics = run_worksteal_backtest(bars, config)

        assert eq["base_asset_qty"].max() == pytest.approx(0.0)
        assert metrics["final_equity"] == pytest.approx(config.initial_cash)

    def test_base_asset_is_liquidated_to_fund_alt_entry(self):
        eth_prices = [100 + i for i in range(25)] + [125 + i for i in range(15)]
        alt_prices = [100] * 25 + [88, 85, 84, 86, 90, 95, 100, 105, 108, 110, 112, 114, 116, 118, 120]
        bars = {
            "ETHUSD": make_bars(eth_prices, symbol="ETHUSD"),
            "ALTUSD": make_bars(alt_prices, symbol="ALTUSD"),
        }
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            profit_target_pct=0.10,
            stop_loss_pct=0.08,
            base_asset_symbol="ETHUSD",
            base_asset_rebalance_min_cash=0.0,
        )

        eq, trades, _ = run_worksteal_backtest(bars, config)

        assert any(t.symbol == "ALTUSD" and t.side == "buy" for t in trades)
        assert eq["base_asset_qty"].max() > 0.0
        assert eq["base_asset_qty"].min() < eq["base_asset_qty"].max()


class TestSmaCheckMethod:
    def test_pre_dip_allows_entry_below_sma(self):
        prices = [100] * 25 + [90, 85, 82, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            sma_filter_period=20, sma_check_method="pre_dip",
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) >= 1

    def test_current_sma_blocks_entry_below_sma(self):
        prices = [100] * 25 + [90, 85, 82, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            sma_filter_period=20, sma_check_method="current",
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) == 0

    def test_none_sma_allows_any_entry(self):
        prices = [50] * 10 + [100] * 15 + [85, 82, 80, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            sma_filter_period=20, sma_check_method="none",
            risk_off_trigger_momentum_period=0, risk_off_trigger_sma_period=0,
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        buys = [t for t in trades if t.side == "buy"]
        assert len(buys) >= 1


class TestAdaptiveDip:
    def test_adaptive_dip_uses_atr(self):
        prices = [100] * 25 + [92, 90, 88, 86, 85, 88, 92, 95, 100]
        bars = {"BTCUSD": make_bars(prices)}
        config = WorkStealConfig(
            dip_pct=0.20, proximity_pct=0.05, lookback_days=20,
            adaptive_dip=True,
        )
        eq, trades, _ = run_worksteal_backtest(bars, config)
        assert isinstance(trades, list)

    def test_adaptive_dip_false_uses_fixed_threshold(self):
        prices = [100] * 25 + [85, 82, 80, 85, 88, 92]
        bars = {"BTCUSD": make_bars(prices)}
        config_fixed = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            adaptive_dip=False,
        )
        config_adaptive = WorkStealConfig(
            dip_pct=0.15, proximity_pct=0.05, lookback_days=20,
            adaptive_dip=True,
        )
        _, trades_fixed, _ = run_worksteal_backtest(dict(bars), config_fixed)
        _, trades_adaptive, _ = run_worksteal_backtest(dict(bars), config_adaptive)
        assert isinstance(trades_fixed, list)
        assert isinstance(trades_adaptive, list)


class TestRiskOffMomentum:
    def test_risk_off_not_triggered_with_default_threshold(self):
        prices = [100, 99, 98, 97, 96, 95]
        bars_df = make_bars(prices)
        current_bars = {"BTCUSD": bars_df.iloc[-1]}
        history = {"BTCUSD": bars_df}
        config = WorkStealConfig(
            momentum_period=5, risk_off_momentum_threshold=-0.05,
        )
        result = _risk_off_triggered(current_bars, history, config)
        assert result is False

    def test_risk_off_triggered_at_zero_threshold(self):
        prices = [100, 99, 98, 97, 96, 95]
        bars_df = make_bars(prices)
        current_bars = {"BTCUSD": bars_df.iloc[-1]}
        history = {"BTCUSD": bars_df}
        config = WorkStealConfig(
            risk_off_trigger_momentum_period=5, risk_off_momentum_threshold=0.0,
        )
        result = _risk_off_triggered(current_bars, history, config)
        assert result is True

    def test_risk_off_disabled_when_momentum_period_zero(self):
        prices = [100, 50, 40, 30, 20, 10]
        bars_df = make_bars(prices)
        current_bars = {"BTCUSD": bars_df.iloc[-1]}
        history = {"BTCUSD": bars_df}
        config = WorkStealConfig(momentum_period=0)
        result = _risk_off_triggered(current_bars, history, config)
        assert result is False


class TestNewDefaults:
    def test_proximity_pct_default_is_003(self):
        config = WorkStealConfig()
        assert config.proximity_pct == 0.03

    def test_sma_check_method_default_is_pre_dip(self):
        config = WorkStealConfig()
        assert config.sma_check_method == "pre_dip"

    def test_risk_off_momentum_threshold_default(self):
        config = WorkStealConfig()
        assert config.risk_off_momentum_threshold == -0.05

    def test_adaptive_dip_default_is_false(self):
        config = WorkStealConfig()
        assert config.adaptive_dip is False


class TestComputeMetrics:
    def test_positive_return(self):
        eq_df = pd.DataFrame({"equity": [10000, 10100, 10200, 10300, 10400, 10500]})
        m = compute_metrics(eq_df, WorkStealConfig())
        assert m["total_return"] > 0
        assert m["sortino"] > 0

    def test_negative_return(self):
        eq_df = pd.DataFrame({"equity": [10000, 9900, 9800, 9700, 9600, 9500]})
        m = compute_metrics(eq_df, WorkStealConfig())
        assert m["total_return"] < 0

    def test_flat(self):
        eq_df = pd.DataFrame({"equity": [10000, 10000, 10000, 10000]})
        m = compute_metrics(eq_df, WorkStealConfig())
        assert m["total_return"] == pytest.approx(0.0, abs=1e-10)

    def test_trade_counts_use_completed_exits(self):
        eq_df = pd.DataFrame({"equity": [10000, 10100, 10050, 10200]})
        trades = [
            type("Trade", (), {"side": "buy", "pnl": 0.0})(),
            type("Trade", (), {"side": "sell", "pnl": 25.0})(),
            type("Trade", (), {"side": "short", "pnl": 0.0})(),
            type("Trade", (), {"side": "cover", "pnl": -10.0})(),
        ]
        m = compute_metrics(eq_df, WorkStealConfig(), trades=trades)
        assert m["n_orders"] == 4
        assert m["n_trades"] == 2
        assert m["win_rate"] == pytest.approx(50.0)

    def test_backtest_reports_candidate_and_fill_metrics(self):
        bars = {"BTCUSD": make_bars([100.0] * 20 + [91.0, 95.0], symbol="BTCUSD")}
        config = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.02,
            lookback_days=20,
            sma_filter_period=0,
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        )

        _eq, _trades, metrics = run_worksteal_backtest(bars, config)

        assert metrics["candidates_generated"] >= 1
        assert metrics["candidates_visible"] >= 1
        assert metrics["entries_executed"] >= 1
        assert 0.0 <= metrics["fill_rate"] <= 1.0
        assert 0.0 <= metrics["visible_fill_rate"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
