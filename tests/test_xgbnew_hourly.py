"""Tests for the hourly XGB pipeline (loader, simulator, windows).

Scope:
* ``load_hourly_symbol_csv`` + ``build_hourly_dataset`` round-trip from real CSVs
  in ``trainingdatahourly/`` when available, with a synthetic fallback so the
  suite works in stripped checkouts.
* ``simulate_hourly`` annualises bar-granular PnL correctly for both stock
  (6.5h/day) and crypto (24/7) hours-per-year budgets.
* No-lookahead sanity on ``build_features_for_symbol_hourly`` (feature row H
  must not depend on bar H's close).
* The multi-window driver's bar-window builder produces the expected ranges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.backtest import (
    BacktestConfig,
    CRYPTO_HOURS_PER_MONTH,
    CRYPTO_HOURS_PER_YEAR,
    STOCK_HOURS_PER_MONTH,
    STOCK_HOURS_PER_YEAR,
    simulate_hourly,
)
from xgbnew.dataset import (
    build_hourly_dataset,
    list_hourly_symbols,
    load_hourly_symbol_csv,
)
from xgbnew.eval_hourly_multiwindow import _build_hourly_windows, _bars_per_year
from xgbnew.features import HOURLY_FEATURE_COLS, build_features_for_symbol_hourly


HOURLY_ROOT = REPO / "trainingdatahourly"


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_hourly_ohlcv(
    n_bars: int = 400,
    start: str = "2025-01-06 14:30:00",
    tz: str = "UTC",
    *,
    seed: int = 11,
    drift_bps: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Generate a log-random-walk at roughly 40bps/hr vol (reasonable for stocks).
    ret = rng.normal(drift_bps / 1e4, 40.0 / 1e4, size=n_bars)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0, 5 / 1e4, size=n_bars))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 10 / 1e4, n_bars)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 10 / 1e4, n_bars)))
    vol = rng.uniform(2e5, 5e6, n_bars)
    ts = pd.date_range(start, periods=n_bars, freq="1h", tz=tz)
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low,
         "close": close, "volume": vol}
    )


class _ConstantScoreModel:
    """Stub model whose ``predict_scores`` returns a constant probability."""

    def __init__(self, prob: float = 0.6) -> None:
        self.prob = float(prob)
        self.feature_cols: list[str] = list(HOURLY_FEATURE_COLS)
        self._fitted = True

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover - trivial
        return pd.Series(np.full(len(df), self.prob), index=df.index, name="xgb_score")


# ── features: no-lookahead sanity ───────────────────────────────────────────


class TestHourlyFeaturesNoLookahead:
    def test_feature_row_h_is_independent_of_bar_h_close(self):
        base = _make_hourly_ohlcv(n_bars=200)
        feat_a = build_features_for_symbol_hourly(base, symbol="X")
        # Perturb bar 100's close strongly — a leak-free feature for row 100
        # would not move because features at row H use only bars < H.
        pert = base.copy()
        pert.loc[100, "close"] = pert.loc[100, "close"] * 1.5
        pert.loc[100, "high"] = max(pert.loc[100, "high"], pert.loc[100, "close"])
        feat_b = build_features_for_symbol_hourly(pert, symbol="X")
        row_a = feat_a.loc[100, HOURLY_FEATURE_COLS[:5]].to_numpy(dtype=float)
        row_b = feat_b.loc[100, HOURLY_FEATURE_COLS[:5]].to_numpy(dtype=float)
        # Drop NaNs (first rows have them). Any non-NaN pair must match.
        mask = np.isfinite(row_a) & np.isfinite(row_b)
        if mask.any():
            assert np.allclose(row_a[mask], row_b[mask])


# ── dataset loader ──────────────────────────────────────────────────────────


class TestHourlyLoader:
    def test_load_symbol_csv_synthetic_stocks(self, tmp_path):
        # Minimal fake layout: <tmp>/stocks/FAKE.csv
        stocks = tmp_path / "stocks"
        stocks.mkdir()
        df = _make_hourly_ohlcv(n_bars=300)
        df["symbol"] = "FAKE"
        df.to_csv(stocks / "FAKE.csv", index=False)
        got = load_hourly_symbol_csv("FAKE", tmp_path)
        assert got is not None
        assert set(got.columns) >= {"timestamp", "open", "high", "low",
                                    "close", "volume", "symbol", "_kind"}
        assert got["_kind"].iloc[0] == "stocks"
        assert len(got) == 300

    def test_load_symbol_csv_missing_returns_none(self, tmp_path):
        (tmp_path / "stocks").mkdir()
        (tmp_path / "crypto").mkdir()
        assert load_hourly_symbol_csv("DOES_NOT_EXIST", tmp_path) is None

    def test_list_symbols_segmented(self, tmp_path):
        # Universe split is by symbol suffix (USD-quote → crypto), not by subdir,
        # because the legacy 'crypto/' folder contains a few stock CSVs (DBX,
        # AAPL, AMZN). Use realistic suffixes here.
        for kind, name in (("stocks", "AAA"), ("stocks", "BBB"), ("crypto", "BTCUSD")):
            d = tmp_path / kind
            d.mkdir(exist_ok=True)
            _make_hourly_ohlcv(250).to_csv(d / f"{name}.csv", index=False)
        assert list_hourly_symbols(tmp_path, universe="stocks") == ["AAA", "BBB"]
        assert list_hourly_symbols(tmp_path, universe="crypto") == ["BTCUSD"]
        assert set(list_hourly_symbols(tmp_path, universe="both")) == {"AAA", "BBB", "BTCUSD"}

    def test_kind_classifier_recognises_usd_suffixes(self, tmp_path):
        # AAPL in the crypto/ subdir must still be classified as 'stocks'.
        (tmp_path / "crypto").mkdir()
        _make_hourly_ohlcv(250).to_csv(tmp_path / "crypto" / "AAPL.csv", index=False)
        got = load_hourly_symbol_csv("AAPL", tmp_path)
        assert got is not None
        assert got["_kind"].iloc[0] == "stocks"
        # And BTCUSD in stocks/ should still be 'crypto'.
        (tmp_path / "stocks").mkdir()
        _make_hourly_ohlcv(250).to_csv(tmp_path / "stocks" / "BTCUSD.csv", index=False)
        got2 = load_hourly_symbol_csv("BTCUSD", tmp_path)
        assert got2 is not None
        assert got2["_kind"].iloc[0] == "crypto"

    def test_build_hourly_dataset_splits_by_timestamp(self, tmp_path):
        (tmp_path / "stocks").mkdir()
        df = _make_hourly_ohlcv(
            n_bars=800, start="2025-01-06 14:30:00", drift_bps=1.0
        )
        df.to_csv(tmp_path / "stocks" / "AAA.csv", index=False)
        train_df, val_df, test_df, kind_map = build_hourly_dataset(
            tmp_path, ["AAA"],
            train_start=None,
            train_end=pd.Timestamp("2025-01-16", tz="UTC"),
            val_end=pd.Timestamp("2025-01-20", tz="UTC"),
            test_end=pd.Timestamp("2025-02-15", tz="UTC"),
            universe="stocks",
            min_bars=50,
            min_dollar_vol=0.0,
        )
        # The three splits must be non-empty and time-ordered.
        assert not train_df.empty and not val_df.empty and not test_df.empty
        assert train_df["timestamp"].max() <= pd.Timestamp("2025-01-16", tz="UTC")
        assert val_df["timestamp"].min() > pd.Timestamp("2025-01-16", tz="UTC")
        assert val_df["timestamp"].max() <= pd.Timestamp("2025-01-20", tz="UTC")
        assert test_df["timestamp"].min() > pd.Timestamp("2025-01-20", tz="UTC")
        assert kind_map == {"AAA": "stocks"}
        for col in HOURLY_FEATURE_COLS[:4]:
            assert col in train_df.columns


# ── simulator annualisation ─────────────────────────────────────────────────


class TestSimulateHourly:
    def _synthetic_scored_frame(self, n: int = 200, up_bps: float = 20.0) -> pd.DataFrame:
        """Two symbols, both known-profitable at a +``up_bps`` bar return."""
        ts = pd.date_range("2025-03-01 14:30:00", periods=n, freq="1h", tz="UTC")
        rows = []
        for sym in ("AAA", "BBB"):
            for i, t in enumerate(ts):
                open_p = 100.0
                close_p = open_p * (1.0 + up_bps / 1e4)
                rows.append({
                    "timestamp": t,
                    "symbol": sym,
                    "actual_open": open_p,
                    "actual_close": close_p,
                    "spread_bps": 5.0,
                    "target_oc": up_bps / 1e4,
                    "target_oc_up": 1,
                    "chronos_oc_return": 0.0,
                    **{c: 0.0 for c in HOURLY_FEATURE_COLS},
                })
        df = pd.DataFrame(rows)
        return df

    def test_positive_pnl_with_profitable_bars(self):
        df = self._synthetic_scored_frame(n=100, up_bps=25.0)
        cfg = BacktestConfig(top_n=1, leverage=1.0, xgb_weight=1.0,
                             commission_bps=0.0, fill_buffer_bps=0.0,
                             fee_rate=0.0, min_dollar_vol=0.0, max_spread_bps=1000.0)
        model = _ConstantScoreModel(0.9)
        res = simulate_hourly(df, model, cfg, bars_per_year=STOCK_HOURS_PER_YEAR)
        assert res.total_return_pct > 0
        # Sanity: per-bar +25bps - 0 fee → compounded > 0 over 100 bars
        assert res.total_trades == 100

    def test_fee_eats_small_moves(self):
        df = self._synthetic_scored_frame(n=50, up_bps=2.0)
        cfg = BacktestConfig(top_n=1, leverage=1.0, xgb_weight=1.0,
                             commission_bps=0.0, fill_buffer_bps=10.0,
                             fee_rate=0.0010, min_dollar_vol=0.0, max_spread_bps=1000.0)
        res = simulate_hourly(df, _ConstantScoreModel(0.9), cfg,
                              bars_per_year=STOCK_HOURS_PER_YEAR)
        # With 10bps fill buffer + 10bps fee per side on a +2bps move, net < 0.
        assert res.total_return_pct < 0

    def test_min_score_gate_blocks_trades(self):
        df = self._synthetic_scored_frame(n=20, up_bps=10.0)
        cfg = BacktestConfig(top_n=1, xgb_weight=1.0, min_score=0.99,
                             fee_rate=0.0, fill_buffer_bps=0.0,
                             min_dollar_vol=0.0, max_spread_bps=1000.0)
        res = simulate_hourly(df, _ConstantScoreModel(0.5), cfg,
                              bars_per_year=STOCK_HOURS_PER_YEAR)
        assert res.total_trades == 0

    def test_crypto_annualisation_higher_than_stocks(self):
        """Crypto has 5.35× more bars per year — same per-bar return → higher annual."""
        df = self._synthetic_scored_frame(n=200, up_bps=3.0)
        cfg = BacktestConfig(top_n=1, xgb_weight=1.0, fee_rate=0.0,
                             fill_buffer_bps=0.0, min_dollar_vol=0.0,
                             max_spread_bps=1000.0)
        stock_res = simulate_hourly(
            df, _ConstantScoreModel(0.9), cfg, bars_per_year=STOCK_HOURS_PER_YEAR
        )
        crypto_res = simulate_hourly(
            df, _ConstantScoreModel(0.9), cfg, bars_per_year=CRYPTO_HOURS_PER_YEAR
        )
        # Total return is identical because total compounding is the same — only
        # the annualisation horizon scales with bars_per_year.
        assert abs(stock_res.total_return_pct - crypto_res.total_return_pct) < 1e-6
        assert crypto_res.annualized_return_pct > stock_res.annualized_return_pct

    def test_monthly_pct_matches_bars_per_month_exponent(self):
        """``monthly_return_pct`` is total compounded to bars_per_month horizon."""
        df = self._synthetic_scored_frame(n=200, up_bps=5.0)
        cfg = BacktestConfig(top_n=1, xgb_weight=1.0, fee_rate=0.0,
                             fill_buffer_bps=0.0, min_dollar_vol=0.0,
                             max_spread_bps=1000.0)
        res = simulate_hourly(
            df, _ConstantScoreModel(0.9), cfg,
            bars_per_year=STOCK_HOURS_PER_YEAR,
            bars_per_month=STOCK_HOURS_PER_MONTH,
        )
        # Reproduce the expected monthly%: (1 + total)^(bars_per_month / n_bars) - 1
        n = len(res.day_results)
        expected = ((1.0 + res.total_return_pct / 100.0) ** (STOCK_HOURS_PER_MONTH / n) - 1.0) * 100.0
        assert abs(res.monthly_return_pct - expected) < 1e-6

    def test_leverage_scales_return(self):
        df = self._synthetic_scored_frame(n=50, up_bps=10.0)
        base = simulate_hourly(
            df, _ConstantScoreModel(0.9),
            BacktestConfig(top_n=1, leverage=1.0, xgb_weight=1.0, fee_rate=0.0,
                           fill_buffer_bps=0.0, min_dollar_vol=0.0,
                           max_spread_bps=1000.0),
            bars_per_year=STOCK_HOURS_PER_YEAR,
        )
        lev = simulate_hourly(
            df, _ConstantScoreModel(0.9),
            BacktestConfig(top_n=1, leverage=2.0, xgb_weight=1.0, fee_rate=0.0,
                           fill_buffer_bps=0.0, min_dollar_vol=0.0,
                           max_spread_bps=1000.0),
            bars_per_year=STOCK_HOURS_PER_YEAR,
        )
        # 2× leverage should roughly double per-bar return before margin cost.
        # (Margin prorated per bar is tiny for lev=2: 6.25% / 1638 ≈ 0.0004%.)
        assert lev.total_return_pct > base.total_return_pct * 1.9


# ── window builder ──────────────────────────────────────────────────────────


class TestBuildHourlyWindows:
    def test_basic_stride(self):
        ts = pd.date_range("2025-01-01", periods=500, freq="1h", tz="UTC").tolist()
        w = _build_hourly_windows(ts, window_bars=100, stride_bars=50)
        assert len(w) == 9  # floor((500 - 100) / 50) + 1
        assert (w[1][0] - w[0][0]) == pd.Timedelta(hours=50)

    def test_insufficient_bars(self):
        ts = pd.date_range("2025-01-01", periods=50, freq="1h", tz="UTC").tolist()
        assert _build_hourly_windows(ts, window_bars=100, stride_bars=10) == []

    def test_bars_per_year_helper(self):
        assert _bars_per_year("stocks") == STOCK_HOURS_PER_YEAR
        assert _bars_per_year("crypto") == CRYPTO_HOURS_PER_YEAR
        assert _bars_per_year("both") == STOCK_HOURS_PER_YEAR


# ── integration: build + simulate on real CSVs when available ───────────────


@pytest.mark.skipif(
    not (HOURLY_ROOT / "stocks").exists(),
    reason="trainingdatahourly/stocks not present",
)
def test_real_data_smoke():
    """End-to-end smoke on 3 real symbols (skipped outside the monorepo)."""
    syms = list_hourly_symbols(HOURLY_ROOT, universe="stocks")[:3]
    if not syms:
        pytest.skip("no hourly stocks CSVs")
    train_df, val_df, test_df, kind_map = build_hourly_dataset(
        HOURLY_ROOT, syms,
        train_start=None,
        train_end=pd.Timestamp("2025-09-30", tz="UTC"),
        val_end=pd.Timestamp("2025-12-31", tz="UTC"),
        test_end=pd.Timestamp("2026-04-10", tz="UTC"),
        universe="stocks", min_bars=200, min_dollar_vol=0.0,
    )
    # At least one split must come back non-empty on the real data.
    assert sum(len(d) for d in (train_df, val_df, test_df)) > 0
