"""Tests for xgbnew eval_multiwindow and live_trader modules."""
from __future__ import annotations

from datetime import datetime, timezone
import types
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.eval_multiwindow import _monthly_return, main as eval_main
import xgbnew.live_trader as live_trader
from xgbnew.live_trader import score_all_symbols


# ─── _monthly_return ──────────────────────────────────────────────────────────

class TestMonthlyReturn:
    def test_zero_return(self):
        assert _monthly_return(0.0, 50) == pytest.approx(0.0, abs=1e-6)

    def test_100pct_over_50d(self):
        # (1+1.0)^(21/50) - 1 ≈ 34%
        result = _monthly_return(100.0, 50)
        assert 0.30 < result < 0.40

    def test_negative(self):
        result = _monthly_return(-10.0, 50)
        assert result < 0

    def test_short_window(self):
        # should not crash
        result = _monthly_return(5.0, 1)
        assert np.isfinite(result)


# ─── eval_multiwindow CLI ─────────────────────────────────────────────────────

def _make_fake_csv_dir(tmp_path: Path, n_symbols: int = 10, n_days: int = 400):
    """Create a minimal temp CSV dir for eval_multiwindow tests.

    Returns (data_dir, sym_file) paths.
    """
    import numpy as np
    import pandas as pd

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    syms = [f"SYM{i}" for i in range(n_symbols)]
    sym_file = tmp_path / "symbols.txt"
    sym_file.write_text("\n".join(syms))
    np.random.seed(7)
    ts = pd.date_range("2021-01-04", periods=n_days, freq="B", tz="UTC")
    for i_sym, sym in enumerate(syms):
        close = 100.0 + np.cumsum(np.random.randn(n_days) * 0.5)
        close = np.clip(close, 1.0, None)
        # Random intraday swing so ~50% up, ~50% down
        oc_return = np.random.randn(n_days) * 0.01
        open_ = close / (1.0 + oc_return)
        df = pd.DataFrame({
            "timestamp": ts,
            "open":   open_,
            "high":   np.maximum(open_, close) * 1.005,
            "low":    np.minimum(open_, close) * 0.995,
            "close":  close,
            "volume": np.random.uniform(1e7, 1e8, n_days),
        })
        df.to_csv(data_dir / f"{sym}.csv", index=False)
    return data_dir, sym_file


class _FakeCSVHelper:
    """Thin wrapper for backward-compat use in TestEvalMultiwindow."""
    def __init__(self, tmp_path, n_symbols=10, n_days=400):
        self.data_dir, self.sym_file = _make_fake_csv_dir(
            tmp_path, n_symbols=n_symbols, n_days=n_days
        )


class TestEvalMultiwindow:
    def test_runs_and_produces_output(self, tmp_path):
        fake = _FakeCSVHelper(tmp_path)
        out_dir = tmp_path / "out"
        # 400 bdays from 2021-01-04 ≈ mid-2022; train on first 250, OOS on rest
        ret = eval_main([
            "--symbols-file", str(fake.sym_file),
            "--data-root",    str(fake.data_dir),
            "--train-start",  "2021-01-01",
            "--train-end",    "2021-12-31",
            "--oos-start",    "2022-01-01",
            "--oos-end",      "2022-12-31",
            "--window-days",  "30",
            "--stride-days",  "15",
            "--top-n",        "2",
            "--output-dir",   str(out_dir),
            "--n-estimators", "30",   # fast
        ])
        assert ret == 0
        jsons = list(out_dir.glob("multiwindow_*.json"))
        assert len(jsons) == 1
        import json
        data = json.loads(jsons[0].read_text())
        assert "median_monthly_pct" in data
        assert "windows" in data
        assert "best" in data
        assert "sweep_results" in data
        assert "coverage" in data
        assert len(data["windows"]) > 0

    def test_model_save_path(self, tmp_path):
        fake = _FakeCSVHelper(tmp_path)
        model_path = tmp_path / "model.pkl"
        ret = eval_main([
            "--symbols-file",    str(fake.sym_file),
            "--data-root",       str(fake.data_dir),
            "--train-start",     "2021-01-01",
            "--train-end",       "2021-10-31",
            "--oos-start",       "2021-11-01",
            "--oos-end",         "2022-12-31",
            "--window-days",     "30",
            "--stride-days",     "20",
            "--top-n",           "1",
            "--model-save-path", str(model_path),
            "--n-estimators",    "20",
            "--output-dir",      str(tmp_path / "out2"),
        ])
        assert ret == 0
        assert model_path.exists()


# ─── score_all_symbols ────────────────────────────────────────────────────────

class TestScoreAllSymbols:
    def _make_data_dir(self, tmp_path: Path, n_days: int = 200) -> tuple[Path, list[str]]:
        import numpy as np
        import pandas as pd
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        syms = ["AAPL", "MSFT", "NVDA"]
        np.random.seed(3)
        ts = pd.date_range("2023-01-03", periods=n_days, freq="B", tz="UTC")
        for sym in syms:
            close = 100.0 + np.cumsum(np.random.randn(n_days) * 0.5)
            close = np.clip(close, 1.0, None)
            oc_ret = np.random.randn(n_days) * 0.01
            open_ = close / (1.0 + oc_ret)
            df = pd.DataFrame({
                "timestamp": ts,
                "open":   open_,
                "high":   np.maximum(open_, close) * 1.005,
                "low":    np.minimum(open_, close) * 0.995,
                "close":  close,
                "volume": np.random.uniform(1e7, 1e8, n_days),
            })
            df.to_csv(data_dir / f"{sym}.csv", index=False)
        return data_dir, syms

    def test_returns_sorted_scores(self, tmp_path):
        data_dir, syms = self._make_data_dir(tmp_path)
        # Need a trained model
        from xgbnew.features import DAILY_FEATURE_COLS
        from xgbnew.dataset import build_daily_dataset
        from xgbnew.model import XGBStockModel
        from datetime import date
        train_df, _, _ = build_daily_dataset(
            data_root=data_dir, symbols=syms,
            train_start=date(2023, 1, 3), train_end=date(2023, 10, 31),
            val_start=date(2023, 11, 1), val_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1), test_end=date(2024, 6, 30),
        )
        if len(train_df) < 10:
            pytest.skip("insufficient synthetic data")
        model = XGBStockModel(n_estimators=20)
        model.fit(train_df, DAILY_FEATURE_COLS, verbose=False)

        df_scores = score_all_symbols(syms, data_dir, model, min_dollar_vol=1e3)
        assert isinstance(df_scores, pd.DataFrame)
        # If any scored, should be sorted desc
        if len(df_scores) > 1:
            scores = df_scores["score"].values
            assert (scores[:-1] >= scores[1:]).all(), "scores not sorted descending"

    def test_empty_returns_empty(self, tmp_path):
        from xgbnew.model import XGBStockModel
        # Minimal model stub
        model = XGBStockModel.__new__(XGBStockModel)
        model._fitted = False
        df = score_all_symbols([], tmp_path / "nodata", model)  # type: ignore
        assert len(df) == 0 or isinstance(df, pd.DataFrame)

    def test_skips_stale_local_daily_bars(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        stale_ts = pd.date_range("2025-10-01", periods=80, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": stale_ts,
                "open": np.linspace(90.0, 110.0, len(stale_ts)),
                "high": np.linspace(91.0, 111.0, len(stale_ts)),
                "low": np.linspace(89.0, 109.0, len(stale_ts)),
                "close": np.linspace(90.5, 110.5, len(stale_ts)),
                "volume": np.full(len(stale_ts), 2e7),
            }
        )
        df.to_csv(data_dir / "AAPL.csv", index=False)

        class _StubModel:
            def predict_scores(self, frame: pd.DataFrame) -> pd.Series:
                return pd.Series(0.8, index=frame.index, name="xgb_score")

        scores = score_all_symbols(
            ["AAPL"],
            data_dir,
            _StubModel(),  # type: ignore[arg-type]
            min_dollar_vol=1e3,
            now=datetime(2026, 4, 17, 13, 25, tzinfo=timezone.utc),
        )

        assert scores.empty

    def test_scores_symbols_in_one_batch_per_model(self, monkeypatch, tmp_path):
        syms = ["AAA", "BBB", "CCC"]
        ts = pd.date_range("2026-04-01", periods=80, freq="B", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": ts,
            "open": np.linspace(90.0, 100.0, len(ts)),
            "high": np.linspace(91.0, 101.0, len(ts)),
            "low": np.linspace(89.0, 99.0, len(ts)),
            "close": np.linspace(90.5, 100.5, len(ts)),
            "volume": np.full(len(ts), 2e7),
        })

        monkeypatch.setattr(live_trader, "_load_symbol_csv", lambda sym, root: raw.copy())

        def fake_features(_df, *, symbol):
            base = {col: 0.1 for col in live_trader.DAILY_FEATURE_COLS}
            base.update({
                "actual_close": {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}[symbol],
                "spread_bps": 5.0,
                "dolvol_20d_log": np.log1p(1e9),
                "ret_5d": {"AAA": 0.01, "BBB": 0.02, "CCC": 0.03}[symbol],
            })
            return pd.DataFrame([base])

        monkeypatch.setattr(live_trader, "build_features_for_symbol", fake_features)

        class _BatchModel:
            def __init__(self, offset):
                self.offset = offset
                self.call_lengths = []

            def predict_scores(self, frame):
                self.call_lengths.append(len(frame))
                return pd.Series(np.linspace(0.1, 0.3, len(frame)) + self.offset)

        m1 = _BatchModel(0.0)
        m2 = _BatchModel(0.2)

        scores = live_trader.score_all_symbols(
            syms,
            tmp_path,
            [m1, m2],  # type: ignore[arg-type]
            min_dollar_vol=1e3,
            now=datetime(2026, 4, 23, 13, 25, tzinfo=timezone.utc),
        )

        assert m1.call_lengths == [3]
        assert m2.call_lengths == [3]
        assert list(scores["symbol"]) == ["CCC", "BBB", "AAA"]
        assert scores.iloc[0]["per_seed_scores"] == pytest.approx([0.3, 0.5])

    def test_score_all_symbols_adds_model_required_panel_features(self, monkeypatch, tmp_path):
        syms = ["AAA", "BBB", "CCC"]
        ts = pd.date_range("2026-04-01", periods=80, freq="B", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": ts,
            "open": np.linspace(90.0, 100.0, len(ts)),
            "high": np.linspace(91.0, 101.0, len(ts)),
            "low": np.linspace(89.0, 99.0, len(ts)),
            "close": np.linspace(90.5, 100.5, len(ts)),
            "volume": np.full(len(ts), 2e7),
        })
        monkeypatch.setattr(live_trader, "_load_symbol_csv", lambda sym, root: raw.copy())

        def fake_features(_df, *, symbol):
            base = {col: 0.1 for col in live_trader.DAILY_FEATURE_COLS}
            base.update({
                "date": datetime(2026, 4, 23).date(),
                "actual_close": {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}[symbol],
                "spread_bps": 5.0,
                "dolvol_20d_log": np.log1p(1e9),
                "ret_1d": {"AAA": 0.01, "BBB": 0.02, "CCC": 0.03}[symbol],
                "ret_5d": {"AAA": 0.01, "BBB": 0.02, "CCC": 0.03}[symbol],
                "vol_20d": {"AAA": 0.20, "BBB": 0.40, "CCC": 0.60}[symbol],
                "rsi_14": {"AAA": 40.0, "BBB": 50.0, "CCC": 60.0}[symbol],
            })
            return pd.DataFrame([base])

        monkeypatch.setattr(live_trader, "build_features_for_symbol", fake_features)

        class _PanelFeatureModel:
            feature_cols = (
                list(live_trader.DAILY_FEATURE_COLS)
                + list(live_trader.DAILY_RANK_FEATURE_COLS)
                + list(live_trader.DAILY_DISPERSION_FEATURE_COLS)
            )

            def predict_scores(self, frame):
                missing = [col for col in self.feature_cols if col not in frame.columns]
                assert missing == []
                assert frame["rank_ret_5d"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
                assert "cs_iqr_ret5" in frame.columns
                assert "cs_skew_ret5" in frame.columns
                return pd.Series(frame["rank_ret_5d"].to_numpy(), index=frame.index)

        scores = live_trader.score_all_symbols(
            syms,
            tmp_path,
            _PanelFeatureModel(),  # type: ignore[arg-type]
            min_dollar_vol=1e3,
            now=datetime(2026, 4, 23, 13, 25, tzinfo=timezone.utc),
        )

        assert list(scores["symbol"]) == ["CCC", "BBB", "AAA"]
        assert scores.iloc[0]["score"] == pytest.approx(1.0)

    def test_score_all_symbols_panel_features_precede_inference_filters(
        self,
        monkeypatch,
        tmp_path,
    ):
        syms = ["AAA", "BBB", "CCC"]
        ts = pd.date_range("2026-04-01", periods=80, freq="B", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": ts,
            "open": np.linspace(90.0, 100.0, len(ts)),
            "high": np.linspace(91.0, 101.0, len(ts)),
            "low": np.linspace(89.0, 99.0, len(ts)),
            "close": np.linspace(90.5, 100.5, len(ts)),
            "volume": np.full(len(ts), 2e7),
        })
        monkeypatch.setattr(live_trader, "_load_symbol_csv", lambda sym, root: raw.copy())

        def fake_features(_df, *, symbol):
            base = {col: 0.1 for col in live_trader.DAILY_FEATURE_COLS}
            base.update({
                "date": datetime(2026, 4, 23).date(),
                "symbol": symbol,
                "actual_close": {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}[symbol],
                "spread_bps": 5.0,
                "dolvol_20d_log": np.log1p({"AAA": 1e6, "BBB": 1e9, "CCC": 1e9}[symbol]),
                "ret_1d": {"AAA": 0.01, "BBB": 0.02, "CCC": 0.03}[symbol],
                "ret_5d": {"AAA": 0.01, "BBB": 0.02, "CCC": 0.03}[symbol],
                "vol_20d": 0.30,
                "rsi_14": 50.0,
            })
            return pd.DataFrame([base])

        monkeypatch.setattr(live_trader, "build_features_for_symbol", fake_features)

        class _RankModel:
            feature_cols = list(live_trader.DAILY_FEATURE_COLS) + ["rank_ret_5d"]

            def predict_scores(self, frame):
                assert list(frame["symbol"]) == ["BBB", "CCC"]
                # AAA is filtered out by min_dollar_vol after feature assembly,
                # but it must still affect the rank feature to match sweeps.
                assert frame["rank_ret_5d"].tolist() == pytest.approx([2 / 3, 1.0])
                return pd.Series(frame["rank_ret_5d"].to_numpy(), index=frame.index)

        scores = live_trader.score_all_symbols(
            syms,
            tmp_path,
            _RankModel(),  # type: ignore[arg-type]
            min_dollar_vol=5e7,
            now=datetime(2026, 4, 23, 13, 25, tzinfo=timezone.utc),
        )

        assert list(scores["symbol"]) == ["CCC", "BBB"]
        assert scores.iloc[0]["score"] == pytest.approx(1.0)

    def test_score_all_symbols_applies_live_risk_rank_filters(self, monkeypatch, tmp_path):
        syms = ["AAA", "BBB", "CCC"]
        ts = pd.date_range("2026-04-01", periods=80, freq="B", tz="UTC")
        raw = pd.DataFrame({
            "timestamp": ts,
            "open": np.linspace(90.0, 100.0, len(ts)),
            "high": np.linspace(91.0, 101.0, len(ts)),
            "low": np.linspace(89.0, 99.0, len(ts)),
            "close": np.linspace(90.5, 100.5, len(ts)),
            "volume": np.full(len(ts), 2e7),
        })
        monkeypatch.setattr(live_trader, "_load_symbol_csv", lambda sym, root: raw.copy())

        def fake_features(_df, *, symbol):
            base = {col: 0.1 for col in live_trader.DAILY_FEATURE_COLS}
            base.update({
                "actual_close": {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}[symbol],
                "spread_bps": 5.0,
                "dolvol_20d_log": np.log1p(1e9),
                "ret_5d": {"AAA": 0.01, "BBB": 0.03, "CCC": 0.02}[symbol],
                "ret_20d": {"AAA": 0.30, "BBB": 0.10, "CCC": 0.20}[symbol],
                "vol_20d": {"AAA": 0.20, "BBB": 0.40, "CCC": 0.70}[symbol],
            })
            return pd.DataFrame([base])

        monkeypatch.setattr(live_trader, "build_features_for_symbol", fake_features)

        class _Model:
            def predict_scores(self, frame):
                assert len(frame) == 1
                return pd.Series([0.8], index=frame.index)

        scores = live_trader.score_all_symbols(
            syms,
            tmp_path,
            _Model(),  # type: ignore[arg-type]
            min_dollar_vol=1e3,
            min_vol_20d=0.10,
            max_vol_20d=0.55,
            max_ret_20d_rank_pct=0.80,
            min_ret_5d_rank_pct=0.50,
            now=datetime(2026, 4, 23, 13, 25, tzinfo=timezone.utc),
        )

        assert list(scores["symbol"]) == ["BBB"]
        assert scores.iloc[0]["ret_20d"] == pytest.approx(0.10)
        assert scores.iloc[0]["vol_20d"] == pytest.approx(0.40)


class TestLiveTraderSizing:
    def test_target_buy_qty_keeps_fractional_size(self):
        qty = live_trader._target_buy_qty(buy_notional=1250.0, price=3125.0)
        assert qty == pytest.approx(0.4, rel=1e-6)

    def test_target_buy_qty_rejects_invalid_inputs(self):
        assert live_trader._target_buy_qty(buy_notional=0.0, price=100.0) == 0.0
        assert live_trader._target_buy_qty(buy_notional=100.0, price=0.0) == 0.0
        assert live_trader._target_buy_qty(buy_notional=-10.0, price=100.0) == 0.0


def test_get_latest_bars_fetches_all_symbol_batches(monkeypatch):
    requested_batches: list[list[str]] = []

    class _FakeBar:
        def __init__(self, symbol: str):
            self.timestamp = datetime(2026, 4, 16, 20, 0, tzinfo=timezone.utc)
            self.open = 100.0
            self.high = 101.0
            self.low = 99.0
            self.close = 100.5
            self.volume = 1_000_000.0
            self.symbol = symbol

    class _FakeStockBarsRequest:
        def __init__(self, *, symbol_or_symbols, timeframe, start, end, feed):
            self.symbol_or_symbols = list(symbol_or_symbols)
            self.timeframe = timeframe
            self.start = start
            self.end = end
            self.feed = feed

    class _FakeStockHistoricalDataClient:
        def __init__(self, key_id, secret):
            self.key_id = key_id
            self.secret = secret

        def get_stock_bars(self, req):
            requested_batches.append(list(req.symbol_or_symbols))
            return types.SimpleNamespace(
                data={symbol: [_FakeBar(symbol)] for symbol in req.symbol_or_symbols}
            )

    fake_data_module = types.ModuleType("alpaca.data")
    fake_data_module.StockHistoricalDataClient = _FakeStockHistoricalDataClient
    fake_data_module.StockBarsRequest = _FakeStockBarsRequest
    fake_data_module.TimeFrame = types.SimpleNamespace(Day="day")

    fake_enums_module = types.ModuleType("alpaca.data.enums")
    fake_enums_module.DataFeed = types.SimpleNamespace(IEX="iex")

    fake_env_real = types.ModuleType("env_real")
    fake_env_real.ALP_KEY_ID_PAPER = "paper"
    fake_env_real.ALP_SECRET_KEY_PAPER = "secret"

    monkeypatch.setitem(sys.modules, "alpaca.data", fake_data_module)
    monkeypatch.setitem(sys.modules, "alpaca.data.enums", fake_enums_module)
    monkeypatch.setitem(sys.modules, "env_real", fake_env_real)

    symbols = [f"SYM{i}" for i in range(450)]
    result = live_trader._get_latest_bars(symbols, n_days=5, batch_size=200)

    sym_batches = [b for b in requested_batches if b and b[0].startswith("SYM")]
    assert len(sym_batches) == 3
    assert sum(len(batch) for batch in sym_batches) == len(symbols)
    assert sym_batches[0][0] == "SYM0"
    assert sym_batches[-1][-1] == "SYM449"
    assert set(result) == set(symbols)
