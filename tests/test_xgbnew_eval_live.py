"""Tests for xgbnew eval_multiwindow and live_trader modules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.eval_multiwindow import _monthly_return, main as eval_main
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
        import pickle
        # Minimal model stub
        model = XGBStockModel.__new__(XGBStockModel)
        model._fitted = False
        df = score_all_symbols([], tmp_path / "nodata", model)  # type: ignore
        assert len(df) == 0 or isinstance(df, pd.DataFrame)
