"""Tests for xgbnew.symbol_kfold.

Covers the bucketing logic (every symbol in exactly one bucket, buckets
are roughly equal-sized, liquidity mode respects the anchor-day ordering)
and the model-path resolver (globs + comma-lists collapse to unique).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew import symbol_kfold  # noqa: E402
from xgbnew.symbol_kfold import _bucket_symbols, _resolve_model_paths  # noqa: E402


def _mock_oos_df(syms, liquidities=None, vols=None):
    rows = []
    for i, s in enumerate(syms):
        rows.append({
            "date": date(2025, 1, 2),
            "symbol": s,
            "dolvol_20d_log": float(liquidities[i] if liquidities else i),
            "vol_20d":        float(vols[i]        if vols        else i),
        })
    return pd.DataFrame(rows)


def test_alpha_buckets_cover_all_symbols_no_overlap():
    syms = [f"SYM{i:03d}" for i in range(20)]
    df = _mock_oos_df(syms)
    out = _bucket_symbols(syms, df, mode="alpha", n_buckets=4)
    collected = []
    for lst in out.values():
        collected.extend(lst)
    assert sorted(collected) == sorted(syms), "every symbol exactly once"
    # Each bucket is 4 or 5 symbols for 20/4.
    sizes = sorted(len(v) for v in out.values())
    assert sizes == [5, 5, 5, 5]


def test_liquidity_buckets_ordered_by_dolvol():
    syms = ["A", "B", "C", "D", "E", "F", "G", "H"]
    liq = [5.0, 1.0, 8.0, 3.0, 7.0, 2.0, 4.0, 6.0]
    df = _mock_oos_df(syms, liquidities=liq)
    out = _bucket_symbols(syms, df, mode="liquidity", n_buckets=4)
    # Bucket 0 = lowest liquidity (sorted asc). We check ordering inside
    # the bucket list is asc by sorted-liquidity order.
    b0 = out[0]
    b3 = out[3]
    # Lowest two liquidities are B(1.0), F(2.0) — bucket 0 should hold these.
    assert set(b0) == {"B", "F"}
    # Highest two are C(8.0), E(7.0).
    assert set(b3) == {"C", "E"}


def test_n_buckets_lt_2_raises():
    syms = ["A", "B"]
    df = _mock_oos_df(syms)
    with pytest.raises(ValueError, match="n_buckets"):
        _bucket_symbols(syms, df, mode="alpha", n_buckets=1)


def test_not_enough_symbols_for_bucket_count_raises():
    syms = ["A", "B"]
    df = _mock_oos_df(syms)
    with pytest.raises(ValueError, match="can't make"):
        _bucket_symbols(syms, df, mode="alpha", n_buckets=4)


def test_liquidity_needs_dolvol_col():
    syms = ["A", "B", "C", "D"]
    df = pd.DataFrame({
        "date": [date(2025, 1, 2)] * 4,
        "symbol": syms,
    })
    with pytest.raises(ValueError, match="dolvol_20d_log"):
        _bucket_symbols(syms, df, mode="liquidity", n_buckets=2)


def test_volatility_needs_vol20d_col():
    syms = ["A", "B", "C", "D"]
    df = pd.DataFrame({
        "date": [date(2025, 1, 2)] * 4,
        "symbol": syms,
    })
    with pytest.raises(ValueError, match="vol_20d"):
        _bucket_symbols(syms, df, mode="volatility", n_buckets=2)


def test_resolve_model_paths_glob(tmp_path):
    # Create some dummy pkl files.
    (tmp_path / "a.pkl").write_text("x")
    (tmp_path / "b.pkl").write_text("x")
    (tmp_path / "c.pkl").write_text("x")
    pattern = str(tmp_path / "*.pkl")
    resolved = _resolve_model_paths(pattern)
    assert len(resolved) == 3
    assert all(p.name.endswith(".pkl") for p in resolved)


def test_resolve_model_paths_comma_and_dedup(tmp_path):
    (tmp_path / "a.pkl").write_text("x")
    (tmp_path / "b.pkl").write_text("x")
    # Comma list with duplicate — dedup by resolved path.
    spec = f"{tmp_path}/a.pkl,{tmp_path}/b.pkl,{tmp_path}/a.pkl"
    resolved = _resolve_model_paths(spec)
    assert len(resolved) == 2


def test_unknown_bucket_mode_raises():
    syms = ["A", "B", "C", "D"]
    df = _mock_oos_df(syms)
    with pytest.raises(ValueError, match="unknown bucket mode"):
        _bucket_symbols(syms, df, mode="nonsense", n_buckets=2)


def _kfold_oos_df() -> pd.DataFrame:
    days = pd.date_range("2026-01-02", periods=35, freq="B").date
    rows = []
    for day in days:
        for symbol in ("A", "B", "C", "D"):
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "ret_1d": 0.0,
                    "dolvol_20d_log": 10.0,
                    "vol_20d": 0.2,
                    "latent_0": 0.1,
                    "latent_1": 0.2,
                    "fm_available": 1.0,
                }
            )
    return pd.DataFrame(rows)


class _FakeFMModel:
    feature_cols: ClassVar[list[str]] = [
        "ret_1d",
        "latent_0",
        "latent_1",
        "fm_available",
    ]

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise AssertionError(f"missing feature columns: {missing}")
        return pd.Series(0.75, index=df.index)


def test_run_kfold_requires_fm_latents_for_fm_feature_model(monkeypatch, tmp_path):
    path = tmp_path / "alltrain_seed0.pkl"
    path.write_bytes(b"model")
    monkeypatch.setattr(symbol_kfold.XGBStockModel, "load", staticmethod(lambda _path: _FakeFMModel()))
    monkeypatch.setattr(
        symbol_kfold,
        "build_daily_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dataset built")),
    )

    with pytest.raises(ValueError, match="require FM latents"):
        symbol_kfold.run_kfold(
            symbols=["A", "B"],
            data_root=tmp_path / "data",
            model_paths=[path],
            train_start=date(2025, 1, 1),
            train_end=date(2025, 12, 31),
            oos_start=date(2026, 1, 2),
            oos_end=date(2026, 3, 1),
            window_days=30,
            stride_days=7,
            leverage=1.0,
            min_score=0.0,
            hold_through=True,
            top_n=1,
            fee_regime="deploy",
            n_buckets=2,
            bucket_mode="alpha",
        )


def test_run_kfold_rejects_fm_n_latents_below_model_requirement(monkeypatch, tmp_path):
    path = tmp_path / "alltrain_seed0.pkl"
    path.write_bytes(b"model")
    latents_path = tmp_path / "latents.parquet"
    latents_path.write_bytes(b"latents")
    monkeypatch.setattr(symbol_kfold.XGBStockModel, "load", staticmethod(lambda _path: _FakeFMModel()))
    monkeypatch.setattr(
        symbol_kfold,
        "load_fm_latents",
        lambda _path: (_ for _ in ()).throw(AssertionError("latents loaded")),
    )

    with pytest.raises(ValueError, match="smaller than model feature_cols require \\(2\\)"):
        symbol_kfold.run_kfold(
            symbols=["A", "B"],
            data_root=tmp_path / "data",
            model_paths=[path],
            train_start=date(2025, 1, 1),
            train_end=date(2025, 12, 31),
            oos_start=date(2026, 1, 2),
            oos_end=date(2026, 3, 1),
            window_days=30,
            stride_days=7,
            leverage=1.0,
            min_score=0.0,
            hold_through=True,
            top_n=1,
            fee_regime="deploy",
            n_buckets=2,
            bucket_mode="alpha",
            fm_latents_path=latents_path,
            fm_n_latents=1,
        )


def test_run_kfold_attaches_fm_latents(monkeypatch, tmp_path):
    path = tmp_path / "alltrain_seed0.pkl"
    path.write_bytes(b"model")
    latents_path = tmp_path / "latents.parquet"
    latents_path.write_bytes(b"latents")
    fm_df = pd.DataFrame(
        {
            "symbol": ["A"],
            "date": [date(2026, 1, 2)],
            "latent_0": [0.1],
            "latent_1": [0.2],
        }
    )
    build_calls = []
    monkeypatch.setattr(symbol_kfold.XGBStockModel, "load", staticmethod(lambda _path: _FakeFMModel()))
    monkeypatch.setattr(symbol_kfold, "load_fm_latents", lambda _path: fm_df)

    def _fake_build_daily_dataset(**kwargs):
        build_calls.append(kwargs)
        return pd.DataFrame(), pd.DataFrame(), _kfold_oos_df()

    monkeypatch.setattr(symbol_kfold, "build_daily_dataset", _fake_build_daily_dataset)
    monkeypatch.setattr(
        symbol_kfold,
        "simulate",
        lambda *_args, **_kwargs: SimpleNamespace(
            total_return_pct=10.0,
            sortino_ratio=2.0,
            max_drawdown_pct=3.0,
            worst_intraday_dd_pct=1.0,
        ),
    )

    results = symbol_kfold.run_kfold(
        symbols=["A", "B", "C", "D"],
        data_root=tmp_path / "data",
        model_paths=[path],
        train_start=date(2025, 1, 1),
        train_end=date(2025, 12, 31),
        oos_start=date(2026, 1, 2),
        oos_end=date(2026, 3, 1),
        window_days=30,
        stride_days=7,
        leverage=1.0,
        min_score=0.0,
        hold_through=True,
        top_n=1,
        fee_regime="deploy",
        n_buckets=2,
        bucket_mode="alpha",
        fm_latents_path=latents_path,
        fm_n_latents=2,
    )

    assert len(results) == 3
    assert build_calls
    assert build_calls[0]["fm_latents"] is fm_df
    assert build_calls[0]["fm_n_latents"] == 2
