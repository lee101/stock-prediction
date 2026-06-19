from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from xgbnew.dataset import build_daily_dataset
from xgbnew.eval_kfold import _split_folds, _train_mask_for_fold


def test_split_folds_covers_span_without_gaps() -> None:
    folds = _split_folds(date(2026, 1, 1), date(2026, 1, 20), 6)

    assert folds[0][0] == date(2026, 1, 1)
    assert folds[-1][1] == date(2026, 1, 20)
    for prev, nxt in zip(folds[:-1], folds[1:], strict=True):
        assert nxt[0] == prev[1] + timedelta(days=1)

    sizes = [(end - start).days + 1 for start, end in folds]
    assert max(sizes) - min(sizes) <= 1


def test_k_contiguous_train_mask_excludes_fold_and_embargo() -> None:
    dates = pd.Series(pd.date_range("2026-01-01", periods=10, freq="D").date)
    mask = _train_mask_for_fold(
        dates,
        fold_start=date(2026, 1, 4),
        fold_end=date(2026, 1, 6),
        cv_mode="k_contiguous",
        embargo_days=1,
    )

    kept = set(dates[mask])
    assert kept == {
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 8),
        date(2026, 1, 9),
        date(2026, 1, 10),
    }


def test_expanding_train_mask_uses_only_pre_fold_history() -> None:
    dates = pd.Series(pd.date_range("2026-01-01", periods=10, freq="D").date)
    mask = _train_mask_for_fold(
        dates,
        fold_start=date(2026, 1, 4),
        fold_end=date(2026, 1, 6),
        cv_mode="expanding",
        embargo_days=1,
    )

    assert set(dates[mask]) == {date(2026, 1, 1), date(2026, 1, 2)}


def _write_daily_csv(path: Path, n: int = 180) -> None:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.6, n))
    close = np.clip(close, 10.0, None)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(2e6, 5e6, n),
        }
    )
    frame.to_csv(path, index=False)


def test_build_daily_dataset_can_attach_chronos_to_training_split(tmp_path: Path) -> None:
    _write_daily_csv(tmp_path / "AAA.csv")
    cache = {
        date(2021, 4, 1): {
            "AAA": {
                "oc_return_pct": 1.25,
                "cc_return_pct": 0.75,
                "pred_open": 100.0,
                "pred_close": 101.0,
            }
        }
    }
    common = dict(
        data_root=tmp_path,
        symbols=["AAA"],
        train_start=date(2021, 3, 1),
        train_end=date(2021, 5, 31),
        val_start=date(2021, 6, 1),
        val_end=date(2021, 6, 30),
        test_start=date(2021, 7, 1),
        test_end=date(2021, 8, 31),
        chronos_cache=cache,
        min_dollar_vol=1.0,
    )

    legacy_train, _, _ = build_daily_dataset(**common)
    chronos_train, _, _ = build_daily_dataset(
        **common,
        attach_chronos_to_train=True,
    )

    assert legacy_train["chronos_available"].sum() == 0.0
    matched = chronos_train[chronos_train["date"] == date(2021, 4, 1)]
    assert not matched.empty
    assert matched["chronos_available"].iloc[0] == 1.0
    assert matched["chronos_oc_return"].iloc[0] == 1.25
