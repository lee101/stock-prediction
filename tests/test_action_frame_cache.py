from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.action_frame_cache import build_action_cache_key, load_or_generate_action_frame


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
                utc=True,
            ),
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, 11.0, 12.0],
        }
    )


def test_build_action_cache_key_changes_with_checkpoint_and_frame(tmp_path: Path) -> None:
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_a.write_text("a")
    ckpt_b.write_text("b")

    frame_a = _frame()
    frame_b = _frame()
    frame_b.loc[1, "feature_b"] = 99.0

    key_a = build_action_cache_key(
        symbol="BTCUSD",
        checkpoint_path=ckpt_a,
        frame=frame_a,
        feature_columns=["feature_a", "feature_b"],
        normalizer={"mean": [0.0]},
        sequence_length=96,
        horizon=1,
    )
    key_b = build_action_cache_key(
        symbol="BTCUSD",
        checkpoint_path=ckpt_b,
        frame=frame_a,
        feature_columns=["feature_a", "feature_b"],
        normalizer={"mean": [0.0]},
        sequence_length=96,
        horizon=1,
    )
    key_c = build_action_cache_key(
        symbol="BTCUSD",
        checkpoint_path=ckpt_a,
        frame=frame_b,
        feature_columns=["feature_a", "feature_b"],
        normalizer={"mean": [0.0]},
        sequence_length=96,
        horizon=1,
    )

    assert key_a != key_b
    assert key_a != key_c


def test_load_or_generate_action_frame_reuses_saved_actions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("weights")
    frame = _frame()
    calls = {"count": 0}

    def _generate() -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame(
            {
                "timestamp": frame["timestamp"],
                "symbol": ["BTCUSD"] * len(frame),
                "buy_price": [100.0, 101.0, 102.0],
            }
        )

    first, first_cached = load_or_generate_action_frame(
        cache_root=tmp_path / "cache",
        symbol="BTCUSD",
        checkpoint_path=checkpoint,
        frame=frame,
        feature_columns=["feature_a", "feature_b"],
        normalizer={"mean": [0.0]},
        sequence_length=96,
        horizon=1,
        generator=_generate,
    )
    second, second_cached = load_or_generate_action_frame(
        cache_root=tmp_path / "cache",
        symbol="BTCUSD",
        checkpoint_path=checkpoint,
        frame=frame,
        feature_columns=["feature_a", "feature_b"],
        normalizer={"mean": [0.0]},
        sequence_length=96,
        horizon=1,
        generator=_generate,
    )

    assert first_cached is False
    assert second_cached is True
    assert calls["count"] == 1
    pd.testing.assert_frame_equal(first, second)
