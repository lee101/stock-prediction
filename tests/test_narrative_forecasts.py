from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from binanceneural.narrative_forecasts import (
    NarrativeSummaryRecord,
    apply_narrative_overlay,
    generate_summary_record,
    resolve_horizon_summary_cache_dir,
    summarize_forecast_row,
)


def test_module_all_exports_public_narrative_helpers() -> None:
    from binanceneural import narrative_forecasts as module

    assert "summarize_forecast_row" in module.__all__
    assert "apply_narrative_overlay" in module.__all__


def _make_history(symbol: str = "DOGEUSD", rows: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series([100.0 + (idx * 0.1) for idx in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1000.0 + pd.Series(range(rows), dtype="float64"),
        }
    )


def test_generate_summary_record_heuristic_has_required_sections() -> None:
    history = _make_history()
    issued_at = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    target_ts = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)

    record = generate_summary_record(
        backend="heuristic",
        model=None,
        symbol="DOGEUSD",
        history=history,
        issued_at=issued_at,
        target_timestamp=target_ts,
        horizon_hours=4,
        context_hours=24 * 7,
    )

    assert record.provider == "heuristic"
    assert "FACTUAL SUMMARY:" in record.summary_text
    assert "PREDICTIVE SIGNALS:" in record.summary_text
    assert -1.0 <= record.signal_strength <= 1.0
    assert 0.0 <= record.confidence <= 1.0


def test_resolve_horizon_summary_cache_dir_matches_default_and_custom_layouts(tmp_path: Path) -> None:
    cache_root = tmp_path / "forecast_cache"

    default_dir = resolve_horizon_summary_cache_dir(
        cache_root=cache_root,
        horizon=4,
        summary_cache_root=None,
    )
    custom_dir = resolve_horizon_summary_cache_dir(
        cache_root=cache_root,
        horizon=4,
        summary_cache_root=tmp_path / "summary_cache",
    )

    assert default_dir == cache_root / "_narrative_summaries" / "h4"
    assert custom_dir == tmp_path / "summary_cache" / "h4"


def test_apply_narrative_overlay_adds_adjusted_and_base_columns(tmp_path: Path) -> None:
    history = _make_history()
    forecast = pd.DataFrame(
        {
            "timestamp": [pd.to_datetime(history["timestamp"].iloc[-1], utc=True)],
            "symbol": ["DOGEUSD"],
            "issued_at": [pd.to_datetime(history["timestamp"].iloc[-2], utc=True)],
            "target_timestamp": [pd.to_datetime(history["timestamp"].iloc[-1], utc=True)],
            "horizon_hours": [1],
            "predicted_close_p50": [102.0],
            "predicted_close_p10": [101.6],
            "predicted_close_p90": [102.4],
            "predicted_high_p50": [102.5],
            "predicted_low_p50": [101.5],
        }
    )

    out = apply_narrative_overlay(
        forecast,
        symbol="DOGEUSD",
        history=history,
        backend="heuristic",
        model=None,
        forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
        summary_cache_dir=tmp_path / "summaries" / "h1",
        context_hours=24 * 7,
        force_rebuild=True,
    )

    assert "base_predicted_close_p50" in out.columns
    assert "narrative_summary" in out.columns
    assert "narrative_signal_strength" in out.columns
    assert out.loc[0, "predicted_high_p50"] >= out.loc[0, "predicted_close_p50"]
    assert out.loc[0, "predicted_low_p50"] <= out.loc[0, "predicted_close_p50"]
    assert (tmp_path / "summaries" / "h1" / "DOGEUSD.parquet").exists()


def test_apply_narrative_overlay_tolerates_corrupt_summary_cache(tmp_path: Path) -> None:
    history = _make_history()
    forecast = pd.DataFrame(
        {
            "timestamp": [pd.to_datetime(history["timestamp"].iloc[-1], utc=True)],
            "symbol": ["DOGEUSD"],
            "issued_at": [pd.to_datetime(history["timestamp"].iloc[-2], utc=True)],
            "target_timestamp": [pd.to_datetime(history["timestamp"].iloc[-1], utc=True)],
            "horizon_hours": [1],
            "predicted_close_p50": [102.0],
            "predicted_high_p50": [102.5],
            "predicted_low_p50": [101.5],
        }
    )
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    corrupt_path = summary_dir / "DOGEUSD.parquet"
    corrupt_path.write_text("not a parquet file", encoding="utf-8")

    with patch("binanceneural.narrative_forecasts.pd.read_parquet", side_effect=ValueError("corrupt parquet")):
        out = apply_narrative_overlay(
            forecast,
            symbol="DOGEUSD",
            history=history,
            backend="heuristic",
            model=None,
            forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert "narrative_summary" in out.columns
    assert out.loc[0, "narrative_provider"] == "heuristic"


def test_summarize_forecast_row_tolerates_summary_cache_write_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    history = _make_history()
    row = {
        "timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "symbol": "DOGEUSD",
        "issued_at": pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
        "target_timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "horizon_hours": 1,
    }

    with patch("binanceneural.narrative_forecasts.pd.DataFrame.to_parquet", side_effect=OSError("disk full")), caplog.at_level("WARNING"):
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row=row,
            backend="heuristic",
            model=None,
            summary_cache_dir=tmp_path / "summaries" / "h1",
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert record.provider == "heuristic"
    assert "FACTUAL SUMMARY:" in record.summary_text
    assert any("Failed to write narrative summary cache" in message for message in caplog.messages)


def test_apply_narrative_overlay_tolerates_summary_cache_write_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    history = _make_history()
    forecast = pd.DataFrame(
        {
            "timestamp": [pd.to_datetime(history["timestamp"].iloc[-1], utc=True)],
            "symbol": ["DOGEUSD"],
            "issued_at": [pd.to_datetime(history["timestamp"].iloc[-2], utc=True)],
            "target_timestamp": [pd.to_datetime(history["timestamp"].iloc[-1], utc=True)],
            "horizon_hours": [1],
            "predicted_close_p50": [102.0],
            "predicted_high_p50": [102.5],
            "predicted_low_p50": [101.5],
        }
    )

    with patch("binanceneural.narrative_forecasts.pd.DataFrame.to_parquet", side_effect=OSError("disk full")), caplog.at_level("WARNING"):
        out = apply_narrative_overlay(
            forecast,
            symbol="DOGEUSD",
            history=history,
            backend="heuristic",
            model=None,
            forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
            summary_cache_dir=tmp_path / "summaries" / "h1",
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert "narrative_summary" in out.columns
    assert out.loc[0, "narrative_provider"] == "heuristic"
    assert any("Failed to write narrative summary cache" in message for message in caplog.messages)


def test_apply_narrative_overlay_skips_invalid_forecast_rows_and_keeps_valid_rows(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    history = _make_history()
    valid_target = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    valid_issued = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    invalid_target = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    invalid_issued = pd.to_datetime(history["timestamp"].iloc[-3], utc=True)
    forecast = pd.DataFrame(
        {
            "timestamp": [valid_target, invalid_target],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [valid_issued, invalid_issued],
            "target_timestamp": [valid_target, invalid_target],
            "horizon_hours": [1, "not-an-int"],
            "predicted_close_p50": [102.0, 101.8],
            "predicted_high_p50": [102.5, 102.3],
            "predicted_low_p50": [101.5, 101.3],
        }
    )

    with caplog.at_level("WARNING"):
        out = apply_narrative_overlay(
            forecast,
            symbol="DOGEUSD",
            history=history,
            backend="heuristic",
            model=None,
            forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
            summary_cache_dir=tmp_path / "summaries" / "h1",
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert len(out) == 2
    assert out.loc[0, "narrative_provider"] == "heuristic"
    assert out.loc[0, "narrative_summary"]
    assert out.loc[1, "predicted_close_p50"] == 101.8
    assert pd.isna(out.loc[1, "narrative_provider"])
    assert any("Skipping narrative overlay for DOGEUSD row" in message for message in caplog.messages)


def test_apply_narrative_overlay_loads_summary_cache_once_per_run(tmp_path: Path) -> None:
    history = _make_history()
    forecast = pd.DataFrame(
        {
            "timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [
                pd.to_datetime(history["timestamp"].iloc[-3], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
            ],
            "target_timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "horizon_hours": [1, 1],
            "predicted_close_p50": [101.8, 102.0],
            "predicted_high_p50": [102.3, 102.5],
            "predicted_low_p50": [101.3, 101.5],
        }
    )

    with patch(
        "binanceneural.narrative_forecasts.NarrativeSummaryCache.load",
        return_value=pd.DataFrame(),
    ) as mock_load:
        out = apply_narrative_overlay(
            forecast,
            symbol="DOGEUSD",
            history=history,
            backend="heuristic",
            model=None,
            forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
            summary_cache_dir=tmp_path / "summaries" / "h1",
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert len(out) == 2
    assert mock_load.call_count == 1


def test_apply_narrative_overlay_reuses_prepared_context_per_issued_at(tmp_path: Path) -> None:
    import binanceneural.narrative_forecasts as narrative_forecasts

    history = _make_history()
    issued_at = pd.to_datetime(history["timestamp"].iloc[-3], utc=True)
    forecast = pd.DataFrame(
        {
            "timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [issued_at, issued_at],
            "target_timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "horizon_hours": [1, 2],
            "predicted_close_p50": [101.8, 102.0],
            "predicted_high_p50": [102.3, 102.5],
            "predicted_low_p50": [101.3, 101.5],
        }
    )

    with patch(
        "binanceneural.narrative_forecasts._history_slice",
        wraps=narrative_forecasts._history_slice,
    ) as mock_history_slice:
        out = apply_narrative_overlay(
            forecast,
            symbol="DOGEUSD",
            history=history,
            backend="heuristic",
            model=None,
            forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
            summary_cache_dir=tmp_path / "summaries" / "h1",
            context_hours=24 * 7,
            force_rebuild=True,
        )

    assert len(out) == 2
    assert mock_history_slice.call_count == 1


def test_apply_narrative_overlay_writes_summary_cache_once_per_run(tmp_path: Path) -> None:
    import binanceneural.narrative_forecasts as narrative_forecasts

    history = _make_history()
    forecast = pd.DataFrame(
        {
            "timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [
                pd.to_datetime(history["timestamp"].iloc[-3], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
            ],
            "target_timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "horizon_hours": [1, 1],
            "predicted_close_p50": [101.8, 102.0],
            "predicted_high_p50": [102.3, 102.5],
            "predicted_low_p50": [101.3, 101.5],
        }
    )

    with patch("binanceneural.narrative_forecasts.NarrativeSummaryCache.write", autospec=True) as mock_write, \
         patch(
             "binanceneural.narrative_forecasts._merge_summary_cache_frame",
             wraps=narrative_forecasts._merge_summary_cache_frame,
         ) as mock_merge:
        out = apply_narrative_overlay(
            forecast,
            symbol="DOGEUSD",
            history=history,
            backend="heuristic",
            model=None,
            forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
            summary_cache_dir=tmp_path / "summaries" / "h1",
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert len(out) == 2
    assert mock_write.call_count == 1
    assert mock_merge.call_count == 0


def test_apply_narrative_overlay_skips_summary_cache_write_when_fully_cached(tmp_path: Path) -> None:
    history = _make_history()
    forecast = pd.DataFrame(
        {
            "timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [
                pd.to_datetime(history["timestamp"].iloc[-3], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
            ],
            "target_timestamp": [
                pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
                pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
            ],
            "horizon_hours": [1, 1],
            "predicted_close_p50": [101.8, 102.0],
            "predicted_high_p50": [102.3, 102.5],
            "predicted_low_p50": [101.3, 101.5],
        }
    )
    summary_dir = tmp_path / "summaries" / "h1"

    first = apply_narrative_overlay(
        forecast,
        symbol="DOGEUSD",
        history=history,
        backend="heuristic",
        model=None,
        forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
        summary_cache_dir=summary_dir,
        context_hours=24 * 7,
        force_rebuild=False,
    )

    with patch("binanceneural.narrative_forecasts.NarrativeSummaryCache.write", autospec=True) as mock_write:
        second = apply_narrative_overlay(
            forecast,
            symbol="DOGEUSD",
            history=history,
            backend="heuristic",
            model=None,
            forecast_cache_dir=tmp_path / "forecast_cache" / "h1",
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert len(first) == 2
    assert len(second) == 2
    assert mock_write.call_count == 0


def test_summarize_forecast_row_reuses_cached_record_without_rewrite(tmp_path: Path) -> None:
    history = _make_history()
    row = {
        "timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "symbol": "DOGEUSD",
        "issued_at": pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
        "target_timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "horizon_hours": 1,
    }
    summary_dir = tmp_path / "summaries" / "h1"

    first = summarize_forecast_row(
        symbol="DOGEUSD",
        history=history,
        row=row,
        backend="heuristic",
        model=None,
        summary_cache_dir=summary_dir,
        context_hours=24 * 7,
        force_rebuild=False,
    )

    with patch("binanceneural.narrative_forecasts.NarrativeSummaryCache.write", autospec=True) as mock_write, patch(
        "binanceneural.narrative_forecasts.generate_summary_record",
        side_effect=AssertionError("cache hit should not rebuild summary"),
    ):
        second = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row=row,
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert second.timestamp == first.timestamp
    assert second.summary_text == first.summary_text
    assert mock_write.call_count == 0


def test_summarize_forecast_row_force_rebuild_bypasses_cached_record(tmp_path: Path) -> None:
    history = _make_history()
    row = {
        "timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "symbol": "DOGEUSD",
        "issued_at": pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
        "target_timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "horizon_hours": 1,
    }
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    cache_path = summary_dir / "DOGEUSD.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": row["timestamp"],
                "symbol": "DOGEUSD",
                "issued_at": row["issued_at"],
                "target_timestamp": row["target_timestamp"],
                "horizon_hours": 1,
                "provider": "heuristic",
                "model": "heuristic",
                "factual_summary": "old factual",
                "predictive_signals": "old predictive",
                "summary_text": "old summary",
                "signal_strength": 0.1,
                "confidence": 0.2,
                "expected_move_pct": 0.01,
                "recent_return_24h": 0.01,
                "recent_return_168h": 0.02,
                "realized_vol_24h": 0.03,
                "volume_z_24h": 0.4,
                "generated_at": pd.Timestamp("2026-01-05T00:00:00Z"),
            }
        ]
    ).to_parquet(cache_path, index=False)

    rebuilt = NarrativeSummaryRecord(
        timestamp=pd.to_datetime(row["timestamp"], utc=True),
        symbol="DOGEUSD",
        issued_at=pd.to_datetime(row["issued_at"], utc=True),
        target_timestamp=pd.to_datetime(row["target_timestamp"], utc=True),
        horizon_hours=1,
        provider="heuristic",
        model="heuristic",
        factual_summary="rebuilt factual",
        predictive_signals="rebuilt predictive",
        summary_text="rebuilt summary",
        signal_strength=0.6,
        confidence=0.7,
        expected_move_pct=0.02,
        recent_return_24h=0.03,
        recent_return_168h=0.04,
        realized_vol_24h=0.05,
        volume_z_24h=0.6,
        generated_at=pd.Timestamp("2026-01-06T00:00:00Z"),
    )

    with patch(
        "binanceneural.narrative_forecasts.generate_summary_record",
        return_value=rebuilt,
    ) as mock_generate:
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row=row,
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=True,
        )

    assert record == rebuilt
    mock_generate.assert_called_once()
    written = pd.read_parquet(cache_path)
    assert written.iloc[-1]["summary_text"] == "rebuilt summary"


def test_summarize_forecast_row_tolerates_malformed_cached_row(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    history = _make_history()
    row = {
        "timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "symbol": "DOGEUSD",
        "issued_at": pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
        "target_timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "horizon_hours": 1,
    }
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [row["timestamp"]],
            "symbol": ["DOGEUSD"],
            "issued_at": [row["issued_at"]],
            "target_timestamp": [row["target_timestamp"]],
            "horizon_hours": ["not-an-int"],
        }
    ).to_parquet(summary_dir / "DOGEUSD.parquet", index=False)

    with patch(
        "binanceneural.narrative_forecasts.generate_summary_record",
        wraps=generate_summary_record,
    ) as mock_generate, caplog.at_level("WARNING"):
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row=row,
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert record.provider == "heuristic"
    assert "FACTUAL SUMMARY:" in record.summary_text
    assert mock_generate.call_count == 1
    assert any("Ignoring invalid narrative summary cache row" in message for message in caplog.messages)


def test_summarize_forecast_row_reconstructs_missing_summary_text_from_cached_fields(tmp_path: Path) -> None:
    history = _make_history()
    row = {
        "timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "symbol": "DOGEUSD",
        "issued_at": pd.to_datetime(history["timestamp"].iloc[-2], utc=True),
        "target_timestamp": pd.to_datetime(history["timestamp"].iloc[-1], utc=True),
        "horizon_hours": 1,
    }
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [row["timestamp"]],
            "symbol": ["DOGEUSD"],
            "issued_at": [row["issued_at"]],
            "target_timestamp": [row["target_timestamp"]],
            "horizon_hours": [1],
            "factual_summary": ["Cached factual summary."],
            "predictive_signals": ["Cached predictive signals."],
            "summary_text": [""],
        }
    ).to_parquet(summary_dir / "DOGEUSD.parquet", index=False)

    with patch("binanceneural.narrative_forecasts.generate_summary_record", side_effect=AssertionError("cache hit should not rebuild summary")):
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row=row,
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert record.factual_summary == "Cached factual summary."
    assert record.predictive_signals == "Cached predictive signals."
    assert record.summary_text == "FACTUAL SUMMARY:\nCached factual summary.\n\nPREDICTIVE SIGNALS:\nCached predictive signals."


def test_summarize_forecast_row_matches_cache_on_timestamp_and_horizon(tmp_path: Path) -> None:
    history = _make_history()
    target_ts = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    issued_at = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [target_ts, target_ts],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [issued_at, issued_at],
            "target_timestamp": [target_ts, target_ts],
            "horizon_hours": [1, 4],
            "factual_summary": ["One-hour factual summary.", "Four-hour factual summary."],
            "predictive_signals": ["One-hour predictive signals.", "Four-hour predictive signals."],
            "summary_text": ["", ""],
        }
    ).to_parquet(summary_dir / "DOGEUSD.parquet", index=False)

    with patch("binanceneural.narrative_forecasts.generate_summary_record", side_effect=AssertionError("matching horizon cache hit should not rebuild summary")):
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row={
                "timestamp": target_ts,
                "symbol": "DOGEUSD",
                "issued_at": issued_at,
                "target_timestamp": target_ts,
                "horizon_hours": 4,
            },
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert record.horizon_hours == 4
    assert record.factual_summary == "Four-hour factual summary."
    assert record.predictive_signals == "Four-hour predictive signals."


def test_summarize_forecast_row_reuses_legacy_cache_without_horizon_column(tmp_path: Path) -> None:
    history = _make_history()
    target_ts = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    issued_at = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [target_ts],
            "symbol": ["DOGEUSD"],
            "issued_at": [issued_at],
            "target_timestamp": [target_ts],
            "factual_summary": ["Legacy factual summary."],
            "predictive_signals": ["Legacy predictive signals."],
            "summary_text": [""],
        }
    ).to_parquet(summary_dir / "DOGEUSD.parquet", index=False)

    with patch(
        "binanceneural.narrative_forecasts.generate_summary_record",
        side_effect=AssertionError("legacy cache hit should not rebuild summary"),
    ):
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row={
                "timestamp": target_ts,
                "symbol": "DOGEUSD",
                "issued_at": issued_at,
                "target_timestamp": target_ts,
                "horizon_hours": 4,
            },
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert record.horizon_hours == 4
    assert record.factual_summary == "Legacy factual summary."
    assert record.predictive_signals == "Legacy predictive signals."


def test_summarize_forecast_row_reuses_legacy_cache_row_in_mixed_horizon_file(tmp_path: Path) -> None:
    history = _make_history()
    target_ts = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    issued_at = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    other_ts = pd.to_datetime(history["timestamp"].iloc[-3], utc=True)
    other_issued_at = pd.to_datetime(history["timestamp"].iloc[-4], utc=True)
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [target_ts, other_ts],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [issued_at, other_issued_at],
            "target_timestamp": [target_ts, other_ts],
            "horizon_hours": [pd.NA, 4],
            "factual_summary": ["Legacy factual summary.", "Four-hour factual summary."],
            "predictive_signals": ["Legacy predictive signals.", "Four-hour predictive signals."],
            "summary_text": ["", ""],
        }
    ).to_parquet(summary_dir / "DOGEUSD.parquet", index=False)

    with patch(
        "binanceneural.narrative_forecasts.generate_summary_record",
        side_effect=AssertionError("mixed legacy cache hit should not rebuild summary"),
    ):
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row={
                "timestamp": target_ts,
                "symbol": "DOGEUSD",
                "issued_at": issued_at,
                "target_timestamp": target_ts,
                "horizon_hours": 1,
            },
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert record.horizon_hours == 1
    assert record.factual_summary == "Legacy factual summary."
    assert record.predictive_signals == "Legacy predictive signals."


def test_summarize_forecast_row_prefers_exact_horizon_over_legacy_row_in_mixed_file(tmp_path: Path) -> None:
    history = _make_history()
    target_ts = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    issued_at = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    summary_dir = tmp_path / "summaries" / "h1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [target_ts, target_ts],
            "symbol": ["DOGEUSD", "DOGEUSD"],
            "issued_at": [issued_at, issued_at],
            "target_timestamp": [target_ts, target_ts],
            "horizon_hours": [pd.NA, 4],
            "factual_summary": ["Legacy factual summary.", "Exact four-hour factual summary."],
            "predictive_signals": ["Legacy predictive signals.", "Exact four-hour predictive signals."],
            "summary_text": ["", ""],
        }
    ).to_parquet(summary_dir / "DOGEUSD.parquet", index=False)

    with patch(
        "binanceneural.narrative_forecasts.generate_summary_record",
        side_effect=AssertionError("mixed cache hit should not rebuild summary"),
    ):
        record = summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row={
                "timestamp": target_ts,
                "symbol": "DOGEUSD",
                "issued_at": issued_at,
                "target_timestamp": target_ts,
                "horizon_hours": 4,
            },
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=False,
        )

    assert record.horizon_hours == 4
    assert record.factual_summary == "Exact four-hour factual summary."
    assert record.predictive_signals == "Exact four-hour predictive signals."


def test_summarize_forecast_row_preserves_distinct_horizons_in_shared_cache(tmp_path: Path) -> None:
    history = _make_history()
    target_ts = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    issued_at = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    summary_dir = tmp_path / "summaries" / "shared"

    for horizon_hours in (1, 4):
        summarize_forecast_row(
            symbol="DOGEUSD",
            history=history,
            row={
                "timestamp": target_ts,
                "symbol": "DOGEUSD",
                "issued_at": issued_at,
                "target_timestamp": target_ts,
                "horizon_hours": horizon_hours,
            },
            backend="heuristic",
            model=None,
            summary_cache_dir=summary_dir,
            context_hours=24 * 7,
            force_rebuild=True,
        )

    cached = pd.read_parquet(summary_dir / "DOGEUSD.parquet")
    assert len(cached) == 2
    assert sorted(cached["horizon_hours"].tolist()) == [1, 4]


def test_summarize_forecast_row_dedupes_string_and_numeric_horizons(tmp_path: Path) -> None:
    history = _make_history()
    target_ts = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    issued_at = pd.to_datetime(history["timestamp"].iloc[-2], utc=True)
    summary_dir = tmp_path / "summaries" / "shared"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [target_ts],
            "symbol": ["DOGEUSD"],
            "issued_at": [issued_at],
            "target_timestamp": [target_ts],
            "horizon_hours": ["4"],
            "factual_summary": ["Legacy string-horizon factual summary."],
            "predictive_signals": ["Legacy string-horizon predictive signals."],
            "summary_text": [""],
        }
    ).to_parquet(summary_dir / "DOGEUSD.parquet", index=False)

    summarize_forecast_row(
        symbol="DOGEUSD",
        history=history,
        row={
            "timestamp": target_ts,
            "symbol": "DOGEUSD",
            "issued_at": issued_at,
            "target_timestamp": target_ts,
            "horizon_hours": 4,
        },
        backend="heuristic",
        model=None,
        summary_cache_dir=summary_dir,
        context_hours=24 * 7,
        force_rebuild=True,
    )

    cached = pd.read_parquet(summary_dir / "DOGEUSD.parquet")
    assert len(cached) == 1
    assert cached.loc[0, "horizon_hours"] == 4
