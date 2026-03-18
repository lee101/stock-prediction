from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Tuple

import pandas as pd

from hyperparam_chronos_hourly import Chronos2HourlyTuner


class _StubTuner(Chronos2HourlyTuner):
    def __init__(self) -> None:
        super().__init__(
            holdout_hours=24,
            prediction_length=6,
            objective="composite",
            cohort_size=1,
        )

    def _build_cohort_map(self, symbols: List[str]) -> Dict[str, Tuple[str, ...]]:  # type: ignore[override]
        return {
            "AAA": ("BBB",),
            "BBB": ("AAA",),
        }

    def evaluate_params(  # type: ignore[override]
        self,
        symbol: str,
        context_length: int,
        skip_rates,
        aggregation_method: str,
        use_multivariate: bool,
        scaler: str,
        prediction_length=None,
    ):
        del skip_rates, aggregation_method, use_multivariate, scaler, prediction_length
        # Context 20: lower MAE but worse smoothness/objective.
        if context_length == 20:
            base = {
                "status": "success",
                "price_mae": 1.0,
                "price_rmse": 1.2,
                "pct_return_mae": 0.02,
                "pct_return_mae_smoothness": 0.06,
                "direction_accuracy": 0.55,
                "objective": 0.04,
                "latency_s": 0.1,
                "n_samples": 6,
            }
        else:
            # Context 40: slightly worse MAE, much better objective.
            base = {
                "status": "success",
                "price_mae": 1.1,
                "price_rmse": 1.3,
                "pct_return_mae": 0.022,
                "pct_return_mae_smoothness": 0.015,
                "direction_accuracy": 0.62,
                "objective": 0.018,
                "latency_s": 0.2,
                "n_samples": 6,
            }
        # Peer symbol gets the same relative ordering.
        if symbol == "BBB":
            base = dict(base)
            base["latency_s"] = 0.15
        return base


def test_grid_search_uses_composite_selection_metric() -> None:
    tuner = _StubTuner()
    grid = {
        "context_length": [20, 40],
        "skip_rates": [(1,)],
        "aggregation_method": ["single"],
        "use_multivariate": [False],
        "scaler": ["none"],
    }
    result = tuner.grid_search(["AAA"], grid)

    assert result["selection_metric"] == "objective"
    best = result["best_per_symbol"]["AAA"]
    assert best["context_length"] == 40
    assert best["cohort_used"] == 1
    assert best["cohort_requested"] == 1


def test_build_cohort_map_floors_series_timestamps() -> None:
    class _RealCohortTuner(Chronos2HourlyTuner):
        def __init__(self) -> None:
            super().__init__(holdout_hours=24, prediction_length=1, objective="composite", cohort_size=1)

        def load_data(self, symbol: str):  # type: ignore[override]
            minutes = "05" if symbol == "AAA" else "55"
            base = 100.0 if symbol == "AAA" else 200.0
            timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="h")
            df = pd.DataFrame(
                {
                    "timestamp": [ts.strftime(f"%Y-%m-%dT%H:{minutes}:00Z") for ts in timestamps],
                    "close": [base + float(i) for i in range(len(timestamps))],
                }
            )
            split_idx = len(df) - self.holdout_hours
            return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    tuner = _RealCohortTuner()

    cohort_map = tuner._build_cohort_map(["AAA", "BBB"])

    assert cohort_map["AAA"] == ("BBB",)
    assert cohort_map["BBB"] == ("AAA",)


def test_predict_single_scale_uses_true_wrapper_multivariate_path() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.predict_df_calls = 0

        def predict_df(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            self.predict_df_calls += 1
            raise AssertionError("predict_df should not be used for multivariate hourly tuning")

    class _FakeWrapper:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def predict_ohlc_multivariate(self, context_df, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(
                {
                    "rows": len(context_df),
                    **kwargs,
                }
            )
            last_ts = pd.to_datetime(context_df["timestamp"].iloc[-1], utc=True)
            future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=3, freq="h", tz="UTC")
            frame = pd.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [2.0, 3.0, 4.0],
                    "low": [0.5, 1.5, 2.5],
                    "close": [1.5, 2.5, 3.5],
                },
                index=future_index,
            )
            return SimpleNamespace(quantile_frames={0.1: frame, 0.5: frame, 0.9: frame})

    class _WrapperBackedTuner(Chronos2HourlyTuner):
        def __init__(self) -> None:
            super().__init__(holdout_hours=24, prediction_length=3, objective="composite")
            self.fake_model = _FakeModel()
            self.fake_wrapper = _FakeWrapper()

        def _load_model(self) -> None:  # type: ignore[override]
            self.model = self.fake_model
            self.wrapper = self.fake_wrapper

    tuner = _WrapperBackedTuner()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="h"),
            "open": [float(i) for i in range(12)],
            "high": [float(i) + 1.0 for i in range(12)],
            "low": [float(i) - 1.0 for i in range(12)],
            "close": [float(i) + 0.5 for i in range(12)],
        }
    )

    quantiles = tuner._predict_single_scale(
        frame,
        "AAA",
        prediction_length=3,
        context_length=8,
        use_multivariate=True,
        scaler="none",
    )

    assert tuner.fake_model.predict_df_calls == 0
    assert len(tuner.fake_wrapper.calls) == 1
    assert tuner.fake_wrapper.calls[0]["symbol"] == "AAA"
    assert tuner.fake_wrapper.calls[0]["context_length"] == 8
    assert set(quantiles) == {0.1, 0.5, 0.9}
    assert list(quantiles[0.5].columns) == ["open", "high", "low", "close"]
    assert quantiles[0.5].index[0] == pd.Timestamp("2026-01-01T12:00:00Z")
