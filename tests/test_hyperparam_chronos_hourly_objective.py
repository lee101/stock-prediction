from __future__ import annotations

from typing import Dict, List, Tuple

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
