from src.portfolio_filters import (
    DropRecord,
    filter_positive_forecasts,
    get_selected_strategy_forecast,
)


def test_filter_positive_forecasts_drops_negative_predictions():
    picks = {
        "BTCUSD": {
            "strategy": "simple",
            "avg_return": 0.01,
            "strategy_candidate_forecasted_pnl": {"simple": 0.02},
        },
        "ETHUSD": {
            "strategy": "simple",
            "avg_return": 0.01,
            "strategy_candidate_forecasted_pnl": {"simple": -0.01},
        },
    }

    filtered, dropped = filter_positive_forecasts(picks)

    assert list(filtered) == ["BTCUSD"]
    assert list(dropped) == ["ETHUSD"]
    assert isinstance(dropped["ETHUSD"], DropRecord)
    assert dropped["ETHUSD"].forecast == -0.01


def test_filter_positive_forecasts_uses_avg_return_when_forecast_missing():
    picks = {
        "SOLUSD": {
            "strategy": "simple",
            "avg_return": 0.004,
        },
        "LINKUSD": {
            "strategy": "simple",
            "avg_return": -0.002,
        },
    }

    filtered, dropped = filter_positive_forecasts(picks, require_positive_forecast=False)

    assert list(filtered) == ["SOLUSD"]
    assert "LINKUSD" in dropped
    assert dropped["LINKUSD"].avg_return == -0.002


def test_filter_positive_forecasts_accepts_when_guards_disabled():
    picks = {
        "UNIUSD": {
            "strategy": "maxdiff",
            "avg_return": -0.001,
            "strategy_candidate_forecasted_pnl": {"maxdiff": -0.003},
        },
    }

    filtered, dropped = filter_positive_forecasts(
        picks,
        require_positive_forecast=False,
        require_positive_avg_return=False,
    )

    assert filtered == picks
    assert dropped == {}


def test_get_selected_strategy_forecast_checks_candidate_map_first():
    entry = {
        "strategy": "maxdiff",
        "strategy_candidate_forecasted_pnl": {"maxdiff": 0.012},
        "maxdiff_forecasted_pnl": -0.5,
        "avg_return": -1.0,
    }
    assert abs(get_selected_strategy_forecast(entry) - 0.012) < 1e-12
