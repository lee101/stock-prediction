from scripts.render_prod_stocks_video import _find_best_window_start


def test_find_best_window_start_prefers_higher_return_then_lower_drawdown():
    scores = {
        0: (0.10, 1.0, 0.08, 4),
        1: (0.25, 0.5, 0.20, 3),
        2: (0.25, 2.0, 0.05, 6),
        3: (0.02, 3.0, 0.01, 2),
    }

    result = _find_best_window_start(
        num_timesteps=8,
        window_steps=4,
        evaluate_start=lambda start: scores[start],
    )

    assert result == {
        "total_return": 0.25,
        "sortino": 2.0,
        "max_drawdown": 0.05,
        "num_trades": 6,
        "window_start": 2,
    }


def test_find_best_window_start_rejects_too_large_window():
    try:
        _find_best_window_start(
            num_timesteps=5,
            window_steps=5,
            evaluate_start=lambda start: (0.0, 0.0, 0.0, 0),
        )
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("expected ValueError")
