from __future__ import annotations


def test_stocks_daily_hourlycovered_frontier_preset_exists() -> None:
    from pufferlib_market.gpu_pool import PRESETS

    jobs = PRESETS["stocks_daily_hourlycovered_frontier"]
    descriptions = {job["description"] for job in jobs}

    assert len(jobs) == 6
    assert "stocks10h_cos_s42" in descriptions
    assert "stocks10h_cos_s123" in descriptions
    assert all(job["hidden_size"] == 1024 for job in jobs)
    assert all(job["periods_per_year"] == 252.0 for job in jobs)


def test_stocks_daily_hourlycovered_refine_preset_exists() -> None:
    from pufferlib_market.gpu_pool import PRESETS

    jobs = PRESETS["stocks_daily_hourlycovered_refine"]
    descriptions = {job["description"] for job in jobs}

    assert len(jobs) == 9
    assert "stocks10h_refine_tp03_s42" in descriptions
    assert "stocks10h_refine_tp04_s15" in descriptions
    assert "stocks10h_refine_tp03_ent008_s42" in descriptions


def test_stocks_daily_hourlycovered_defensive_preset_exists() -> None:
    from pufferlib_market.gpu_pool import PRESETS

    jobs = PRESETS["stocks_daily_hourlycovered_defensive"]
    descriptions = {job["description"] for job in jobs}

    assert len(jobs) == 9
    assert "stocks10h_def_tp05_s42" in descriptions
    assert "stocks10h_def_tp06_s15" in descriptions
    assert "stocks10h_def_tp05_lr1e4_s42" in descriptions
    assert all(job["hidden_size"] == 1024 for job in jobs)
