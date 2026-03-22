"""Tests for pufferlib_market.early_stopper."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pufferlib_market.early_stopper import (
    BestKnownTracker,
    PolynomialEarlyStopper,
    combined_score,
)


# ---------------------------------------------------------------------------
# combined_score
# ---------------------------------------------------------------------------


class TestCombinedScore:
    def test_all_present(self):
        # 0.5*1.5 + 0.5*0.2 = 0.85
        result = combined_score(0.2, 1.5, 0.6)
        assert abs(result - 0.85) < 1e-9

    def test_only_sortino(self):
        # val_return=None, val_wr=None → only sortino contributes
        result = combined_score(None, 1.5, None)
        assert abs(result - 1.5) < 1e-9

    def test_only_return(self):
        result = combined_score(0.2, None, None)
        assert abs(result - 0.2) < 1e-9

    def test_return_and_wr_no_sortino(self):
        # val_wr not used in scoring; only return contributes when sortino=None
        result = combined_score(0.4, None, 0.7)
        assert abs(result - 0.4) < 1e-9

    def test_sortino_and_wr_no_return(self):
        # val_wr not used in scoring; only sortino contributes when return=None
        result = combined_score(None, 2.0, 0.9)
        assert abs(result - 2.0) < 1e-9

    def test_all_none(self):
        assert combined_score(None, None, None) is None

    def test_zero_values(self):
        result = combined_score(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_negative_values(self):
        result = combined_score(-1.0, -2.0, 0.5)
        assert abs(result - (-1.5)) < 1e-9


# ---------------------------------------------------------------------------
# PolynomialEarlyStopper
# ---------------------------------------------------------------------------


class TestPolynomialEarlyStopper:
    def test_linear_two_obs(self):
        s = PolynomialEarlyStopper()
        # y = x: at progress=0.25 score=0.25, at 0.5 score=0.5
        s.add_observation(0.25, 0.25)
        s.add_observation(0.50, 0.50)
        proj = s.projected_final()
        assert proj is not None
        assert abs(proj - 1.0) < 1e-6

    def test_quadratic_three_obs(self):
        s = PolynomialEarlyStopper()
        # y = x^2: at 0.25→0.0625, 0.5→0.25, 0.75→0.5625
        s.add_observation(0.25, 0.0625)
        s.add_observation(0.50, 0.25)
        s.add_observation(0.75, 0.5625)
        proj = s.projected_final()
        assert proj is not None
        assert abs(proj - 1.0) < 1e-6

    def test_prune_diverging(self):
        # Bad trajectory: starting negative, barely improving
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, -0.5)
        s.add_observation(0.50, 0.1)
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75)
        assert prune is True
        assert proj is not None

    def test_no_prune_converging(self):
        # Good trajectory: heading toward ~2.1 while best=2.0
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 1.5)
        s.add_observation(0.50, 1.8)
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75)
        assert prune is False
        assert proj is not None

    def test_no_prune_cold_start(self):
        # best_known <= -1e6 → never prune
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, -100.0)
        s.add_observation(0.50, -200.0)
        prune, proj = s.should_prune(best_known=-1e7, tolerance=0.75)
        assert prune is False
        assert proj is None

    def test_no_prune_exactly_minus_1e6(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, -100.0)
        s.add_observation(0.50, -200.0)
        prune, proj = s.should_prune(best_known=-1e6, tolerance=0.75)
        assert prune is False
        assert proj is None

    def test_no_prune_insufficient_obs_default(self):
        # min_obs=2: with 1 observation → no prune
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 0.1)
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75)
        assert prune is False
        assert proj is None

    def test_no_prune_insufficient_obs_custom(self):
        # min_obs=3: with 2 observations → no prune
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 0.1)
        s.add_observation(0.50, 0.5)
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75, min_obs=3)
        assert prune is False
        assert proj is None

    def test_degenerate_same_x_values(self):
        # All observations at the same x → polyfit raises RankWarning or is
        # numerically degenerate; should return (False, None)
        s = PolynomialEarlyStopper()
        s.add_observation(0.5, 1.0)
        s.add_observation(0.5, 2.0)
        s.add_observation(0.5, 3.0)
        # projected_final may or may not raise, but should never crash
        result = s.projected_final()
        # should_prune must also not crash
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75)
        # We only verify it doesn't crash; both prune=False and prune=True
        # are acceptable depending on the polyfit result
        assert isinstance(prune, bool)

    def test_projected_final_none_with_no_obs(self):
        s = PolynomialEarlyStopper()
        assert s.projected_final() is None

    def test_projected_final_none_with_one_obs(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.5, 1.0)
        assert s.projected_final() is None

    def test_degree_capped_at_two(self):
        # With 5 observations degree must be min(4, 2)=2 — just verify no crash
        s = PolynomialEarlyStopper()
        for i in range(1, 6):
            s.add_observation(i * 0.1, float(i))
        proj = s.projected_final()
        assert proj is not None

    def test_prune_borderline_at_tolerance(self):
        # projected = best_known * tolerance exactly → no prune (strict <)
        s = PolynomialEarlyStopper()
        # linear: at 0.25→0.75, at 0.5→0.75 → projected at 1.0 = 0.75
        # With a flat line projected = 0.75 = 2.0 * 0.375 — craft exact case:
        # We want projected == best_known * tolerance = 2.0 * 0.75 = 1.5
        # linear through (0.25,1.5),(0.5,1.5) → flat at 1.5
        s.add_observation(0.25, 1.5)
        s.add_observation(0.50, 1.5)
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75)
        # projected == threshold → should NOT prune (strict <)
        assert prune is False


# ---------------------------------------------------------------------------
# BestKnownTracker
# ---------------------------------------------------------------------------


class TestBestKnownTracker:
    def test_unknown_track_returns_neg_inf(self, tmp_path):
        tracker = BestKnownTracker(tmp_path / "bests.json")
        val = tracker.get_best("stocks_daily")
        assert math.isinf(val) and val < 0

    def test_update_new_best(self, tmp_path):
        path = tmp_path / "bests.json"
        tracker = BestKnownTracker(path)
        updated = tracker.update("stocks_daily", 1.5, description="run1")
        assert updated is True
        assert abs(tracker.get_best("stocks_daily") - 1.5) < 1e-9

    def test_no_update_lower_score(self, tmp_path):
        path = tmp_path / "bests.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 1.5)
        updated = tracker.update("stocks_daily", 1.0)
        assert updated is False
        assert abs(tracker.get_best("stocks_daily") - 1.5) < 1e-9

    def test_no_update_equal_score(self, tmp_path):
        path = tmp_path / "bests.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 1.5)
        updated = tracker.update("stocks_daily", 1.5)
        assert updated is False

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "bests.json"
        tracker = BestKnownTracker(path)
        tracker.update("binance_crypto", 3.2, description="trial42")
        tracker.update("mixed", 0.9)

        reloaded = BestKnownTracker(path)
        assert abs(reloaded.get_best("binance_crypto") - 3.2) < 1e-9
        assert abs(reloaded.get_best("mixed") - 0.9) < 1e-9
        assert reloaded.get_best("hourly_crypto") == -math.inf

    def test_all_bests(self, tmp_path):
        path = tmp_path / "bests.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 1.0)
        tracker.update("binance_crypto", 2.0)
        bests = tracker.all_bests()
        assert set(bests.keys()) == {"stocks_daily", "binance_crypto"}
        assert abs(bests["stocks_daily"] - 1.0) < 1e-9
        assert abs(bests["binance_crypto"] - 2.0) < 1e-9

    def test_atomic_write_creates_file(self, tmp_path):
        path = tmp_path / "bests.json"
        tracker = BestKnownTracker(path)
        tracker.update("mixed", 5.5)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "mixed" in data
        assert abs(data["mixed"]["score"] - 5.5) < 1e-9

    def test_start_empty_if_file_missing(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        tracker = BestKnownTracker(path)
        assert tracker.all_bests() == {}

    def test_corrupt_file_starts_empty(self, tmp_path):
        path = tmp_path / "bests.json"
        path.write_text("not valid json{{")
        tracker = BestKnownTracker(path)
        assert tracker.all_bests() == {}

    def test_multiple_tracks_independent(self, tmp_path):
        path = tmp_path / "bests.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 1.0)
        tracker.update("binance_crypto", 2.0)
        tracker.update("hourly_crypto", 0.5)
        # Updating one track must not affect others
        tracker.update("stocks_daily", 1.5)
        assert abs(tracker.get_best("binance_crypto") - 2.0) < 1e-9
        assert abs(tracker.get_best("hourly_crypto") - 0.5) < 1e-9
        assert abs(tracker.get_best("stocks_daily") - 1.5) < 1e-9
