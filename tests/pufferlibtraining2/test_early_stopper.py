"""Tests for pufferlib_market.early_stopper module."""
import json
import math
import tempfile
from pathlib import Path

import pytest

from pufferlib_market.early_stopper import (
    BestKnownTracker,
    PolynomialEarlyStopper,
    combined_score,
)


class TestCombinedScore:
    def test_both_metrics(self):
        score = combined_score(0.5, 1.0)
        assert score == pytest.approx(0.75)  # (0.5*0.5 + 1.0*0.5) / 1.0

    def test_only_sortino(self):
        score = combined_score(None, 2.0)
        assert score == pytest.approx(2.0)

    def test_only_return(self):
        score = combined_score(0.3, None)
        assert score == pytest.approx(0.3)

    def test_no_metrics(self):
        assert combined_score(None, None) is None

    def test_with_wr(self):
        score = combined_score(0.5, 1.0, val_wr=0.6)
        # wr is accepted but not included in weighting (only return+sortino)
        assert score == pytest.approx(0.75)


class TestPolynomialEarlyStopper:
    def test_prune_declining(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, -0.5)
        s.add_observation(0.50, 0.1)
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75)
        assert prune is True
        assert proj == pytest.approx(1.3)  # linear extrapolation: y = -0.5 + 2.4*x → y(1.0) = 1.3

    def test_no_prune_strong_growth(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 1.5)
        s.add_observation(0.50, 1.8)
        prune, proj = s.should_prune(best_known=2.0, tolerance=0.75)
        assert prune is False

    def test_single_obs_no_prune(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 0.1)
        prune, proj = s.should_prune(best_known=2.0)
        assert prune is False
        assert proj is None

    def test_no_obs_no_prune(self):
        s = PolynomialEarlyStopper()
        prune, proj = s.should_prune(best_known=2.0)
        assert prune is False
        assert proj is None

    def test_best_known_neg_inf_no_prune(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, -10.0)
        s.add_observation(0.50, -5.0)
        prune, proj = s.should_prune(best_known=-math.inf)
        assert prune is False

    def test_three_obs_quadratic(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 0.5)
        s.add_observation(0.50, 1.0)
        s.add_observation(0.75, 1.5)
        proj = s.projected_final()
        assert proj is not None

    def test_projected_final_none_with_one_obs(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 1.0)
        assert s.projected_final() is None

    def test_tolerance_boundary(self):
        s = PolynomialEarlyStopper()
        s.add_observation(0.25, 1.0)
        s.add_observation(0.50, 1.5)
        # proj ~= 2.0 (linear), best_known=2.0, tolerance=1.0 → threshold=2.0
        # 2.0 < 2.0 is False, so no prune
        prune, proj = s.should_prune(best_known=2.0, tolerance=1.0)
        assert prune is False


class TestBestKnownTracker:
    def test_update_and_get(self, tmp_path):
        path = tmp_path / "best.json"
        tracker = BestKnownTracker(path)
        assert tracker.get_best("stocks_daily") == -math.inf
        updated = tracker.update("stocks_daily", 1.5, "trial_a")
        assert updated is True
        assert tracker.get_best("stocks_daily") == pytest.approx(1.5)

    def test_no_downgrade(self, tmp_path):
        path = tmp_path / "best.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 2.0, "good")
        updated = tracker.update("stocks_daily", 1.0, "worse")
        assert updated is False
        assert tracker.get_best("stocks_daily") == pytest.approx(2.0)

    def test_persistence(self, tmp_path):
        path = tmp_path / "best.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 3.0, "best_trial")
        # Reload from disk
        tracker2 = BestKnownTracker(path)
        assert tracker2.get_best("stocks_daily") == pytest.approx(3.0)

    def test_multiple_tracks(self, tmp_path):
        path = tmp_path / "best.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 1.0, "s")
        tracker.update("hourly_crypto", 2.0, "c")
        assert tracker.get_best("stocks_daily") == pytest.approx(1.0)
        assert tracker.get_best("hourly_crypto") == pytest.approx(2.0)
        assert tracker.get_best("mixed") == -math.inf

    def test_all_bests(self, tmp_path):
        path = tmp_path / "best.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 1.5)
        tracker.update("hourly_crypto", 2.5)
        bests = tracker.all_bests()
        assert bests["stocks_daily"] == pytest.approx(1.5)
        assert bests["hourly_crypto"] == pytest.approx(2.5)

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "best.json"
        path.write_text("not valid json")
        tracker = BestKnownTracker(path)
        assert tracker.get_best("any") == -math.inf

    def test_atomic_write(self, tmp_path):
        path = tmp_path / "best.json"
        tracker = BestKnownTracker(path)
        tracker.update("stocks_daily", 5.0)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()
