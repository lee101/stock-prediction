"""Tests for differentiable_loss_utils optimized loss functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest
from differentiable_loss_utils import (
    compute_hourly_objective,
    compute_multiwindow_objective,
    compute_loss_by_type,
    combined_sortino_pnl_loss,
    compute_sharpe_objective,
    compute_calmar_objective,
    compute_log_wealth_objective,
    compute_sortino_dd_objective,
    _sortino_core,
    _has_smoothness,
    _apply_smoothness,
    _as_tensor,
    HOURLY_PERIODS_PER_YEAR,
)


class TestSortinoCore:
    def test_basic(self):
        r = torch.tensor([[0.01, -0.005, 0.02, -0.01, 0.015]])
        periods = torch.tensor([HOURLY_PERIODS_PER_YEAR], dtype=torch.float32)
        score, sortino, annual = _sortino_core(r, periods, 0.05)
        assert score.shape == (1,)
        assert sortino.shape == (1,)
        assert annual.shape == (1,)
        assert sortino.item() > 0

    def test_matches_compute_hourly(self):
        r = torch.randn(3, 48)
        s1, sort1, ann1 = compute_hourly_objective(r, return_weight=0.05)
        periods = _as_tensor(HOURLY_PERIODS_PER_YEAR, r)
        s2, sort2, ann2 = _sortino_core(r, periods, 0.05)
        assert torch.allclose(s1, s2, atol=1e-5)
        assert torch.allclose(sort1, sort2, atol=1e-5)
        assert torch.allclose(ann1, ann2, atol=1e-5)


class TestComputeHourlyObjective:
    def test_positive_returns_positive_sortino(self):
        r = torch.tensor([[0.01, 0.02, 0.01, 0.015, 0.01]])
        score, sortino, annual = compute_hourly_objective(r)
        assert sortino.item() > 0
        assert annual.item() > 0

    def test_negative_returns_negative_sortino(self):
        r = torch.tensor([[-0.01, -0.02, -0.01, -0.015, -0.01]])
        score, sortino, annual = compute_hourly_objective(r)
        assert sortino.item() < 0

    def test_0d_raises(self):
        with pytest.raises(ValueError):
            compute_hourly_objective(torch.tensor(0.5))

    def test_smoothness_penalty_float(self):
        r = torch.randn(2, 24)
        s0, _, _ = compute_hourly_objective(r, smoothness_penalty=0.0)
        s1, _, _ = compute_hourly_objective(r, smoothness_penalty=1.0)
        assert not torch.allclose(s0, s1)

    def test_smoothness_penalty_tensor(self):
        r = torch.randn(2, 24)
        s0, _, _ = compute_hourly_objective(r, smoothness_penalty=0.0)
        pen = torch.tensor([1.0, 0.0])
        s1, _, _ = compute_hourly_objective(r, smoothness_penalty=pen)
        assert not torch.allclose(s0[0], s1[0])
        assert torch.allclose(s0[1], s1[1], atol=1e-6)

    def test_backward(self):
        r = torch.randn(4, 48, requires_grad=True)
        s, sort, ann = compute_hourly_objective(r)
        s.sum().backward()
        assert r.grad is not None
        assert not torch.all(r.grad == 0)

    def test_tensor_periods(self):
        r = torch.randn(2, 10)
        periods = torch.tensor([8760.0, 365.0])
        s, sort, ann = compute_hourly_objective(r, periods_per_year=periods)
        assert s.shape == (2,)


class TestComputeMultiwindowObjective:
    def test_basic(self):
        r = torch.randn(2, 48)
        score, sortino, annual = compute_multiwindow_objective(r)
        assert score.shape == (2,)
        assert sortino.shape == (2,)

    def test_minimax_le_mean(self):
        r = torch.randn(3, 48)
        s_min, _, _ = compute_multiwindow_objective(r, aggregation="minimax")
        s_mean, _, _ = compute_multiwindow_objective(r, aggregation="mean")
        assert (s_min <= s_mean + 1e-5).all()

    def test_dd_penalty(self):
        r = torch.randn(2, 48)
        s0, _, _ = compute_multiwindow_objective(r, dd_penalty=0.0)
        s1, _, _ = compute_multiwindow_objective(r, dd_penalty=1.0)
        assert not torch.allclose(s0, s1)

    def test_backward(self):
        r = torch.randn(2, 48, requires_grad=True)
        score, _, _ = compute_multiwindow_objective(r)
        score.sum().backward()
        assert r.grad is not None

    def test_dd_backward(self):
        r = torch.randn(2, 48, requires_grad=True)
        score, _, _ = compute_multiwindow_objective(r, dd_penalty=0.5)
        score.sum().backward()
        assert r.grad is not None

    def test_single_frac(self):
        r = torch.randn(2, 48)
        score, sortino, annual = compute_multiwindow_objective(r, window_fractions=(1.0,))
        s2, sort2, ann2 = compute_hourly_objective(r)
        assert torch.allclose(sortino, sort2, atol=1e-5)

    def test_softmin(self):
        r = torch.randn(2, 48)
        s, _, _ = compute_multiwindow_objective(r, aggregation="softmin")
        assert s.shape == (2,)

    def test_dd_penalty_consistency(self):
        """DD penalty from shared cumsum should match independent cumsum."""
        torch.manual_seed(42)
        r = torch.randn(2, 48)
        fracs = [0.5, 1.0]
        score_opt, _, _ = compute_multiwindow_objective(r, dd_penalty=1.0, window_fractions=fracs)
        T = r.shape[-1]
        scores_ref = []
        for frac in sorted(fracs):
            w = max(int(T * frac), 2)
            sub = r[..., -w:]
            sc, _, _ = compute_hourly_objective(sub)
            cum = torch.cumsum(sub, dim=-1)
            rmax = torch.cummax(cum, dim=-1).values
            max_dd = (rmax - cum).max(dim=-1).values
            sc = sc - 1.0 * max_dd
            scores_ref.append(sc)
        stacked = torch.stack(scores_ref, dim=0)
        score_ref = stacked.min(dim=0).values
        assert torch.allclose(score_opt, score_ref, atol=1e-5), f"opt={score_opt} ref={score_ref}"


class TestComputeLossByType:
    def test_sortino(self):
        r = torch.randn(2, 24)
        loss, score, ratio, annual = compute_loss_by_type(r, "sortino")
        assert loss.ndim == 0

    def test_sharpe(self):
        r = torch.randn(2, 24)
        loss, score, ratio, annual = compute_loss_by_type(r, "sharpe")
        assert loss.ndim == 0

    def test_calmar(self):
        r = torch.randn(2, 24)
        loss, score, ratio, annual = compute_loss_by_type(r, "calmar")
        assert loss.ndim == 0

    def test_log_wealth(self):
        r = torch.randn(2, 24)
        loss, score, ratio, annual = compute_loss_by_type(r, "log_wealth")
        assert loss.ndim == 0

    def test_sortino_dd(self):
        r = torch.randn(2, 24)
        loss, score, ratio, annual = compute_loss_by_type(r, "sortino_dd")
        assert loss.ndim == 0

    def test_multiwindow(self):
        r = torch.randn(2, 24)
        loss, score, ratio, annual = compute_loss_by_type(r, "multiwindow")
        assert loss.ndim == 0

    def test_multiwindow_dd(self):
        r = torch.randn(2, 24)
        loss, score, ratio, annual = compute_loss_by_type(r, "multiwindow_dd")
        assert loss.ndim == 0

    def test_smoothness_zero_no_overhead(self):
        r = torch.randn(2, 24)
        loss0, _, _, _ = compute_loss_by_type(r, "sortino", smoothness_penalty=0.0)
        loss1, _, _, _ = compute_loss_by_type(r, "sortino", smoothness_penalty=0.0)
        assert torch.allclose(loss0, loss1)

    def test_smoothness_nonzero(self):
        r = torch.randn(2, 24)
        loss0, _, _, _ = compute_loss_by_type(r, "sortino", smoothness_penalty=0.0)
        loss1, _, _, _ = compute_loss_by_type(r, "sortino", smoothness_penalty=1.0)
        assert not torch.allclose(loss0, loss1)

    def test_backward_all_types(self):
        for lt in ["sortino", "sharpe", "calmar", "log_wealth", "sortino_dd", "multiwindow", "multiwindow_dd"]:
            r = torch.randn(2, 24, requires_grad=True)
            loss, _, _, _ = compute_loss_by_type(r, lt)
            loss.backward()
            assert r.grad is not None, f"No grad for {lt}"


class TestCombinedSortinoPnlLoss:
    def test_basic(self):
        r = torch.randn(2, 24)
        loss = combined_sortino_pnl_loss(r)
        assert loss.ndim == 0

    def test_smoothness_zero(self):
        r = torch.randn(2, 24)
        l0 = combined_sortino_pnl_loss(r, smoothness_penalty=0.0)
        l1 = combined_sortino_pnl_loss(r, smoothness_penalty=0.0)
        assert torch.allclose(l0, l1)

    def test_smoothness_nonzero(self):
        r = torch.randn(2, 24)
        l0 = combined_sortino_pnl_loss(r, smoothness_penalty=0.0)
        l1 = combined_sortino_pnl_loss(r, smoothness_penalty=1.0)
        assert not torch.allclose(l0, l1)

    def test_backward(self):
        r = torch.randn(2, 24, requires_grad=True)
        loss = combined_sortino_pnl_loss(r)
        loss.backward()
        assert r.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
