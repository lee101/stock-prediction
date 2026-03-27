"""Tests for portfolio evaluator and research features."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sharpnessadjustedproximalpolicy.portfolio_eval import (
    combine_portfolio,
    compute_portfolio_metrics,
    get_best_checkpoints,
    warp_checkpoints,
)
from sharpnessadjustedproximalpolicy.config import SAPConfig
from sharpnessadjustedproximalpolicy.trainer import _spectral_reg


class TestCombinePortfolio:
    def test_equal_weight(self):
        pv = {
            "A": np.array([1.0, 1.1, 1.2]),
            "B": np.array([1.0, 0.9, 1.0]),
        }
        combined = combine_portfolio(pv, method="equal")
        assert len(combined) == 3
        np.testing.assert_allclose(combined[0], 1.0)
        # (1.1+0.9)/2 = 1.0, (1.2+1.0)/2 = 1.1
        np.testing.assert_allclose(combined[1], 1.0)
        np.testing.assert_allclose(combined[2], 1.1)

    def test_equal_weight_single(self):
        pv = {"A": np.array([1.0, 1.5, 2.0])}
        combined = combine_portfolio(pv, method="equal")
        np.testing.assert_allclose(combined, [1.0, 1.5, 2.0])

    def test_inverse_vol(self):
        pv = {
            "A": np.linspace(1.0, 1.5, 200),  # smooth uptrend
            "B": np.array([1.0 + 0.3 * np.sin(i / 5) + i * 0.001 for i in range(200)]),  # volatile
        }
        combined = combine_portfolio(pv, method="inverse_vol")
        assert len(combined) == 200
        # Smooth asset should get higher weight
        assert combined[-1] > 1.0

    def test_sqrt_sortino(self):
        pv = {
            "A": np.linspace(1.0, 2.0, 100),  # strong uptrend
            "B": np.linspace(1.0, 0.5, 100),  # downtrend
        }
        combined = combine_portfolio(pv, method="sqrt_sortino")
        assert len(combined) == 100
        # Should weight more toward A
        assert combined[-1] > 1.0

    def test_different_lengths_uses_min(self):
        pv = {
            "A": np.array([1.0, 1.1, 1.2, 1.3]),
            "B": np.array([1.0, 0.9, 1.0]),
        }
        combined = combine_portfolio(pv, method="equal")
        assert len(combined) == 3


class TestPortfolioMetrics:
    def test_flat_pv(self):
        pv = np.ones(100)
        m = compute_portfolio_metrics(pv)
        assert abs(m["total_return_pct"]) < 0.01
        assert abs(m["max_drawdown_pct"]) < 0.01

    def test_uptrend(self):
        pv = np.linspace(1.0, 2.0, 100)
        m = compute_portfolio_metrics(pv)
        assert abs(m["total_return_pct"] - 100.0) < 1.0
        assert m["sortino"] > 0
        assert m["max_drawdown_pct"] >= -0.1  # monotone up = ~0 DD

    def test_crash(self):
        pv = np.concatenate([np.ones(50), np.linspace(1.0, 0.5, 50)])
        m = compute_portfolio_metrics(pv)
        assert m["total_return_pct"] < 0
        assert m["max_drawdown_pct"] < -40


class TestSpectralReg:
    def test_returns_positive_scalar(self):
        model = torch.nn.Linear(10, 5)
        reg = _spectral_reg(model, weight=1.0)
        assert reg.ndim == 0
        assert reg.item() > 0

    def test_weight_scales_linearly(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 5)
        torch.manual_seed(0)
        r1 = _spectral_reg(model, weight=1.0)
        torch.manual_seed(0)
        r2 = _spectral_reg(model, weight=2.0)
        # Same random u -> exactly 2x
        np.testing.assert_allclose(r2.item(), r1.item() * 2.0, rtol=1e-5)

    def test_zero_weight_returns_zero(self):
        model = torch.nn.Linear(10, 5)
        reg = _spectral_reg(model, weight=0.0)
        assert reg.item() == 0.0


class TestSAPConfig:
    def test_new_fields_default(self):
        sc = SAPConfig()
        assert sc.spectral_reg_weight == 0.0
        assert sc.multi_period_windows == []
        assert sc.multi_period_weight == 0.0

    def test_new_fields_set(self):
        sc = SAPConfig(
            spectral_reg_weight=0.01,
            multi_period_windows=[24, 72],
            multi_period_weight=0.3,
        )
        assert sc.spectral_reg_weight == 0.01
        assert sc.multi_period_windows == [24, 72]
        assert sc.multi_period_weight == 0.3


class TestWARP:
    def test_single_checkpoint(self, tmp_path):
        sd = {"w": torch.randn(3, 4)}
        ckpt = {"state_dict": sd, "config": {}, "metrics": {"score": 1.0}}
        p = tmp_path / "ckpt.pt"
        torch.save(ckpt, p)
        result = warp_checkpoints([p], top_k=3)
        assert "state_dict" in result
        torch.testing.assert_close(result["state_dict"]["w"], sd["w"])

    def test_multiple_checkpoints_averaged(self, tmp_path):
        paths = []
        for i in range(3):
            sd = {"w": torch.ones(3, 4) * (i + 1)}
            ckpt = {"state_dict": sd, "config": {}, "metrics": {"score": float(i)}, "feature_columns": []}
            p = tmp_path / f"ckpt_{i}.pt"
            torch.save(ckpt, p)
            paths.append(p)

        result = warp_checkpoints(paths, top_k=2)
        # top-2 by score: i=2 (score=2) and i=1 (score=1), avg = (3+2)/2 = 2.5
        np.testing.assert_allclose(result["state_dict"]["w"].numpy(), 2.5 * np.ones((3, 4)), atol=1e-5)


class TestGetBestCheckpoints:
    def test_parses_csv(self, tmp_path):
        csv_content = """symbol,config,best_epoch,val_sortino,val_return,sharpness,wd_scale,wall_s,error
BTCUSD,periodic_wd01,7,85.917,2.0298,360.1,1.57,3928,
AVAXUSD,baseline_wd01,6,5.014,-0.5422,0.0,1.00,108,
AAVEUSD,periodic_wd01,,,,,,,"Insufficient hourly history"
"""
        p = tmp_path / "lb.csv"
        p.write_text(csv_content)
        best = get_best_checkpoints(p)
        assert "BTCUSD" in best
        assert "AVAXUSD" not in best  # excluded
        assert "AAVEUSD" not in best  # excluded
        assert best["BTCUSD"]["epoch"] == 7
