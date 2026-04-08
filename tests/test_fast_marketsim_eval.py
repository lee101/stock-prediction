from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pufferlib_market import fast_marketsim_eval as marketsim
from torch import nn


class _FixedPolicy(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        return self._logits.repeat(batch, 1), torch.zeros(batch, dtype=torch.float32)


def test_make_policy_fn_applies_tradable_and_short_masks() -> None:
    policy = _FixedPolicy([0.0, 1.0, 9.0, 8.0, 7.0])
    tradable_mask = torch.tensor([True, False], dtype=torch.bool)
    policy_fn = marketsim.make_policy_fn(
        policy,
        torch.device("cpu"),
        num_symbols=2,
        tradable_mask=tradable_mask,
        disable_shorts=True,
    )

    action = policy_fn(np.zeros(16, dtype=np.float32))

    assert action == 1


def test_cache_signature_changes_with_prod_constraints() -> None:
    base = marketsim._cache_config_signature(
        data_dir="pufferlib_market/data",
        periods=[30, 60],
        fee_rate=0.0,
        fill_buffer_bps=8.0,
        periods_per_year=365.0,
        slippage_bps=3.0,
        trailing_stop_pct=0.003,
        max_hold_bars=6,
        min_notional_usd=12.0,
        max_leverage=1.0,
        tradable_symbols="BTCUSD,ETHUSD",
        disable_shorts=True,
    )
    changed = marketsim._cache_config_signature(
        data_dir="pufferlib_market/data",
        periods=[30, 60],
        fee_rate=0.0,
        fill_buffer_bps=8.0,
        periods_per_year=365.0,
        slippage_bps=3.0,
        trailing_stop_pct=0.003,
        max_hold_bars=6,
        min_notional_usd=12.0,
        max_leverage=0.5,
        tradable_symbols="BTCUSD,ETHUSD",
        disable_shorts=True,
    )

    assert base != changed


def test_is_cached_requires_matching_signature() -> None:
    signature = "sig-a"
    cache = {
        "checkpoint.pt": {
            "mtime": 123.0,
            "config_signature": signature,
            "results": [{"return_pct": 1.23}],
        }
    }

    assert marketsim.is_cached(cache, "checkpoint.pt", 123.0, signature) is True
    assert marketsim.is_cached(cache, "checkpoint.pt", 123.0, "sig-b") is False


def test_fast_eval_all_forwards_prod_constraints_to_sequential(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_seq(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame()

    monkeypatch.setattr(marketsim, "fast_eval_sequential", _fake_seq)

    marketsim.fast_eval_all(
        checkpoint_dirs=[("dir_a", "unused.bin")],
        periods=[30],
        root=".",
        parallel=False,
        max_leverage=0.5,
        tradable_symbols="BTCUSD,ETHUSD",
        disable_shorts=True,
    )

    kwargs = captured["kwargs"]
    assert kwargs["max_leverage"] == 0.5
    assert kwargs["tradable_symbols"] == "BTCUSD,ETHUSD"
    assert kwargs["disable_shorts"] is True


def test_fast_eval_all_forwards_prod_constraints_to_parallel(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_parallel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame()

    monkeypatch.setattr(marketsim, "fast_eval_parallel", _fake_parallel)

    marketsim.fast_eval_all(
        checkpoint_dirs=[("dir_a", "unused.bin")],
        periods=[30],
        root=".",
        parallel=True,
        max_workers=2,
        max_leverage=0.5,
        tradable_symbols="BTCUSD,ETHUSD",
        disable_shorts=True,
    )

    kwargs = captured["kwargs"]
    assert kwargs["max_leverage"] == 0.5
    assert kwargs["tradable_symbols"] == "BTCUSD,ETHUSD"
    assert kwargs["disable_shorts"] is True
