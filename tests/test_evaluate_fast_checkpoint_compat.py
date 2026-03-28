from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
def test_load_policy_for_eval_ignores_encoder_norm_keys():
    from pufferlib_market.evaluate_fast import TradingPolicy, _load_policy_for_eval

    obs_size = 21
    num_symbols = 1
    hidden = 8
    policy = TradingPolicy(obs_size, 3, hidden=hidden)
    state_dict = policy.state_dict()
    state_dict["encoder_norm.weight"] = torch.ones(hidden)
    state_dict["encoder_norm.bias"] = torch.zeros(hidden)

    loaded_policy, loaded_state = _load_policy_for_eval(
        payload={"model": state_dict},
        obs_size=obs_size,
        num_symbols=num_symbols,
        arch="mlp",
        hidden_size=hidden,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded_policy, TradingPolicy)
    assert "encoder_norm.weight" in loaded_state
