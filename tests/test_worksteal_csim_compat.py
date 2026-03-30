"""Tests for C simulator compatibility gating."""
from __future__ import annotations

import pytest

from binance_worksteal import csim_sweep, csim_sweep2
from binance_worksteal.csim.compat import (
    assert_csim_compatible_configs,
    config_supports_csim,
    find_first_csim_incompatible_config,
    get_csim_incompatibility_reasons,
    summarize_csim_incompatibility,
)
from binance_worksteal.strategy import WorkStealConfig


def test_default_strategy_config_is_not_csim_compatible():
    config = WorkStealConfig()

    reasons = get_csim_incompatibility_reasons(config)

    assert "risk_off regime switching" in reasons
    assert not config_supports_csim(config)


def test_supported_subset_reports_compatible():
    config = WorkStealConfig(
        sma_filter_period=20,
        sma_check_method="current",
        risk_off_trigger_sma_period=0,
        risk_off_trigger_momentum_period=0,
    )

    assert config_supports_csim(config)
    assert get_csim_incompatibility_reasons(config) == []


def test_summary_truncates_long_reason_list():
    config = WorkStealConfig(
        realistic_fill=True,
        daily_checkpoint_only=True,
        ref_price_method="close",
        market_breadth_filter=0.7,
        risk_off_trigger_sma_period=30,
        risk_off_trigger_momentum_period=7,
    )

    summary = summarize_csim_incompatibility(config, max_items=3)

    assert summary == "realistic_fill, daily_checkpoint_only, ref_price_method=close (+2 more)"


def test_find_first_incompatible_config_reports_index_and_summary():
    configs = [
        WorkStealConfig(
            sma_filter_period=20,
            sma_check_method="current",
            risk_off_trigger_sma_period=0,
            risk_off_trigger_momentum_period=0,
        ),
        WorkStealConfig(realistic_fill=True),
    ]

    result = find_first_csim_incompatible_config(configs)

    assert result is not None
    index, _config, summary = result
    assert index == 1
    assert summary == "realistic_fill, risk_off regime switching"


def test_assert_csim_compatible_configs_raises_with_context():
    configs = [WorkStealConfig(realistic_fill=True)]

    with pytest.raises(ValueError, match="example generated C-sim-incompatible config #0"):
        assert_csim_compatible_configs(configs, context="example")


def test_csim_sweep_generate_configs_are_compatible():
    configs = csim_sweep.generate_configs()

    assert configs
    assert all(config_supports_csim(config) for config in configs)


def test_csim_sweep2_generate_configs_are_compatible():
    configs = csim_sweep2.generate_configs()

    assert configs
    assert all(config_supports_csim(config) for config in configs)
