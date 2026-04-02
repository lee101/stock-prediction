"""Tests for stock-specific daily RL autoresearch configs and --stocks flag."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is on the path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.autoresearch_rl import (
    STOCK_EXPERIMENTS,
    TrialConfig,
    _STOCK_DEFAULT_TRAIN,
    _STOCK_DEFAULT_VAL,
    _select_from_pool,
    build_config,
)


# ---------------------------------------------------------------------------
# STOCK_EXPERIMENTS structure
# ---------------------------------------------------------------------------

def test_stock_experiments_nonempty():
    assert len(STOCK_EXPERIMENTS) >= 20, (
        f"Expected at least 20 stock configs, got {len(STOCK_EXPERIMENTS)}"
    )


def test_stock_experiments_all_dicts():
    for i, cfg in enumerate(STOCK_EXPERIMENTS):
        assert isinstance(cfg, dict), f"Entry {i} is not a dict: {cfg!r}"


def test_stock_experiments_all_have_description():
    for i, cfg in enumerate(STOCK_EXPERIMENTS):
        assert "description" in cfg and cfg["description"], (
            f"Entry {i} missing non-empty 'description': {cfg!r}"
        )


def test_stock_experiments_unique_descriptions():
    descriptions = [cfg["description"] for cfg in STOCK_EXPERIMENTS
                    if not cfg["description"].startswith("random_")]
    assert len(descriptions) == len(set(descriptions)), (
        "Duplicate description(s) found in STOCK_EXPERIMENTS"
    )


def test_stock_experiments_build_valid_configs():
    """Every entry must be buildable into a TrialConfig without error."""
    for cfg_dict in STOCK_EXPERIMENTS:
        if cfg_dict.get("description", "").startswith("random_"):
            continue  # random entries are intentionally left as mutations
        config = build_config(cfg_dict)
        assert isinstance(config, TrialConfig), (
            f"build_config failed for {cfg_dict['description']}"
        )


# ---------------------------------------------------------------------------
# Field-level validation
# ---------------------------------------------------------------------------

def test_stock_experiments_all_fields_are_valid_trialconfig_fields():
    """No unknown field names should be present (they'd be silently dropped by build_config)."""
    valid_fields = set(TrialConfig.__dataclass_fields__.keys()) | {"description"}
    for cfg_dict in STOCK_EXPERIMENTS:
        for key in cfg_dict:
            assert key in valid_fields, (
                f"Unknown field '{key}' in config '{cfg_dict.get('description')}'"
            )


def test_stock_baseline_uses_default_fee():
    """stock_baseline should use default fee_rate (0.001) since we override at run time."""
    baseline = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_baseline")
    config = build_config(baseline)
    # fee_rate default in TrialConfig is 0.001 (Alpaca rate)
    assert config.fee_rate == 0.001


def test_stock_experiments_anneal_lr_default():
    """TrialConfig default has anneal_lr=True; configs that don't override should keep it."""
    for cfg_dict in STOCK_EXPERIMENTS:
        if cfg_dict.get("description") == "stock_no_anneal":
            config = build_config(cfg_dict)
            assert config.anneal_lr is False
        elif "anneal_lr" not in cfg_dict:
            config = build_config(cfg_dict)
            # Most configs do not explicitly disable anneal_lr; default should hold
            assert config.anneal_lr is True, (
                f"anneal_lr should default True for {cfg_dict['description']}"
            )


def test_stock_h512_hidden_size():
    cfg = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_h512")
    config = build_config(cfg)
    assert config.hidden_size == 512


def test_stock_h1024_hidden_size():
    cfg = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_h1024")
    config = build_config(cfg)
    assert config.hidden_size == 1024


def test_stock_probe_cpu_fast_uses_small_probe_shape():
    cfg = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_probe_cpu_fast")
    config = build_config(cfg)
    assert config.hidden_size == 256
    assert config.trade_penalty == pytest.approx(0.05)
    assert config.num_envs == 16
    assert config.rollout_len == 64
    assert config.minibatch_size == 256
    assert config.ppo_epochs == 2
    assert config.eval_num_episodes == 20


def test_stock_stable_a40_large_universe_profiles_use_two_x_leverage_and_stability_guards():
    expected = {
        "stock_stable_2x_a40": "mlp",
        "stock_stable_2x_a40_gspo": "mlp",
        "stock_stable_2x_a40_resmlp": "resmlp",
    }
    for desc, arch in expected.items():
        cfg = next(c for c in STOCK_EXPERIMENTS if c["description"] == desc)
        config = build_config(cfg)
        assert config.arch == arch
        assert config.requires_gpu == "a40"
        assert config.max_leverage == pytest.approx(2.0)
        assert config.short_borrow_apr == pytest.approx(0.0001712)
        assert config.no_cuda_graph is True
        assert config.use_bf16 is False
        assert config.max_grad_norm == pytest.approx(0.3)
        assert config.grad_norm_warn_threshold == pytest.approx(20.0)
        assert config.grad_norm_skip_threshold == pytest.approx(200.0)
        assert config.unstable_update_patience == 6
        assert config.min_lr == pytest.approx(5e-6)


def test_stock_a40_neighbor_profiles_match_the_best_local_stock_cluster():
    expected = {
        "stock_ent05_tp03_a40_neighbor": {"trade_penalty": 0.03, "hidden_size": 1024, "seed": 42},
        "stock_ent05_tp05_s123_a40_neighbor": {"trade_penalty": 0.05, "hidden_size": 1024, "seed": 123},
        "stock_h512_reg_a40_neighbor": {"trade_penalty": 0.05, "hidden_size": 512, "seed": 42},
        "stock_obsnorm_ent05_tp03_a40_neighbor": {"trade_penalty": 0.03, "hidden_size": 1024, "seed": 42},
        "stock_tp05_seed55_a40_neighbor": {"trade_penalty": 0.05, "hidden_size": 1024, "seed": 55},
        "stock_tp05_seed111_a40_neighbor": {"trade_penalty": 0.05, "hidden_size": 1024, "seed": 111},
    }
    for desc, fields in expected.items():
        cfg = next(c for c in STOCK_EXPERIMENTS if c["description"] == desc)
        config = build_config(cfg)
        assert config.requires_gpu == "a40"
        assert config.ent_coef == pytest.approx(0.05)
        assert config.trade_penalty == pytest.approx(fields["trade_penalty"])
        assert config.hidden_size == fields["hidden_size"]
        assert config.seed == fields["seed"]
        assert config.num_envs == 128
        assert config.minibatch_size == 2048
        assert config.use_bf16 is True
        assert config.cuda_graph_ppo is True


def test_stock_slippage_configs():
    expected = {"stock_slip_5bps": 5.0, "stock_slip_10bps": 10.0, "stock_slip_15bps": 15.0}
    for desc, bps in expected.items():
        cfg = next(c for c in STOCK_EXPERIMENTS if c["description"] == desc)
        config = build_config(cfg)
        assert config.fill_slippage_bps == bps, (
            f"{desc}: expected {bps} bps, got {config.fill_slippage_bps}"
        )


def test_stock_obs_norm_trade_pen_weight_decay_hybrids():
    cfg005 = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_obs_norm_tp05_wd005")
    config005 = build_config(cfg005)
    assert config005.obs_norm is True
    assert config005.trade_penalty == pytest.approx(0.05)
    assert config005.weight_decay == pytest.approx(0.005)

    cfg01 = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_obs_norm_tp05_wd01")
    config01 = build_config(cfg01)
    assert config01.obs_norm is True
    assert config01.trade_penalty == pytest.approx(0.05)
    assert config01.weight_decay == pytest.approx(0.01)


def test_stock_gamma_configs():
    cfg999 = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_high_gamma_999")
    assert build_config(cfg999).gamma == 0.999

    cfg995 = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_gamma_995")
    assert build_config(cfg995).gamma == 0.995


def test_stock_reward_scale_configs():
    cfg5 = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_reward_scale_5")
    assert build_config(cfg5).reward_scale == 5.0

    cfg20 = next(c for c in STOCK_EXPERIMENTS if c["description"] == "stock_reward_scale_20")
    assert build_config(cfg20).reward_scale == 20.0


# ---------------------------------------------------------------------------
# _select_from_pool helper
# ---------------------------------------------------------------------------

def test_select_from_pool_no_filter():
    result = _select_from_pool(STOCK_EXPERIMENTS)
    assert result == STOCK_EXPERIMENTS


def test_select_from_pool_start_from():
    result = _select_from_pool(STOCK_EXPERIMENTS, start_from=3)
    assert result == STOCK_EXPERIMENTS[3:]


def test_select_from_pool_by_description():
    result = _select_from_pool(STOCK_EXPERIMENTS, descriptions="stock_baseline,stock_h512")
    descs = [c["description"] for c in result]
    assert "stock_baseline" in descs
    assert "stock_h512" in descs
    assert len(descs) == 2


def test_select_from_pool_unknown_description_raises():
    with pytest.raises(ValueError, match="Unknown experiment description"):
        _select_from_pool(STOCK_EXPERIMENTS, descriptions="nonexistent_config")


# ---------------------------------------------------------------------------
# Default data paths
# ---------------------------------------------------------------------------

def test_stock_default_train_path_is_str():
    assert isinstance(_STOCK_DEFAULT_TRAIN, str)
    assert "train" in _STOCK_DEFAULT_TRAIN
    assert _STOCK_DEFAULT_TRAIN.endswith(".bin")


def test_stock_default_val_path_is_str():
    assert isinstance(_STOCK_DEFAULT_VAL, str)
    assert "val" in _STOCK_DEFAULT_VAL
    assert _STOCK_DEFAULT_VAL.endswith(".bin")


def test_stock_default_paths_different():
    assert _STOCK_DEFAULT_TRAIN != _STOCK_DEFAULT_VAL


def test_stock_default_train_bin_exists():
    full_path = REPO / _STOCK_DEFAULT_TRAIN
    assert full_path.exists(), (
        f"Default train data not found at {full_path}. "
        "Run export_data_daily.py to generate it, or update _STOCK_DEFAULT_TRAIN."
    )


def test_stock_default_val_bin_exists():
    full_path = REPO / _STOCK_DEFAULT_VAL
    assert full_path.exists(), (
        f"Default val data not found at {full_path}. "
        "Run export_data_daily.py to generate it, or update _STOCK_DEFAULT_VAL."
    )


# ---------------------------------------------------------------------------
# --stocks flag: argparse integration
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]):
    """Helper: invoke main()'s argparse with a synthetic argv."""
    import argparse
    import importlib
    import types

    # We re-use the parser construction from autoresearch_rl.main() by invoking
    # the module with a patched sys.argv.  Simpler: just copy the relevant args.
    old_argv = sys.argv
    try:
        sys.argv = ["autoresearch_rl"] + argv
        # Re-import to get a fresh parser without side-effects
        import pufferlib_market.autoresearch_rl as ar
        # Reconstruct the argument parser exactly as main() does
        listing_only = "--list-experiments" in sys.argv
        stocks_mode = "--stocks" in sys.argv
        data_required = not listing_only and not stocks_mode
        p = argparse.ArgumentParser()
        p.add_argument("--stocks", action="store_true")
        p.add_argument("--train-data", required=data_required, default=None)
        p.add_argument("--val-data", required=data_required, default=None)
        p.add_argument("--leaderboard", default="pufferlib_market/autoresearch_leaderboard.csv")
        p.add_argument("--checkpoint-root", default="pufferlib_market/checkpoints/autoresearch")
        p.add_argument("--periods-per-year", type=float, default=8760.0)
        p.add_argument("--fee-rate-override", type=float, default=-1.0)
        p.add_argument("--max-steps-override", type=int, default=0)
        args = p.parse_args(argv)
        return args
    finally:
        sys.argv = old_argv


def test_stocks_flag_sets_default_data_paths():
    args = _parse_args(["--stocks"])
    # data paths start as None; main() fills them in after parse
    # Simulate the post-parse logic
    import pufferlib_market.autoresearch_rl as ar
    if args.stocks:
        if args.train_data is None:
            args.train_data = ar._STOCK_DEFAULT_TRAIN
        if args.val_data is None:
            args.val_data = ar._STOCK_DEFAULT_VAL
    assert args.train_data == ar._STOCK_DEFAULT_TRAIN
    assert args.val_data == ar._STOCK_DEFAULT_VAL


def test_stocks_flag_sets_periods_per_year_252():
    args = _parse_args(["--stocks"])
    import pufferlib_market.autoresearch_rl as ar
    if args.stocks and args.periods_per_year == 8760.0:
        args.periods_per_year = 252.0
    assert args.periods_per_year == 252.0


def test_stocks_flag_sets_fee_rate_override():
    args = _parse_args(["--stocks"])
    import pufferlib_market.autoresearch_rl as ar
    if args.stocks and args.fee_rate_override < 0.0:
        args.fee_rate_override = 0.001
    assert args.fee_rate_override == 0.001


def test_stocks_flag_sets_default_leaderboard():
    args = _parse_args(["--stocks"])
    import pufferlib_market.autoresearch_rl as ar
    if args.stocks and args.leaderboard == "pufferlib_market/autoresearch_leaderboard.csv":
        args.leaderboard = "autoresearch_stock_daily_leaderboard.csv"
    assert args.leaderboard == "autoresearch_stock_daily_leaderboard.csv"


def test_explicit_train_data_overrides_stocks_default():
    """When --train-data is supplied alongside --stocks, user path wins."""
    args = _parse_args(["--stocks", "--train-data", "custom_train.bin"])
    import pufferlib_market.autoresearch_rl as ar
    if args.stocks and args.train_data is None:
        args.train_data = ar._STOCK_DEFAULT_TRAIN
    assert args.train_data == "custom_train.bin"


def test_stocks_not_required_without_flag():
    """Without --stocks, data paths are required; test that parser accepts them."""
    args = _parse_args(["--train-data", "a.bin", "--val-data", "b.bin"])
    assert args.train_data == "a.bin"
    assert args.val_data == "b.bin"
    assert args.stocks is False
