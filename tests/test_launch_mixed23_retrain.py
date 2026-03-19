"""Tests for scripts/launch_mixed23_retrain.py config generation."""
import pytest
from scripts.launch_mixed23_retrain import (
    TrainConfig,
    build_configs,
    config_to_train_cmd,
    config_to_eval_cmd,
    TRAIN_DATA,
    CHECKPOINT_ROOT,
    REPO,
)


@pytest.mark.unit
def test_build_configs_returns_five():
    configs = build_configs()
    assert len(configs) == 5


@pytest.mark.unit
def test_config_names_unique():
    configs = build_configs()
    names = [c.name for c in configs]
    assert len(names) == len(set(names))


@pytest.mark.unit
def test_cosine_lr_config():
    cfg = [c for c in build_configs() if c.name == "cosine_lr_ent_anneal"][0]
    assert cfg.lr_schedule == "cosine"
    assert cfg.lr_warmup_frac == 0.10
    assert cfg.anneal_ent is True
    assert cfg.ent_coef == 0.05
    assert cfg.ent_coef_end == 0.01


@pytest.mark.unit
def test_wide_config():
    cfg = [c for c in build_configs() if c.name == "wide_h2048"][0]
    assert cfg.hidden_size == 2048
    assert cfg.arch == "resmlp"


@pytest.mark.unit
def test_deep_config():
    cfg = [c for c in build_configs() if c.name == "deep_4block_lowlr"][0]
    assert cfg.lr == 1e-4
    assert cfg.arch == "resmlp"


@pytest.mark.unit
def test_high_entropy_config():
    cfg = [c for c in build_configs() if c.name == "high_entropy_gc05"][0]
    assert cfg.ent_coef == 0.03
    assert cfg.max_grad_norm == 0.5


@pytest.mark.unit
def test_long_rollout_config():
    cfg = [c for c in build_configs() if c.name == "long_rollout_bigbatch"][0]
    assert cfg.rollout_len == 512
    assert cfg.minibatch_size == 4096


@pytest.mark.unit
def test_train_cmd_has_data_path():
    cfg = build_configs()[0]
    cmd = config_to_train_cmd(cfg, REPO)
    assert "--data-path" in cmd
    assert any(TRAIN_DATA in arg for arg in cmd)


@pytest.mark.unit
def test_train_cmd_anneal_ent_flag():
    cfg = [c for c in build_configs() if c.anneal_ent][0]
    cmd = config_to_train_cmd(cfg, REPO)
    assert "--anneal-ent" in cmd


@pytest.mark.unit
def test_train_cmd_no_anneal_ent_when_false():
    cfg = TrainConfig(name="test", anneal_ent=False)
    cmd = config_to_train_cmd(cfg, REPO)
    assert "--anneal-ent" not in cmd


@pytest.mark.unit
def test_train_cmd_checkpoint_dir():
    cfg = build_configs()[0]
    cmd = config_to_train_cmd(cfg, REPO)
    idx = cmd.index("--checkpoint-dir")
    ckpt_dir = cmd[idx + 1]
    assert CHECKPOINT_ROOT in ckpt_dir
    assert cfg.name in ckpt_dir


@pytest.mark.unit
def test_eval_cmd_output_path():
    cfg = build_configs()[0]
    cmd = config_to_eval_cmd(cfg, REPO)
    assert "comprehensive_marketsim_eval.py" in " ".join(cmd)
    assert "--output" in cmd


@pytest.mark.unit
def test_all_configs_valid_arch():
    for cfg in build_configs():
        assert cfg.arch in ("mlp", "resmlp")


@pytest.mark.unit
def test_all_configs_positive_lr():
    for cfg in build_configs():
        assert cfg.lr > 0


@pytest.mark.unit
def test_train_cmd_obs_norm_flag():
    cfg = TrainConfig(name="test", obs_norm=True)
    cmd = config_to_train_cmd(cfg, REPO)
    assert "--obs-norm" in cmd


@pytest.mark.unit
def test_train_cmd_clip_vloss_flag():
    cfg = TrainConfig(name="test", clip_vloss=True)
    cmd = config_to_train_cmd(cfg, REPO)
    assert "--clip-vloss" in cmd
