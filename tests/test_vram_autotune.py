from types import SimpleNamespace

from hftraining.config import ExperimentConfig, TrainingConfig
from hftraining import run_training as hf_run
from pufferlibtraining import train_ppo as ppo


def test_pufferlib_autotunes_batches(monkeypatch):
    args = SimpleNamespace(base_batch_size=24, rl_batch_size=96, device="cuda:0")

    monkeypatch.setattr(ppo, "_detect_vram_for_device", lambda device: 24 * 1024 ** 3)
    monkeypatch.setattr(ppo, "cli_flag_was_provided", lambda flag: False)

    ppo._maybe_autotune_batches(args)

    assert args.base_batch_size == 48
    assert args.rl_batch_size == 128


def test_hftraining_autotunes_batch_size(monkeypatch):
    config = ExperimentConfig()
    config.system.device = "cuda"
    default_batch = TrainingConfig().batch_size
    assert config.training.batch_size == default_batch

    monkeypatch.setattr(hf_run, "detect_total_vram_bytes", lambda device=None: 24 * 1024 ** 3)
    monkeypatch.setattr(hf_run, "cli_flag_was_provided", lambda flag: False)

    hf_run.maybe_autotune_batch_size(config, "cuda")

    assert config.training.batch_size == 24
