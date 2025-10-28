import pathlib

import numpy as np
import pandas as pd

from pufferlibtraining import pufferrl


def _write_dummy_data(tmp_path, symbol="AAA", rows=128):
    idx = np.arange(rows)
    data = pd.DataFrame(
        {
            "timestamps": idx,
            "open": 100 + np.sin(idx / 10) * 0.5,
            "high": 100.5 + np.sin(idx / 9) * 0.5,
            "low": 99.5 + np.sin(idx / 11) * 0.5,
            "close": 100 + np.sin(idx / 8) * 0.5,
            "volume": np.random.lognormal(mean=12, sigma=0.2, size=rows),
        }
    )
    path = tmp_path / f"{symbol}.csv"
    data.to_csv(path, index=False)
    return path.parent


def test_load_config_defaults(tmp_path):
    cfg, env_cfg = pufferrl._load_config(None)
    assert cfg.rollout_len == 128
    assert env_cfg.context_len == 128


def test_train_smoke(tmp_path, monkeypatch):
    data_dir = _write_dummy_data(tmp_path)
    cfg_path = tmp_path / "rl.ini"
    cfg_path.write_text(
        "\n".join(
            [
                "[vec]",
                "num_envs = 4",
                "num_workers = 0",
                "",
                "[train]",
                "rollout_len = 4",
                "minibatches = 2",
                "update_iters = 1",
                "learning_rate = 1e-3",
                "max_updates = 1",
                "mixed_precision = fp32",
                "torch_compile = false",
                "gamma = 0.9",
                "",
                "[env]",
                f"data_dir = {data_dir}",
                "context_len = 8",
                "episode_len = 16",
            ]
        )
    )

    pufferrl.train(str(cfg_path))

