from pathlib import Path

from FastForecaster.config import FastForecasterConfig
from FastForecaster.sweep import _trial_config


def test_trial_config_uses_trial_specific_output_dir(tmp_path: Path):
    base = FastForecasterConfig(
        output_dir=tmp_path / "root",
        min_rows_per_symbol=400,
        lookback=128,
        horizon=16,
    )
    cfg = _trial_config(
        base,
        output_dir=tmp_path / "sweep" / "trial_x",
        return_loss_weight=0.2,
        direction_loss_weight=0.0,
        direction_margin_scale=16.0,
        learning_rate=2.5e-4,
        weight_decay=0.01,
        dropout=0.05,
        horizon_weight_power=0.3,
        qk_norm=True,
        use_ema_eval=True,
        ema_decay=0.999,
    )

    assert cfg.output_dir == tmp_path / "sweep" / "trial_x"
    assert cfg.checkpoint_dir == cfg.output_dir / "checkpoints"
    assert cfg.metrics_dir == cfg.output_dir / "metrics"

