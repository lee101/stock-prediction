from __future__ import annotations

from pathlib import Path

from scripts.run_crypto_lora_batch import SweepConfig, build_train_cmd, iter_sweep_configs


def test_build_train_cmd_passes_batch_size() -> None:
    cmd = build_train_cmd(
        run_id="bnb_batch",
        cfg=SweepConfig(
            symbol="BNBUSDT",
            preaug="differencing",
            context_length=512,
            batch_size=16,
            learning_rate=2e-5,
            num_steps=2400,
            prediction_length=24,
            lora_r=16,
        ),
        data_root=Path("trainingdatahourlybinance"),
        output_root=Path("chronos2_finetuned"),
        results_dir=Path("analysis/remote_runs/demo/lora_results"),
    )
    assert "--batch-size" in cmd
    batch_idx = cmd.index("--batch-size")
    assert cmd[batch_idx + 1] == "16"


def test_iter_sweep_configs_propagates_batch_size() -> None:
    configs = iter_sweep_configs(
        symbols=["BNBUSDT"],
        preaugs=["differencing"],
        context_lengths=[384, 512],
        batch_size=12,
        learning_rates=[2e-5, 5e-5],
        num_steps=2400,
        prediction_length=24,
        lora_r=16,
    )
    assert len(configs) == 4
    assert {cfg.batch_size for cfg in configs} == {12}
