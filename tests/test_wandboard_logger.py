from __future__ import annotations

import os
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import wandboard
from wandboard import WandBoardLogger
from unittest.mock import MagicMock, Mock, patch


class WandBoardLoggerLoggingTests(unittest.TestCase):
    def test_log_metrics_emits_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            with self.assertLogs(wandboard.logger, level=logging.INFO) as captured:
                with WandBoardLogger(
                    enable_wandb=False,
                    log_dir=log_dir,
                    tensorboard_subdir="metrics_enabled",
                    log_metrics=True,
                    metric_log_level=logging.INFO,
                ) as tracker:
                    tracker.log({"loss": 0.123, "accuracy": 0.987}, step=5)

        mirror_messages = [message for message in captured.output if "Mirror metrics" in message]
        self.assertTrue(mirror_messages, "Expected metrics mirror log message when logging is enabled.")
        self.assertIn("loss", mirror_messages[0])
        self.assertIn("accuracy", mirror_messages[0])

    def test_log_metrics_disabled_does_not_emit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            with self.assertLogs(wandboard.logger, level=logging.DEBUG) as captured:
                with WandBoardLogger(
                    enable_wandb=False,
                    log_dir=log_dir,
                    tensorboard_subdir="metrics_disabled",
                    log_metrics=False,
                ) as tracker:
                    tracker.log({"loss": 0.456}, step=3)

        mirror_messages = [message for message in captured.output if "Mirror metrics" in message]
        self.assertFalse(mirror_messages, "Metrics mirroring logs should be absent when logging is disabled.")

    def test_defaults_populate_project_and_entity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            log_dir = Path(tmp_dir)
            with WandBoardLogger(
                enable_wandb=False,
                log_dir=log_dir,
                tensorboard_subdir="defaults_populated",
            ) as tracker:
                self.assertEqual(tracker.project, "stock")
                self.assertEqual(tracker.entity, "lee101p")

    def test_blank_project_and_entity_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            log_dir = Path(tmp_dir)
            with WandBoardLogger(
                enable_wandb=False,
                log_dir=log_dir,
                tensorboard_subdir="blank_config",
                project="",
                entity="",
            ) as tracker:
                self.assertEqual(tracker.project, "")
                self.assertEqual(tracker.entity, "")

    def test_log_sweep_point_updates_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(wandboard, "_WANDB_AVAILABLE", True):
            writer = MagicMock()
            writer.flush = MagicMock()
            writer.close = MagicMock()
            with patch("wandboard.SummaryWriter", return_value=writer):
                table_mock = MagicMock()
                run_mock = MagicMock()
                run_mock.finish = MagicMock()
                stub_wandb = MagicMock()
                stub_wandb.init.return_value = run_mock
                stub_wandb.Table.return_value = table_mock
                stub_wandb.Image = MagicMock()
                with patch.object(wandboard, "wandb", stub_wandb):
                    with WandBoardLogger(
                        enable_wandb=True,
                        log_dir=Path(tmp_dir),
                        tensorboard_subdir="sweep",
                    ) as logger:
                        logger.log_sweep_point(
                            hparams={"learning_rate": 0.001, "optimizer": {"name": "adam"}},
                            metrics={"val": {"loss": 0.42}, "duration": 12.5},
                            step=3,
                            table_name="faltrain_sweep",
                        )

        writer.add_hparams.assert_called_once()
        stub_wandb.Table.assert_called_once()
        self.assertTrue(table_mock.add_data.called)
        run_mock.log.assert_called_once()
        logged_payload = run_mock.log.call_args[0][0]
        self.assertIn("faltrain_sweep", logged_payload)
        self.assertIn("faltrain_sweep/duration", logged_payload)


class WandbSweepAgentTests(unittest.TestCase):
    def test_register_and_run_invokes_agent(self) -> None:
        sweep_config = {"method": "grid", "parameters": {"lr": {"values": [0.0001, 0.001]}}}
        captured_configs: list[dict[str, Any]] = []

        def sweep_body(config: Mapping[str, Any]) -> None:
            captured_configs.append(dict(config))

        stub_wandb = MagicMock()
        stub_wandb.sweep.return_value = "sweep123"
        stub_wandb.agent = MagicMock()
        stub_wandb.config = {"lr": 0.001, "batch_size": 64}

        with patch.object(wandboard, "_WANDB_AVAILABLE", True), patch.object(
            wandboard, "wandb", stub_wandb
        ), patch("wandboard.multiprocessing.current_process") as current_process:
            current_process.return_value.name = "MainProcess"
            agent = wandboard.WandbSweepAgent(
                sweep_config=sweep_config,
                function=sweep_body,
                project="project-name",
                entity="entity-name",
                count=7,
            )
            sweep_id = agent.register()
            self.assertEqual(sweep_id, "sweep123")
            stub_wandb.sweep.assert_called_once()

            agent.run()

            stub_wandb.agent.assert_called_once()
            agent_kwargs = stub_wandb.agent.call_args.kwargs
            self.assertEqual(agent_kwargs["sweep_id"], "sweep123")
            self.assertEqual(agent_kwargs["count"], 7)
            self.assertEqual(agent_kwargs["project"], "project-name")
            self.assertEqual(agent_kwargs["entity"], "entity-name")

            sweep_callable = agent_kwargs["function"]
            stub_wandb.config = {"lr": 0.01, "batch_size": 128}
            sweep_callable()
            self.assertTrue(captured_configs)
            self.assertEqual(captured_configs[-1], {"lr": 0.01, "batch_size": 128})


if __name__ == "__main__":
    unittest.main()
