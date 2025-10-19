from __future__ import annotations

import logging
import tempfile
import unittest
import os
from pathlib import Path

import wandboard
from wandboard import WandBoardLogger
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
