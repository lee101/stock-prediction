import pytest

from gymrl.train_ppo_allocator import WandBoardMetricsCallback


class _DummyMetricsLogger:
    def __init__(self) -> None:
        self.logged = []
        self.flushed = False

    def log(self, metrics, *, step=None, commit=None):
        self.logged.append((metrics, step))

    def flush(self) -> None:
        self.flushed = True


class _DummyLogger:
    def __init__(self) -> None:
        self.name_to_value = {}


class _DummyModel:
    def __init__(self, logger: _DummyLogger) -> None:
        self.logger = logger
        self.num_timesteps = 0

    def get_env(self):
        return object()


def test_wandboard_metrics_callback_logs_scalars():
    metrics_logger = _DummyMetricsLogger()
    callback = WandBoardMetricsCallback(metrics_logger, log_every=5)
    sb3_logger = _DummyLogger()
    sb3_logger.name_to_value = {
        "rollout/ep_rew_mean": 1.5,
        "time/time_elapsed": 2.0,
        "misc/non_numeric": "skip",
    }
    model = _DummyModel(sb3_logger)
    callback.init_callback(model)

    model.num_timesteps = 5
    assert callback.on_step() is True

    assert metrics_logger.logged, "Expected metrics to be logged on first eligible step."
    payload, step = metrics_logger.logged[0]
    assert step == 5
    assert payload["sb3/rollout/ep_rew_mean"] == pytest.approx(1.5)
    assert payload["sb3/time/time_elapsed"] == pytest.approx(2.0)
    assert "sb3/misc/non_numeric" not in payload
    assert payload["training/num_timesteps"] == pytest.approx(5.0)

    # Advance fewer than log_every timesteps -> no new log entry.
    model.logger.name_to_value["rollout/ep_rew_mean"] = 2.5
    model.num_timesteps = 6
    assert callback.on_step() is True
    assert len(metrics_logger.logged) == 1

    callback._on_training_end()
    assert metrics_logger.flushed is True
