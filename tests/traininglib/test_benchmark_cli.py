from traininglib import benchmark_cli
import builtins
import pytest


def test_cli_outputs_table(monkeypatch):
    captured = {}

    def fake_print(msg):
        captured["msg"] = msg

    monkeypatch.setattr(builtins, "print", fake_print)
    output = benchmark_cli.run_cli(
        ["--optimizers", "adamw", "shampoo", "--runs", "1", "--epochs", "2", "--batch-size", "32"]
    )
    assert "adamw" in output
    assert "shampoo" in output
    assert captured["msg"] == output


def test_cli_raises_for_unknown_optimizer():
    with pytest.raises(ValueError):
        benchmark_cli.run_cli(["--optimizers", "unknown_opt"])
