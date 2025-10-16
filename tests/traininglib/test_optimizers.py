import pytest
import torch

from traininglib.benchmarking import RegressionBenchmark
from traininglib.optimizers import optimizer_registry


@pytest.mark.parametrize(
    "name",
    [
        "adamw",
        "adam",
        "sgd",
        "shampoo",
        "muon",
        "lion",
        "adafactor",
    ],
)
def test_registry_contains_expected_optimizers(name: str) -> None:
    assert name in optimizer_registry.names()


@pytest.mark.parametrize("optimizer_name", ["adamw", "shampoo", "muon", "lion", "adafactor"])
def test_benchmark_reduces_loss_for_each_optimizer(optimizer_name: str) -> None:
    bench = RegressionBenchmark(epochs=4, batch_size=64)
    result = bench.run(optimizer_name)
    assert result["final_loss"] < result["initial_loss"]


def test_shampoo_and_muon_compete_with_adamw() -> None:
    bench = RegressionBenchmark(epochs=6, batch_size=64)
    adamw_loss = bench.run("adamw")["final_loss"]
    shampoo_loss = bench.run("shampoo")["final_loss"]
    muon_loss = bench.run("muon")["final_loss"]

    # Allow a small tolerance because the synthetic dataset is noisy, but the
    # advanced optimizers should match or beat AdamW in practice.
    tolerance = adamw_loss * 0.05
    assert shampoo_loss <= adamw_loss + tolerance
    assert muon_loss <= adamw_loss + tolerance


def test_run_many_stats_are_reasonable() -> None:
    bench = RegressionBenchmark(epochs=3, batch_size=64)
    stats = bench.run_many("adamw", runs=3)
    assert stats["final_loss_std"] >= 0.0
    assert len(stats["runs"]) == 3
    seeds = {run["seed"] for run in stats["runs"]}
    assert len(seeds) == 3  # distinct seeds applied


def test_compare_reports_final_loss_mean_for_each_optimizer() -> None:
    bench = RegressionBenchmark(epochs=3, batch_size=64)
    results = bench.compare(["adamw", "shampoo"], runs=2)
    assert set(results.keys()) == {"adamw", "shampoo"}
    for name, payload in results.items():
        assert payload["final_loss_mean"] > 0
        assert "runs" in payload
        assert len(payload["runs"]) == 2
