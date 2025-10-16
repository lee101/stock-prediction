"""
Benchmark helpers for comparing optimizers in a consistent, lightweight way.

The aim is to provide a repeatable harness that exercises optimizers on a small
synthetic regression task.  It runs quickly enough to live in the test suite
while still surfacing regressions when we tweak hyper-parameters or swap out an
optimizer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - import guarded in tests.
    raise RuntimeError(
        "torch is required to use traininglib.benchmarking. "
        "Install it via `pip install torch --index-url https://download.pytorch.org/whl/cpu`."
    ) from exc

from .optimizers import create_optimizer, optimizer_registry


@dataclass
class RegressionBenchmark:
    """Simple synthetic regression benchmark for optimizer comparisons."""

    input_dim: int = 16
    hidden_dim: int = 32
    output_dim: int = 1
    num_samples: int = 1024
    batch_size: int = 128
    noise_std: float = 0.05
    epochs: int = 5
    seed: int = 314
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover - validated in caller tests.
            raise RuntimeError("torch is required for the RegressionBenchmark.")
        self._seed_used = self.seed
        self._resample(self.seed)

    def _build_model(self) -> nn.Module:
        torch.manual_seed(self._seed_used)  # Deterministic initialisation across runs.
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        model.to(self.device)
        return model

    def _iterate_batches(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        generator = torch.Generator(device=self.device).manual_seed(self._seed_used)
        indices = torch.arange(self.num_samples, device=self.device)
        for _ in range(self.epochs):
            perm = indices[torch.randperm(self.num_samples, generator=generator)]
            for start in range(0, self.num_samples, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                yield self._features[batch_idx], self._targets[batch_idx]

    def _resample(self, seed: int) -> None:
        self._seed_used = seed
        torch.manual_seed(seed)
        self._features = torch.randn(self.num_samples, self.input_dim, device=self.device)
        weight = torch.randn(self.input_dim, self.output_dim, device=self.device)
        bias = torch.randn(self.output_dim, device=self.device)
        signal = self._features @ weight + bias
        noise = torch.randn_like(signal) * self.noise_std
        self._targets = signal + noise

    def run(
        self,
        optimizer_name: str,
        *,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_kwargs: Optional[MutableMapping[str, float]] = None,
        seed: Optional[int] = None,
    ) -> Mapping[str, float | List[float]]:
        """Train a tiny MLP on the synthetic task and report final metrics."""
        self._resample(seed or self.seed)
        model = self._build_model()
        criterion = nn.MSELoss()
        defaults = optimizer_registry.get_defaults(optimizer_name)
        effective_lr = lr if lr is not None else defaults.get("lr", 1e-3)
        config: Dict[str, float] = {
            "lr": effective_lr,
        }
        if weight_decay is not None:
            config["weight_decay"] = weight_decay
        elif "weight_decay" in defaults:
            config["weight_decay"] = defaults["weight_decay"]
        if optimizer_kwargs:
            config.update(optimizer_kwargs)

        optimizer = create_optimizer(optimizer_name, model.parameters(), **config)
        history: List[float] = []
        # Pre-calculate full-batch loss for comparability.
        with torch.no_grad():
            initial_loss = criterion(model(self._features), self._targets).item()
        history.append(initial_loss)

        for features, targets in self._iterate_batches():
            optimizer.zero_grad(set_to_none=True)
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                full_loss = criterion(model(self._features), self._targets).item()
            history.append(full_loss)

        return {
            "seed": self._seed_used,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "history": history,
        }

    def compare(
        self,
        optimizer_names: Iterable[str],
        *,
        lr_overrides: Optional[Mapping[str, float]] = None,
        weight_decay_overrides: Optional[Mapping[str, float]] = None,
        optimizer_kwargs: Optional[Mapping[str, Mapping[str, float]]] = None,
        runs: int = 1,
        base_seed: Optional[int] = None,
    ) -> Mapping[str, Mapping[str, float | List[float]]]:
        """Run the benchmark for several optimizers and return their metrics."""
        results: Dict[str, Mapping[str, float | List[float]]] = {}
        base = self.seed if base_seed is None else base_seed
        for name in optimizer_names:
            run_metrics: List[Mapping[str, float | List[float]]] = []
            for run_idx in range(runs):
                seed = base + run_idx
                run_metrics.append(
                    self.run(
                        name,
                        lr=lr_overrides.get(name) if lr_overrides else None,
                        weight_decay=(
                            weight_decay_overrides.get(name)
                            if weight_decay_overrides
                            else None
                        ),
                        optimizer_kwargs=(
                            dict(optimizer_kwargs[name])
                            if optimizer_kwargs and name in optimizer_kwargs
                            else None
                        ),
                        seed=seed,
                    )
                )
            final_losses = [float(result["final_loss"]) for result in run_metrics]
            results[name] = {
                "runs": run_metrics,
                "final_loss_mean": statistics.mean(final_losses),
                "final_loss_std": statistics.pstdev(final_losses) if len(final_losses) > 1 else 0.0,
            }
        return results

    def run_many(
        self,
        optimizer_name: str,
        *,
        runs: int = 3,
        base_seed: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        optimizer_kwargs: Optional[MutableMapping[str, float]] = None,
    ) -> Mapping[str, float | List[Mapping[str, float | List[float]]]]:
        """Convenience wrapper to run the same optimizer multiple times."""
        base = self.seed if base_seed is None else base_seed
        run_metrics: List[Mapping[str, float | List[float]]] = []
        for run_idx in range(runs):
            seed = base + run_idx
            run_metrics.append(
                self.run(
                    optimizer_name,
                    lr=lr,
                    weight_decay=weight_decay,
                    optimizer_kwargs=optimizer_kwargs,
                    seed=seed,
                )
            )
        final_losses = [float(result["final_loss"]) for result in run_metrics]
        return {
            "runs": run_metrics,
            "final_loss_mean": statistics.mean(final_losses),
            "final_loss_std": statistics.pstdev(final_losses) if len(final_losses) > 1 else 0.0,
        }
