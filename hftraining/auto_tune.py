#!/usr/bin/env python3
"""
Auto-tuning utilities for hftraining

Currently supports batch size tuning to maximize throughput within memory limits.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


class AutoBatchTuner:
    """Tune batch size by trying nearby candidates and measuring throughput.

    Can benchmark against a fixed test batch for stable measurements.
    """

    def __init__(self, device: torch.device, steps: int = 10, num_workers: int = 0, pin_memory: bool = False):
        self.device = device
        self.steps = max(3, int(steps))
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _sys_metrics(self) -> Dict[str, float]:
        m = {}
        try:
            if torch.cuda.is_available():
                m['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(0) / (1024**2)
        except Exception:
            pass
        return m

    def _try_batch(self, trainer, dataset=None, batch_size: int = 0, test_batch: Dict[str, torch.Tensor] | None = None) -> Tuple[bool, Dict[str, float]]:
        """Attempt a short training loop with the given batch size.

        If `test_batch` is provided, reuse it for all steps; otherwise build a
        DataLoader with shuffle=False and benchmark on the first batch repeated.
        """
        samples = 0
        times: List[float] = []
        ok = True
        try:
            model = trainer.model
            model.train()
            non_block = bool(torch.cuda.is_available())

            if test_batch is None:
                if dataset is None:
                    raise ValueError("dataset or test_batch must be provided")
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
                it = iter(loader)
                batch = next(it)
            else:
                batch = test_batch

            # Move one batch to device once, reuse for multiple benchmark steps
            batch = {k: v.to(self.device, non_blocking=non_block) for k, v in batch.items()}

            # Reset peak memory for cleaner measurement
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            steps = 0
            while steps < self.steps:
                t0 = time.time()
                # Prefer a dedicated benchmark step if available to avoid optimizer stepping
                if hasattr(trainer, 'benchmark_step'):
                    trainer.benchmark_step(batch)
                else:
                    trainer.training_step(batch)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                dt = max(1e-9, time.time() - t0)
                times.append(dt)
                samples += batch_size if batch_size else (batch['input_ids'].shape[0])
                steps += 1
        except torch.cuda.OutOfMemoryError:
            ok = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except StopIteration:
            pass
        except Exception:
            ok = False
        avg_step = sum(times) / len(times) if times else float('inf')
        sps = samples / sum(times) if times and sum(times) > 0 else 0.0
        metrics = {
            'avg_step_s': avg_step,
            'samples_per_sec': sps,
        }
        metrics.update(self._sys_metrics())
        return ok, metrics

    def suggest(self, initial_batch: int) -> List[int]:
        """Generate candidate batch sizes around the initial value."""
        base = max(1, int(initial_batch))
        candidates = sorted(set([max(1, base // 2), base, base * 2, base * 4]))
        return candidates

    def tune(self, trainer, dataset=None, initial_batch: int = 0, test_batch: Dict[str, torch.Tensor] | None = None) -> Dict[str, float]:
        """Run tuning and return the best batch size and metrics.

        Pass `test_batch` to run over a fixed evaluation batch for stability.
        """
        results = []
        for b in self.suggest(initial_batch):
            ok, m = self._try_batch(trainer, dataset=dataset, batch_size=b, test_batch=test_batch)
            results.append({'batch_size': b, 'ok': ok, **m})
        # Filter feasible
        feasible = [r for r in results if r['ok'] and r['samples_per_sec'] > 0]
        if not feasible:
            # Fallback to initial
            return {'batch_size': initial_batch, 'samples_per_sec': 0.0, 'avg_step_s': float('inf')}
        # Pick highest throughput
        best = max(feasible, key=lambda r: r['samples_per_sec'])
        return best
