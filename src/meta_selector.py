"""Meta-selector: picks top-K models by trailing momentum of their simulated PnLs.

Used by trade_daily_stock_prod.py to replace the softmax ensemble with model selection.
Each model is simulated individually on recent daily bars, and the meta-selector
follows the K models with the best trailing performance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


@dataclass
class MetaSignal:
    selected_models: list[str]
    selected_actions: list[int]
    selected_symbols: list[str | None]
    confidences: list[float]
    model_returns: dict[str, float]


def _mask_shorts(logits: torch.Tensor, num_symbols: int) -> torch.Tensor:
    masked = logits.clone()
    masked[:, 1 + num_symbols:] = torch.finfo(masked.dtype).min
    return masked


class MetaSelector:
    """Loads N models, simulates each on recent bars, selects top-K by momentum."""

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        symbols: list[str],
        *,
        top_k: int = 2,
        lookback: int = 5,
        device: str = "cpu",
    ):
        self.symbols = list(symbols)
        self.top_k = top_k
        self.lookback = lookback
        self.device = device
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]

        from pufferlib_market.evaluate_holdout import load_policy
        self.policies = []
        self.names = []
        for cp in self.checkpoint_paths:
            try:
                loaded = load_policy(cp, len(symbols), features_per_sym=16, device=torch.device(device))
                self.policies.append(loaded.policy)
                self.names.append(cp.stem)
            except Exception as e:
                log.warning("skip %s: %s", cp.stem, e)

        # Per-model equity tracking (updated each day)
        self.model_equity: dict[str, list[float]] = {n: [10000.0] for n in self.names}
        self.model_positions: dict[str, int | None] = {n: None for n in self.names}
        self.model_entry_prices: dict[str, float] = {n: 0.0 for n in self.names}
        self._day_count = 0

    def update_model_pnls(self, prices: dict[str, float], fee_rate: float = 0.001):
        """Mark-to-market each model's simulated position."""
        for name in self.names:
            sym_idx = self.model_positions.get(name)
            if sym_idx is not None and 0 <= sym_idx < len(self.symbols):
                sym = self.symbols[sym_idx]
                price = prices.get(sym, 0.0)
                entry = self.model_entry_prices.get(name, 0.0)
                eq = self.model_equity[name]
                if entry > 0 and price > 0:
                    # Simple PnL: (price - entry) / entry * allocation
                    pnl_pct = (price - entry) / entry
                    new_eq = eq[-1] * (1 + pnl_pct)
                    eq.append(new_eq)
                else:
                    eq.append(eq[-1])
            else:
                self.model_equity[name].append(self.model_equity[name][-1])
        self._day_count += 1

    def get_meta_signal(
        self,
        features: np.ndarray,
        prices: dict[str, float],
        portfolio_obs_suffix: np.ndarray | None = None,
    ) -> MetaSignal:
        """Run all models, select top-K by trailing momentum, return their signals."""
        S = len(self.symbols)
        F = 16

        # Build obs
        obs_size = S * F + 5 + S
        obs = np.zeros(obs_size, dtype=np.float32)
        obs[:S * F] = features.reshape(-1)[:S * F]
        if portfolio_obs_suffix is not None:
            obs[S * F:] = portfolio_obs_suffix[:5 + S]

        # Run each model
        model_actions = {}
        model_symbols = {}
        model_confs = {}

        obs_t = torch.from_numpy(obs).to(self.device).view(1, -1)
        for i, (name, policy) in enumerate(zip(self.names, self.policies)):
            with torch.no_grad():
                logits, value = policy(obs_t)
            logits = _mask_shorts(logits, S)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(logits, dim=-1).item())

            if action == 0:
                model_actions[name] = 0
                model_symbols[name] = None
                model_confs[name] = float(probs[0, 0].item())
            elif 1 <= action <= S:
                model_actions[name] = action
                model_symbols[name] = self.symbols[action - 1]
                flat_prob = float(probs[0, 0].item())
                action_prob = float(probs[0, action].item())
                model_confs[name] = action_prob / max(action_prob + flat_prob, 1e-8)
            else:
                model_actions[name] = 0
                model_symbols[name] = None
                model_confs[name] = 0.0

            # Update simulated position for this model
            old_pos = self.model_positions.get(name)
            new_pos = (action - 1) if 1 <= action <= S else None

            if old_pos != new_pos:
                # Close old position
                if old_pos is not None and 0 <= old_pos < S:
                    old_sym = self.symbols[old_pos]
                    old_price = prices.get(old_sym, 0.0)
                    entry = self.model_entry_prices.get(name, 0.0)
                    if entry > 0 and old_price > 0:
                        pnl = (old_price - entry) / entry - 2 * 0.001  # round-trip fee
                        self.model_equity[name][-1] *= (1 + pnl)

                # Open new position
                if new_pos is not None and 0 <= new_pos < S:
                    self.model_entry_prices[name] = prices.get(self.symbols[new_pos], 0.0)
                else:
                    self.model_entry_prices[name] = 0.0
                self.model_positions[name] = new_pos

        # Select top-K by trailing momentum
        model_returns = {}
        for name in self.names:
            eq = self.model_equity[name]
            lb = min(self.lookback, len(eq) - 1)
            if lb > 0 and eq[-lb - 1] > 0:
                ret = (eq[-1] - eq[-lb - 1]) / eq[-lb - 1]
            else:
                ret = 0.0
            model_returns[name] = ret

        sorted_models = sorted(self.names, key=lambda n: model_returns[n], reverse=True)
        selected = sorted_models[:self.top_k]

        return MetaSignal(
            selected_models=selected,
            selected_actions=[model_actions[n] for n in selected],
            selected_symbols=[model_symbols[n] for n in selected],
            confidences=[model_confs[n] for n in selected],
            model_returns=model_returns,
        )
