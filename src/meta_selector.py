"""Meta-selector: picks top-K models by trailing momentum of their simulated PnLs.

Used by trade_daily_stock_prod.py to replace the softmax ensemble with model selection.
Each model is simulated individually on recent daily bars, and the meta-selector
follows the K models with the best trailing performance.
"""
from __future__ import annotations

import json
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


def _compute_features_from_frames(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    day_idx: int,
) -> np.ndarray:
    """Compute 16-feature vector for each symbol at day_idx."""
    from pufferlib_market.inference_daily import compute_daily_features
    F = 16
    features = np.zeros((len(symbols), F), dtype=np.float32)
    for i, sym in enumerate(symbols):
        if sym not in frames:
            continue
        df = frames[sym]
        if day_idx >= len(df):
            continue
        window = df.iloc[:day_idx + 1]
        if len(window) < 2:
            continue
        try:
            features[i] = compute_daily_features(window)
        except Exception:
            pass
    return features


class MetaSelector:
    """Loads N models, simulates each on recent bars, selects top-K by momentum."""

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        symbols: list[str],
        *,
        top_k: int = 1,
        lookback: int = 3,
        fee_rate: float = 0.001,
        device: str = "cpu",
        state_path: Path | None = None,
        max_drawdown_filter: float = 0.05,
    ):
        self.symbols = list(symbols)
        self.top_k = top_k
        self.lookback = lookback
        self.fee_rate = fee_rate
        self.device = device
        self.max_drawdown_filter = max_drawdown_filter
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.state_path = state_path

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

        self.model_equity: dict[str, list[float]] = {n: [10000.0] for n in self.names}
        self.model_positions: dict[str, int | None] = {n: None for n in self.names}
        self.model_entry_prices: dict[str, float] = {n: 0.0 for n in self.names}
        self._day_count = 0

        if state_path and state_path.exists():
            self._load_state()

    def _load_state(self):
        try:
            data = json.loads(self.state_path.read_text())
            for name in self.names:
                if name in data.get("equity", {}):
                    self.model_equity[name] = data["equity"][name]
                if name in data.get("positions", {}):
                    self.model_positions[name] = data["positions"][name]
                if name in data.get("entry_prices", {}):
                    self.model_entry_prices[name] = data["entry_prices"][name]
            self._day_count = data.get("day_count", 0)
            log.info("loaded meta state: %d days, %d models", self._day_count, len(self.names))
        except Exception as e:
            log.warning("failed to load meta state: %s", e)

    def _save_state(self):
        if not self.state_path:
            return
        data = {
            "day_count": self._day_count,
            "equity": self.model_equity,
            "positions": {k: v for k, v in self.model_positions.items()},
            "entry_prices": self.model_entry_prices,
            "names": self.names,
            "symbols": self.symbols,
            "lookback": self.lookback,
            "top_k": self.top_k,
            "max_drawdown_filter": self.max_drawdown_filter,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2))

    def warmup_from_frames(
        self,
        frames: dict[str, pd.DataFrame],
        min_days: int = 30,
    ):
        """Simulate all models over historical bars to build momentum history."""
        if self._day_count > 0:
            log.info("already warmed up (%d days), skipping", self._day_count)
            return

        sample_df = next(iter(frames.values()))
        n_days = len(sample_df)
        if n_days < min_days:
            log.warning("only %d days available, need %d for warmup", n_days, min_days)
            return

        log.info("warming up meta-selector over %d days...", n_days)
        S = len(self.symbols)
        for day_idx in range(1, n_days):
            features = _compute_features_from_frames(frames, self.symbols, day_idx)
            prices = {}
            for i, sym in enumerate(self.symbols):
                if sym in frames and day_idx < len(frames[sym]):
                    prices[sym] = float(frames[sym]["close"].iloc[day_idx])
            self._step_models(features, prices)

        self._save_state()
        log.info("warmup done: %d days simulated", self._day_count)

    def _step_models(self, features: np.ndarray, prices: dict[str, float]):
        """Run all models for one day, update equity tracking."""
        S = len(self.symbols)
        F = 16
        obs_size = S * F + 5 + S
        obs = np.zeros(obs_size, dtype=np.float32)
        obs[:S * F] = features.reshape(-1)[:S * F]

        obs_t = torch.from_numpy(obs).to(self.device).view(1, -1)
        for name, policy in zip(self.names, self.policies):
            with torch.inference_mode():
                logits, _ = policy(obs_t)
            logits = _mask_shorts(logits, S)
            action = int(torch.argmax(logits, dim=-1).item())
            new_pos = (action - 1) if 1 <= action <= S else None
            old_pos = self.model_positions.get(name)

            if old_pos != new_pos:
                if old_pos is not None and 0 <= old_pos < S:
                    old_sym = self.symbols[old_pos]
                    old_price = prices.get(old_sym, 0.0)
                    entry = self.model_entry_prices.get(name, 0.0)
                    if entry > 0 and old_price > 0:
                        pnl = (old_price - entry) / entry - 2 * self.fee_rate
                        self.model_equity[name][-1] *= (1 + pnl)

                if new_pos is not None and 0 <= new_pos < S:
                    self.model_entry_prices[name] = prices.get(self.symbols[new_pos], 0.0)
                else:
                    self.model_entry_prices[name] = 0.0
                self.model_positions[name] = new_pos

            # Mark-to-market current position
            cur_pos = self.model_positions.get(name)
            if cur_pos is not None and 0 <= cur_pos < S:
                sym = self.symbols[cur_pos]
                price = prices.get(sym, 0.0)
                entry = self.model_entry_prices.get(name, 0.0)
                if entry > 0 and price > 0:
                    mtm = (price - entry) / entry
                    self.model_equity[name].append(self.model_equity[name][-1] * (1 + mtm))
                else:
                    self.model_equity[name].append(self.model_equity[name][-1])
            else:
                self.model_equity[name].append(self.model_equity[name][-1])

        self._day_count += 1

    def get_meta_signal(
        self,
        features: np.ndarray,
        prices: dict[str, float],
    ) -> MetaSignal:
        """Run all models, select top-K by trailing momentum, return their signals."""
        S = len(self.symbols)
        F = 16
        obs_size = S * F + 5 + S
        obs = np.zeros(obs_size, dtype=np.float32)
        obs[:S * F] = features.reshape(-1)[:S * F]

        model_actions = {}
        model_symbols = {}
        model_confs = {}

        obs_t = torch.from_numpy(obs).to(self.device).view(1, -1)
        for name, policy in zip(self.names, self.policies):
            with torch.inference_mode():
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

        # Step model simulations
        self._step_models(features, prices)
        self._save_state()

        # Select top-K by trailing momentum with drawdown filter
        model_returns = {}
        model_in_drawdown = {}
        for name in self.names:
            eq = self.model_equity[name]
            lb = min(self.lookback, len(eq) - 1)
            if lb > 0 and eq[-lb - 1] > 0:
                ret = (eq[-1] - eq[-lb - 1]) / eq[-lb - 1]
            else:
                ret = 0.0
            model_returns[name] = ret

            # Check recent drawdown (20-day window)
            recent = eq[-min(20, len(eq)):]
            if len(recent) > 1:
                peak = max(recent)
                dd = (peak - recent[-1]) / max(peak, 1e-8)
            else:
                dd = 0.0
            model_in_drawdown[name] = dd > self.max_drawdown_filter

        # Filter out models in drawdown, fallback to unfiltered if all filtered
        eligible = [n for n in self.names if not model_in_drawdown[n]]
        if not eligible:
            eligible = list(self.names)
        sorted_models = sorted(eligible, key=lambda n: model_returns[n], reverse=True)
        selected = sorted_models[:self.top_k]

        n_filtered = sum(1 for v in model_in_drawdown.values() if v)
        log.info("meta selected: %s (returns: %s, %d/%d filtered by dd>%.0f%%)",
                 selected, {n: f"{model_returns[n]:+.2%}" for n in selected},
                 n_filtered, len(self.names), self.max_drawdown_filter * 100)

        return MetaSignal(
            selected_models=selected,
            selected_actions=[model_actions[n] for n in selected],
            selected_symbols=[model_symbols[n] for n in selected],
            confidences=[model_confs[n] for n in selected],
            model_returns=model_returns,
        )
