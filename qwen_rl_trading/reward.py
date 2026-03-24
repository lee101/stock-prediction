"""Reward computation wrapping BinanceMarketSimulator for GRPO training."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from .schema import TradingPlan, validate_plan, plan_to_sim_actions
from .data_prompt import MarketSnapshot

log = logging.getLogger(__name__)

PENALTY_INVALID = -2.0
PENALTY_FLAT = -1.0
PENALTY_SIM_ERROR = -1.5

REWARD_SIM_CONFIG = SimulationConfig(
    maker_fee=0.001,
    initial_cash=10_000.0,
    max_hold_hours=6,
    fill_buffer_bps=5.0,
    decision_lag_bars=2,
)


def compute_sortino(returns: np.ndarray, periods_per_year: float = 8760.0) -> float:
    """Compute Sortino ratio from hourly returns array."""
    if len(returns) < 2:
        return 0.0
    mean_ret = float(np.mean(returns))
    neg = returns[returns < 0.0]
    if neg.size == 0:
        return max(mean_ret * 100, 0.0)  # all positive = good
    downside_dev = float(np.sqrt(np.mean(neg * neg)))
    if downside_dev <= 0:
        return 0.0
    return float(mean_ret / downside_dev * np.sqrt(periods_per_year))


def compute_reward(
    plan_text: str,
    snapshot: MarketSnapshot,
    config: Optional[SimulationConfig] = None,
    initial_cash: float = 10_000.0,
) -> float:
    """Parse plan, simulate, return Sortino as reward.

    Returns PENALTY_INVALID for unparseable output, PENALTY_FLAT for all-FLAT plans.
    """
    cfg = config or REWARD_SIM_CONFIG
    plan = validate_plan(plan_text)
    if plan is None:
        return PENALTY_INVALID

    actions_df = plan_to_sim_actions(plan, snapshot.forward_bars, initial_cash=initial_cash)
    if actions_df.empty:
        return PENALTY_FLAT

    try:
        sim = BinanceMarketSimulator(cfg)
        result = sim.run(bars=snapshot.forward_bars, actions=actions_df)
        equity = result.combined_equity
        if equity is None or len(equity) < 2:
            return 0.0
        returns = equity.pct_change().dropna().values
        sortino = compute_sortino(returns)
        total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1) if equity.iloc[0] > 0 else 0.0
        # blend sortino + return for more signal
        return float(np.clip(sortino * 0.5 + total_ret * 50, -5.0, 50.0))
    except Exception as e:
        log.warning("sim error: %s", e)
        return PENALTY_SIM_ERROR


def compute_reward_detailed(
    plan_text: str,
    snapshot: MarketSnapshot,
    config: Optional[SimulationConfig] = None,
) -> dict:
    """Like compute_reward but returns full metrics dict."""
    cfg = config or REWARD_SIM_CONFIG
    plan = validate_plan(plan_text)
    if plan is None:
        return {"reward": PENALTY_INVALID, "error": "invalid_json", "n_trades": 0}

    actions_df = plan_to_sim_actions(plan, snapshot.forward_bars)
    if actions_df.empty:
        return {"reward": PENALTY_FLAT, "error": "all_flat", "n_trades": 0}

    try:
        sim = BinanceMarketSimulator(cfg)
        result = sim.run(bars=snapshot.forward_bars, actions=actions_df)
        equity = result.combined_equity
        if equity is None or len(equity) < 2:
            return {"reward": 0.0, "sortino": 0.0, "total_return": 0.0, "n_trades": 0}
        returns = equity.pct_change().dropna().values
        sortino = compute_sortino(returns)
        total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1)
        max_dd = _max_drawdown(equity.values)
        n_trades = sum(len(sr.trades) for sr in result.per_symbol.values())
        reward = float(np.clip(sortino * 0.5 + total_ret * 50, -5.0, 50.0))
        return {
            "reward": reward,
            "sortino": sortino,
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "n_trades": n_trades,
        }
    except Exception as e:
        return {"reward": PENALTY_SIM_ERROR, "error": str(e), "n_trades": 0}


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


class GRPORewardFn:
    """Callable reward function compatible with TRL GRPOTrainer.

    TRL calls reward_fn(completions=...) -> list[float].
    We map window_ids embedded in prompts back to MarketSnapshots.
    """

    def __init__(
        self,
        snapshot_map: dict[str, MarketSnapshot],
        config: Optional[SimulationConfig] = None,
    ):
        self.snapshot_map = snapshot_map
        self.config = config or REWARD_SIM_CONFIG

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """Compute rewards for a batch of completions.

        The prompts contain [window:XXXX] tags that map to snapshots.
        """
        prompts = kwargs.get("prompts", [])
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            snapshot = self._find_snapshot(prompt)
            if snapshot is None:
                rewards.append(PENALTY_INVALID)
                continue
            r = compute_reward(completion, snapshot, self.config)
            rewards.append(r)
        return rewards

    def _find_snapshot(self, prompt: str) -> Optional[MarketSnapshot]:
        import re
        m = re.search(r"\[window:([a-f0-9]+)\]", prompt)
        if m:
            wid = m.group(1)
            return self.snapshot_map.get(wid)
        return None
