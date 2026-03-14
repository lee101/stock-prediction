"""RL+Gemini Hybrid Trading Bridge.

Takes RL model signals (action logits/probabilities) and uses Gemini 3.1 Flash Lite
with thinking to refine them into executable TradePlans with limit prices.

Flow:
  1. RL model produces action logits for each symbol
  2. Extract top-k confident signals (highest logit gap vs flat)
  3. Build a prompt with RL signal + price history + Chronos forecasts
  4. Gemini refines with limit prices, confidence, and reasoning
  5. Return executable TradePlan per symbol

Usage:
  from unified_orchestrator.rl_gemini_bridge import RLGeminiBridge
  bridge = RLGeminiBridge(checkpoint_path="pufferlib_market/checkpoints/best.pt")
  plans = bridge.generate_plans(symbols, price_data)
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_hourly_trader.gemini_wrapper import TradePlan


# ─── RL Signal Extraction ────────────────────────────────────────────


@dataclass
class RLSignal:
    """Extracted signal from RL model for one observation."""
    symbol_idx: int
    symbol_name: str
    direction: str  # "long", "short", or "flat"
    confidence: float  # 0-1, derived from softmax probability
    logit_gap: float  # gap between chosen action and flat action
    allocation_pct: float  # suggested position size
    level_offset_bps: float = 0.0  # suggested entry price offset around the current price


@dataclass(frozen=True)
class CheckpointSpec:
    """Architecture + action-space metadata inferred from a checkpoint."""

    obs_size: int
    num_actions: int
    hidden_size: int
    arch: str
    num_blocks: int
    alloc_bins: int
    level_bins: int
    max_offset_bps: float
    disable_shorts: bool


def decode_rl_action(
    logits: np.ndarray,
    num_symbols: int,
    symbol_names: list[str],
    alloc_bins: int = 1,
    level_bins: int = 1,
    max_offset_bps: float = 0.0,
    top_k: int = 3,
) -> list[RLSignal]:
    """Decode RL logits into ranked trading signals.

    Args:
        logits: Raw logits from policy network, shape (num_actions,)
        num_symbols: Number of symbols in the action space
        symbol_names: Human-readable symbol names
        alloc_bins: Number of allocation bins per symbol
        level_bins: Number of price level bins per symbol
        top_k: Return top-k signals by confidence

    Returns:
        List of RLSignal sorted by confidence (highest first)
    """
    probs = _softmax(logits)
    flat_prob = probs[0]
    flat_logit = logits[0]

    per_symbol_actions = alloc_bins * level_bins
    side_block = num_symbols * per_symbol_actions

    signals = []
    for sym_idx in range(num_symbols):
        sym_name = symbol_names[sym_idx] if sym_idx < len(symbol_names) else f"SYM{sym_idx}"

        # Aggregate probabilities for this symbol (long + short)
        long_start = 1 + sym_idx * per_symbol_actions
        long_end = long_start + per_symbol_actions
        long_prob = probs[long_start:long_end].sum()
        long_logit = logits[long_start:long_end].max()

        short_start = 1 + side_block + sym_idx * per_symbol_actions
        short_end = short_start + per_symbol_actions
        if short_end <= len(probs):
            short_prob = probs[short_start:short_end].sum()
            short_logit = logits[short_start:short_end].max()
        else:
            short_prob = 0.0
            short_logit = -float("inf")

        # Determine best direction for this symbol
        if long_prob > short_prob and long_prob > flat_prob * 0.3:
            direction = "long"
            confidence = float(long_prob / (long_prob + flat_prob + 1e-8))
            logit_gap = float(long_logit - flat_logit)
            # Best allocation bin
            best_idx = int(probs[long_start:long_end].argmax())
            best_alloc = best_idx // level_bins
            best_level = best_idx % max(1, level_bins)
            alloc_pct = float(best_alloc + 1) / max(alloc_bins, 1)
            if level_bins > 1 and max_offset_bps > 0:
                frac = best_level / max(1, level_bins - 1)
                level_offset_bps = (2.0 * frac - 1.0) * float(max_offset_bps)
            else:
                level_offset_bps = 0.0
        elif short_prob > long_prob and short_prob > flat_prob * 0.3:
            direction = "short"
            confidence = float(short_prob / (short_prob + flat_prob + 1e-8))
            logit_gap = float(short_logit - flat_logit)
            best_idx = int(probs[short_start:short_end].argmax())
            best_alloc = best_idx // level_bins
            best_level = best_idx % max(1, level_bins)
            alloc_pct = float(best_alloc + 1) / max(alloc_bins, 1)
            if level_bins > 1 and max_offset_bps > 0:
                frac = best_level / max(1, level_bins - 1)
                level_offset_bps = (2.0 * frac - 1.0) * float(max_offset_bps)
            else:
                level_offset_bps = 0.0
        else:
            direction = "flat"
            confidence = float(flat_prob)
            logit_gap = 0.0
            alloc_pct = 0.0
            level_offset_bps = 0.0

        signals.append(RLSignal(
            symbol_idx=sym_idx,
            symbol_name=sym_name,
            direction=direction,
            confidence=confidence,
            logit_gap=logit_gap,
            allocation_pct=min(1.0, alloc_pct),
            level_offset_bps=float(level_offset_bps),
        ))

    # Sort by confidence (non-flat first, then by confidence)
    signals.sort(key=lambda s: (s.direction != "flat", s.confidence), reverse=True)
    return signals[:top_k]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def build_portfolio_observation(
    features: np.ndarray,
    *,
    cash_ratio: float = 1.0,
    position_value_ratio: float = 0.0,
    unrealized_pnl_ratio: float = 0.0,
    hold_fraction: float = 0.0,
    step_fraction: float = 0.0,
    position_symbol_idx: Optional[int] = None,
    position_direction: str = "long",
) -> np.ndarray:
    """Build an observation matching the PPO market environment layout."""

    feature_arr = np.asarray(features, dtype=np.float32)
    if feature_arr.ndim != 2:
        raise ValueError(f"features must have shape [num_symbols, 16], got {feature_arr.shape}")
    num_symbols, feature_width = feature_arr.shape
    if feature_width != 16:
        raise ValueError(f"features must contain 16 values per symbol, got {feature_width}")

    obs = np.zeros(num_symbols * feature_width + 5 + num_symbols, dtype=np.float32)
    obs[: num_symbols * feature_width] = feature_arr.reshape(-1)

    base = num_symbols * feature_width
    obs[base + 0] = float(cash_ratio)
    obs[base + 1] = float(position_value_ratio)
    obs[base + 2] = float(unrealized_pnl_ratio)
    obs[base + 3] = float(hold_fraction)
    obs[base + 4] = float(step_fraction)

    if position_symbol_idx is not None:
        if not 0 <= int(position_symbol_idx) < num_symbols:
            raise ValueError(
                f"position_symbol_idx {position_symbol_idx} out of range for {num_symbols} symbols"
            )
        obs[base + 5 + int(position_symbol_idx)] = -1.0 if position_direction == "short" else 1.0

    return obs


# ─── Prompt Builder for Hybrid ───────────────────────────────────────


def _format_forecasts(forecast_1h: Optional[dict], forecast_24h: Optional[dict],
                      current_price: float) -> str:
    """Format Chronos2 forecasts for the prompt."""
    lines = []
    if forecast_1h:
        p50 = forecast_1h.get("predicted_close_p50", 0)
        p10 = forecast_1h.get("predicted_close_p10", 0)
        p90 = forecast_1h.get("predicted_close_p90", 0)
        hi = forecast_1h.get("predicted_high_p50", 0)
        lo = forecast_1h.get("predicted_low_p50", 0)
        delta = (p50 - current_price) / current_price * 100 if current_price > 0 else 0
        lines.append(
            f"  1h: close=${p50:.2f} ({delta:+.2f}%) "
            f"range=[${p10:.2f}, ${p90:.2f}] "
            f"high=${hi:.2f} low=${lo:.2f}"
        )
    if forecast_24h:
        p50 = forecast_24h.get("predicted_close_p50", 0)
        p10 = forecast_24h.get("predicted_close_p10", 0)
        p90 = forecast_24h.get("predicted_close_p90", 0)
        delta = (p50 - current_price) / current_price * 100 if current_price > 0 else 0
        lines.append(
            f"  24h: close=${p50:.2f} ({delta:+.2f}%) "
            f"range=[${p10:.2f}, ${p90:.2f}]"
        )
    return "\n".join(lines) if lines else "  (no forecasts available)"


def _format_prev_plan(prev_plan: Optional[dict]) -> str:
    """Format previous hour's trading plan for context."""
    if not prev_plan:
        return "  No previous plan"
    d = prev_plan.get("direction", "hold")
    bp = prev_plan.get("buy_price", 0)
    sp = prev_plan.get("sell_price", 0)
    conf = prev_plan.get("confidence", 0)
    filled = prev_plan.get("filled", "unknown")
    pnl = prev_plan.get("pnl_pct", 0)
    return (
        f"  Direction: {d} | Buy: ${bp:.2f} | Sell: ${sp:.2f} | "
        f"Conf: {conf:.2f} | Filled: {filled} | PnL: {pnl:+.2f}%"
    )


def build_hybrid_prompt(
    symbol: str,
    rl_signal: RLSignal,
    history_rows: list[dict],
    current_price: float,
    portfolio_context: str = "",
    forecast_1h: Optional[dict] = None,
    forecast_24h: Optional[dict] = None,
    prev_plan: Optional[dict] = None,
    all_rl_signals: Optional[list[RLSignal]] = None,
) -> str:
    """Build a prompt combining RL signal + Chronos2 forecasts + prev plan for Gemini.

    Gemini's job is to:
    1. Validate or override the RL signal based on market context + forecasts
    2. Set precise limit entry/exit prices optimizing for Sortino (risk-adjusted)
    3. Consider portfolio-level allocation across all signals
    """
    # Format history table (last 24 bars for full context)
    history_lines = []
    for row in history_rows[-24:]:
        ts = row.get("timestamp", "")
        if isinstance(ts, str) and len(ts) > 16:
            ts = ts[11:16]  # HH:MM
        history_lines.append(
            f"  {ts}  O:{row['open']:>10.2f}  H:{row['high']:>10.2f}  "
            f"L:{row['low']:>10.2f}  C:{row['close']:>10.2f}  "
            f"V:{row.get('volume', 0):>12.0f}"
        )
    history_table = "\n".join(history_lines)

    # Compute trend context
    closes = [r["close"] for r in history_rows if "close" in r]
    if len(closes) >= 2:
        ret_1h = (closes[-1] / closes[-2] - 1) * 100
        ret_12h = (closes[-1] / closes[-12] - 1) * 100 if len(closes) >= 12 else 0
        ret_24h = (closes[-1] / closes[-24] - 1) * 100 if len(closes) >= 24 else 0
    else:
        ret_1h = ret_12h = ret_24h = 0.0

    # Compute ATR for volatility context
    if len(history_rows) >= 12:
        ranges = [(r["high"] - r["low"]) for r in history_rows[-12:]]
        atr = sum(ranges) / len(ranges)
        atr_pct = atr / current_price * 100 if current_price > 0 else 0
    else:
        atr_pct = 0

    # RL signal
    rl_desc = (
        f"Direction: {rl_signal.direction.upper()}\n"
        f"  Confidence: {rl_signal.confidence:.1%}\n"
        f"  Logit gap vs FLAT: {rl_signal.logit_gap:+.2f}\n"
        f"  Suggested allocation: {rl_signal.allocation_pct:.0%}"
    )

    # Other symbols' signals for portfolio context
    portfolio_signals = ""
    if all_rl_signals:
        other = [s for s in all_rl_signals if s.symbol_name != symbol and s.direction != "flat"]
        if other:
            sig_lines = [f"  {s.symbol_name}: {s.direction.upper()} "
                         f"(conf={s.confidence:.0%})" for s in other[:5]]
            portfolio_signals = "\n## Other RL Signals (portfolio context)\n" + "\n".join(sig_lines)

    # Forecasts
    fc_text = _format_forecasts(forecast_1h, forecast_24h, current_price)

    # Previous plan
    prev_text = _format_prev_plan(prev_plan)

    prompt = f"""You are an expert quantitative portfolio optimizer. An RL trading model trained
on historical data has generated signals. Your job is to REFINE the signal into an
optimal limit-order plan that maximizes risk-adjusted returns (Sortino ratio).

## RL Model Signal for {symbol}
{rl_desc}

The RL model has 77-92% win rate in backtests. Trust the direction unless
market microstructure or forecasts strongly contradict it.

## Market Data
Symbol: {symbol}
Current Price: ${current_price:.2f}
1h Return: {ret_1h:+.2f}% | 12h Return: {ret_12h:+.2f}% | 24h Return: {ret_24h:+.2f}%
ATR (12h): {atr_pct:.2f}% of price

## Recent Price History (hourly, last 24 bars)
{history_table}

## Chronos2 ML Forecasts
{fc_text}

## Previous Hour's Plan
{prev_text}
{portfolio_signals}
{portfolio_context}

## Optimization Objective
MAXIMIZE Sortino ratio (risk-adjusted returns). This means:
- Set tight limit prices that capture edge while limiting downside
- Buy_price should be at a level where you'd be happy to own (support level or slight dip)
- Sell_price should be achievable within 1-3 hours based on ATR
- If Chronos2 forecasts disagree with RL direction, REDUCE confidence
- If Chronos2 forecasts AGREE, this is a high-confidence setup
- Consider the previous plan's outcome - did it fill? Was it profitable?

## Rules
- ALL orders must be LIMIT orders (never market)
- If LONG: buy_price < current_price, sell_price > buy_price
- If SHORT: sell_price > current_price, buy_price < sell_price (cover target)
- If no edge or signals conflict: direction="hold", prices=0
- Be precise with decimal places matching the asset
- Confidence 0.0-1.0: weight by both RL strength and forecast agreement

Return a JSON trade plan."""

    return prompt


# ─── Bridge Class ────────────────────────────────────────────────────


class RLGeminiBridge:
    """Loads an RL checkpoint and generates Gemini-refined trade plans."""

    def __init__(
        self,
        checkpoint_path: str,
        hidden_size: int = 1024,
        arch: str = "mlp",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.hidden_size = hidden_size
        self.arch = arch
        self._policy = None
        self._obs_norm = None
        self._checkpoint_spec: CheckpointSpec | None = None
        self._checkpoint_payload: dict | None = None

    def _load_checkpoint_payload(self) -> dict:
        if self._checkpoint_payload is None:
            payload = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
            if not isinstance(payload, dict):
                payload = {"model": payload}
            self._checkpoint_payload = payload
        return self._checkpoint_payload

    def get_checkpoint_spec(self) -> CheckpointSpec:
        """Infer policy architecture and action-space metadata from the checkpoint."""

        if self._checkpoint_spec is not None:
            return self._checkpoint_spec

        payload = self._load_checkpoint_payload()
        state_dict = payload.get("model", payload)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint payload in {self.checkpoint_path}")

        config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}

        if "input_proj.weight" in state_dict:
            arch = "resmlp"
            input_key = "input_proj.weight"
            hidden_size = int(state_dict[input_key].shape[0])
            obs_size = int(state_dict[input_key].shape[1])
            block_ids = {
                int(parts[1])
                for key in state_dict
                if key.startswith("blocks.")
                for parts in [key.split(".")]
                if len(parts) > 2 and parts[1].isdigit()
            }
            num_blocks = max(block_ids) + 1 if block_ids else int(config.get("num_blocks", 3))
        elif "encoder.0.weight" in state_dict:
            arch = "mlp"
            input_key = "encoder.0.weight"
            hidden_size = int(state_dict[input_key].shape[0])
            obs_size = int(state_dict[input_key].shape[1])
            num_blocks = 0
        else:
            raise ValueError(f"Could not infer architecture from checkpoint {self.checkpoint_path}")

        actor_bias_keys = [
            key
            for key in state_dict
            if key.startswith("actor.") and key.endswith(".bias") and key.split(".")[-2].isdigit()
        ]
        if not actor_bias_keys:
            raise ValueError(f"Could not infer action space from checkpoint {self.checkpoint_path}")
        actor_bias_key = max(actor_bias_keys, key=lambda key: int(key.split(".")[-2]))
        num_actions = int(state_dict[actor_bias_key].shape[0])

        alloc_bins = max(1, int(payload.get("action_allocation_bins", 1)))
        level_bins = max(1, int(payload.get("action_level_bins", 1)))
        max_offset_bps = float(payload.get("action_max_offset_bps", 0.0))
        disable_shorts = bool(payload.get("disable_shorts", False))

        self._checkpoint_spec = CheckpointSpec(
            obs_size=obs_size,
            num_actions=num_actions,
            hidden_size=hidden_size,
            arch=arch,
            num_blocks=num_blocks,
            alloc_bins=alloc_bins,
            level_bins=level_bins,
            max_offset_bps=max_offset_bps,
            disable_shorts=disable_shorts,
        )
        return self._checkpoint_spec

    def _load_policy(self, obs_size: int | None = None, num_actions: int | None = None) -> nn.Module:
        """Load policy from checkpoint."""
        if self._policy is not None:
            return self._policy

        from pufferlib_market.train import TradingPolicy, ResidualTradingPolicy

        spec = self.get_checkpoint_spec()
        if obs_size is not None and int(obs_size) != spec.obs_size:
            raise ValueError(
                f"Observation size {obs_size} does not match checkpoint {self.checkpoint_path} "
                f"(expected {spec.obs_size})"
            )
        if num_actions is not None and int(num_actions) != spec.num_actions:
            raise ValueError(
                f"Action count {num_actions} does not match checkpoint {self.checkpoint_path} "
                f"(expected {spec.num_actions})"
            )

        if spec.arch == "resmlp":
            policy = ResidualTradingPolicy(
                spec.obs_size,
                spec.num_actions,
                hidden=spec.hidden_size,
                num_blocks=max(1, spec.num_blocks),
            )
        else:
            policy = TradingPolicy(spec.obs_size, spec.num_actions, hidden=spec.hidden_size)

        payload = self._load_checkpoint_payload()
        state_dict = payload.get("model", payload)
        policy.load_state_dict(state_dict)

        policy.to(self.device)
        policy.eval()
        self._policy = policy
        return policy

    def get_rl_signals(
        self,
        obs: np.ndarray,
        num_symbols: int,
        symbol_names: list[str],
        alloc_bins: int | None = None,
        level_bins: int | None = None,
        top_k: int = 3,
    ) -> list[RLSignal]:
        """Get RL signals from observation.

        Args:
            obs: Observation array, shape (obs_size,)
            num_symbols: Number of symbols
            symbol_names: List of symbol name strings
            alloc_bins: Action allocation bins
            level_bins: Action level bins
            top_k: Number of top signals to return
        """
        spec = self.get_checkpoint_spec()
        alloc_bins = spec.alloc_bins if alloc_bins is None else int(alloc_bins)
        level_bins = spec.level_bins if level_bins is None else int(level_bins)
        per_symbol_actions = alloc_bins * level_bins
        num_actions = 1 + 2 * num_symbols * per_symbol_actions
        obs_size = obs.shape[-1]

        policy = self._load_policy(obs_size, num_actions)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            logits, value = policy(obs_t)
            logits_np = logits.squeeze(0).cpu().numpy()

        return decode_rl_action(
            logits_np, num_symbols, symbol_names,
            alloc_bins, level_bins, spec.max_offset_bps, top_k,
        )

    def generate_plans(
        self,
        rl_signals: list[RLSignal],
        price_histories: dict[str, list[dict]],
        current_prices: dict[str, float],
        model: str = "gemini-3.1-flash-lite-preview",
        thinking_level: str = "HIGH",
        portfolio_context: str = "",
        dry_run: bool = True,
        forecasts_1h: Optional[dict[str, dict]] = None,
        forecasts_24h: Optional[dict[str, dict]] = None,
        prev_plans: Optional[dict[str, dict]] = None,
    ) -> dict[str, TradePlan]:
        """Generate Gemini-refined trade plans from RL signals.

        Args:
            rl_signals: RL signals from get_rl_signals()
            price_histories: Dict of symbol -> list of OHLCV dicts
            current_prices: Dict of symbol -> current price
            model: Gemini model to use
            thinking_level: Gemini thinking level
            portfolio_context: Portfolio context string for prompt
            dry_run: If True, skip LLM calls and return RL-only plans
            forecasts_1h: Dict of symbol -> Chronos2 1h forecast dict
            forecasts_24h: Dict of symbol -> Chronos2 24h forecast dict
            prev_plans: Dict of symbol -> previous hour's plan dict

        Returns:
            Dict of symbol -> TradePlan
        """
        from llm_hourly_trader.providers import call_llm

        plans = {}
        for signal in rl_signals:
            sym = signal.symbol_name
            if signal.direction == "flat":
                continue
            if sym not in current_prices:
                continue

            history = price_histories.get(sym, [])
            price = current_prices[sym]

            if dry_run:
                # Generate plan directly from RL signal without LLM
                if signal.direction == "long":
                    buy_price = price * 0.998  # 0.2% below
                    sell_price = price * 1.01   # 1% above
                elif signal.direction == "short":
                    buy_price = price * 0.99    # cover 1% below
                    sell_price = price * 1.002  # entry 0.2% above
                else:
                    continue

                plans[sym] = TradePlan(
                    direction=signal.direction,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    confidence=signal.confidence,
                    reasoning=f"RL signal: {signal.direction} conf={signal.confidence:.2f} "
                              f"gap={signal.logit_gap:+.2f}",
                )
                continue

            # Build hybrid prompt with forecasts + prev plan + all signals
            fc_1h = (forecasts_1h or {}).get(sym)
            fc_24h = (forecasts_24h or {}).get(sym)
            prev = (prev_plans or {}).get(sym)

            prompt = build_hybrid_prompt(
                symbol=sym,
                rl_signal=signal,
                history_rows=history,
                current_price=price,
                portfolio_context=portfolio_context,
                forecast_1h=fc_1h,
                forecast_24h=fc_24h,
                prev_plan=prev,
                all_rl_signals=rl_signals,
            )

            try:
                plan = call_llm(
                    prompt, model=model, thinking_level=thinking_level,
                )
                plans[sym] = plan
            except Exception as e:
                # Fall back to RL-only plan
                if signal.direction == "long":
                    buy_price = price * 0.998
                    sell_price = price * 1.01
                elif signal.direction == "short":
                    buy_price = price * 0.99
                    sell_price = price * 1.002
                else:
                    continue

                plans[sym] = TradePlan(
                    direction=signal.direction,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    confidence=signal.confidence * 0.8,  # Discount for no LLM validation
                    reasoning=f"RL-only (LLM error: {e}): {signal.direction}",
                )

        return plans
