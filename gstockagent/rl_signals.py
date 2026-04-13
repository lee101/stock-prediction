"""Extract per-symbol RL signals from trained PufferLib crypto30 ensemble."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CHECKPOINTS = [
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s2.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s19.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s21.pt",
    REPO / "pufferlib_market/checkpoints/crypto30_ensemble/s23.pt",
]

CRYPTO30_SYMBOLS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AVAXUSD",
    "LINKUSD", "AAVEUSD", "LTCUSD", "XRPUSD", "DOTUSD",
    "UNIUSD", "NEARUSD", "APTUSD", "ICPUSD", "SHIBUSD",
    "ADAUSD", "FILUSD", "ARBUSD", "OPUSD", "INJUSD",
    "SUIUSD", "TIAUSD", "SEIUSD", "ATOMUSD", "ALGOUSD",
    "BCHUSD", "BNBUSD", "TRXUSD", "PEPEUSD", "MATICUSD",
]

_ensemble = None


def _load_ensemble():
    global _ensemble
    if _ensemble is not None:
        return _ensemble
    from pufferlib_market.inference_daily import DailyPPOTrader
    traders = []
    for cp in DEFAULT_CHECKPOINTS:
        if cp.exists():
            t = DailyPPOTrader(str(cp), device="cpu", long_only=True,
                               symbols=CRYPTO30_SYMBOLS,
                               allow_unsafe_checkpoint_loading=True)
            traders.append(t)
    _ensemble = traders
    return traders


def get_rl_signals(daily_dfs: dict, prices: dict) -> dict:
    """
    Get per-symbol RL signal from ensemble of PPO models.

    Args:
        daily_dfs: {symbol: DataFrame} with OHLCV, 60+ rows
        prices: {symbol: float} current prices

    Returns:
        dict of {symbol: {"action": str, "confidence": float, "value": float,
                          "long_prob": float, "short_prob": float, "flat_prob": float}}
    """
    traders = _load_ensemble()
    if not traders:
        return {}

    from pufferlib_market.inference_daily import compute_daily_features

    # Compute features for all symbols
    num_sym = len(CRYPTO30_SYMBOLS)
    fps = traders[0].features_per_sym
    features = np.zeros((num_sym, fps), dtype=np.float32)
    for i, sym in enumerate(CRYPTO30_SYMBOLS):
        short_sym = sym.replace("USD", "")
        if short_sym in daily_dfs:
            try:
                features[i, :16] = compute_daily_features(daily_dfs[short_sym])
            except Exception:
                pass

    # Get ensemble probabilities
    all_probs = []
    all_values = []
    for trader in traders:
        obs = trader.build_observation(features, prices)
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.inference_mode():
            logits, value = trader.policy(obs_t)
            logits = trader.apply_action_constraints(logits)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            all_probs.append(probs)
            all_values.append(value.item())

    # Average ensemble
    avg_probs = torch.stack(all_probs).mean(dim=0)
    avg_value = np.mean(all_values)

    # Aggregate per-symbol probabilities
    signals = {}
    flat_prob = avg_probs[0].item()

    for i, sym in enumerate(CRYPTO30_SYMBOLS):
        short_sym = sym.replace("USD", "")
        # Sum all long actions for this symbol
        long_prob = 0.0
        short_prob = 0.0

        trader = traders[0]
        if trader.per_symbol_actions > 1:
            # Actions: 0=flat, 1..side_block=long, side_block+1..=short
            for a in range(trader.per_symbol_actions):
                long_idx = 1 + i * trader.per_symbol_actions + a
                if long_idx < len(avg_probs):
                    long_prob += avg_probs[long_idx].item()
                short_idx = 1 + trader.side_block + i * trader.per_symbol_actions + a
                if short_idx < len(avg_probs):
                    short_prob += avg_probs[short_idx].item()
        else:
            long_idx = 1 + i
            if long_idx < len(avg_probs):
                long_prob = avg_probs[long_idx].item()
            short_idx = 1 + num_sym + i
            if short_idx < len(avg_probs):
                short_prob = avg_probs[short_idx].item()

        if long_prob > short_prob and long_prob > 0.01:
            action = "long"
            conf = long_prob
        elif short_prob > long_prob and short_prob > 0.01:
            action = "short"
            conf = short_prob
        else:
            action = "flat"
            conf = flat_prob

        signals[short_sym] = {
            "action": action,
            "confidence": round(conf, 4),
            "value": round(avg_value, 4),
            "long_prob": round(long_prob, 4),
            "short_prob": round(short_prob, 4),
            "flat_prob": round(flat_prob, 4),
        }

    return signals


def format_rl_signals_for_prompt(signals: dict) -> str:
    """Format RL signals as a table for LLM prompt injection."""
    if not signals:
        return "No RL signals available."
    lines = ["Symbol | RL_Action | Confidence | Long% | Short% | Flat%"]
    lines.append("---|---|---|---|---|---")
    for sym in sorted(signals.keys()):
        s = signals[sym]
        if s["long_prob"] < 0.005 and s["short_prob"] < 0.005:
            continue
        lines.append(
            f"{sym} | {s['action']} | {s['confidence']:.1%} | "
            f"{s['long_prob']:.1%} | {s['short_prob']:.1%} | {s['flat_prob']:.1%}"
        )
    return "\n".join(lines)
