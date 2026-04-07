#!/usr/bin/env python3
"""Generate parity test cases: Python reference sim + ctrader C sim comparison.

Implements the exact same algorithm as ctrader/market_sim.c simulate() in Python,
generates random test cases, runs both, and saves expected outputs for C-side verification.
"""
import json
import os
import struct

import numpy as np

ANNUALIZE_FACTOR = 8760.0


def py_simulate(
    high, low, close, buy_price, sell_price, buy_amount, sell_amount,
    max_leverage=1.0, can_short=0, maker_fee=0.001, margin_hourly_rate=0.0,
    initial_cash=10000.0, fill_buffer_pct=0.0, min_edge=0.0,
    max_hold_bars=0, intensity_scale=1.0,
):
    """Pure Python replica of ctrader/market_sim.c simulate()."""
    n_bars = len(close)
    cash = initial_cash
    inventory = 0.0
    margin_cost_total = 0.0
    bars_in_position = 0
    num_trades = 0
    equity_curve = np.zeros(n_bars + 1)
    equity_curve[0] = cash

    for i in range(n_bars):
        c = close[i]
        h = high[i]
        l = low[i]
        bp = buy_price[i]
        sp = sell_price[i]
        ba = np.clip(buy_amount[i] * intensity_scale, 0.0, 100.0) / 100.0
        sa = np.clip(sell_amount[i] * intensity_scale, 0.0, 100.0) / 100.0

        equity = cash + inventory * c

        if cash < 0.0:
            interest = (-cash) * margin_hourly_rate
            cash -= interest
            margin_cost_total += interest
        if inventory < 0.0:
            bv = (-inventory) * c
            interest = bv * margin_hourly_rate
            cash -= interest
            margin_cost_total += interest

        if max_hold_bars > 0 and inventory > 0.0 and bars_in_position >= max_hold_bars:
            fp = c * 0.999
            cash += inventory * fp * (1.0 - maker_fee)
            num_trades += 1
            inventory = 0.0
            bars_in_position = 0
            equity_curve[i + 1] = cash
            continue

        edge = (sp - bp) / bp if (bp > 0.0 and sp > 0.0) else 0.0
        if min_edge > 0.0 and edge < min_edge:
            if inventory > 0.0:
                bars_in_position += 1
            equity_curve[i + 1] = cash + inventory * c
            continue

        sold_this_bar = False
        if sa > 0.0 and sp > 0.0 and h >= sp * (1.0 + fill_buffer_pct):
            sell_qty = 0.0
            if inventory > 0.0:
                sell_qty = sa * inventory
                if sell_qty > inventory:
                    sell_qty = inventory
            elif can_short:
                max_sv = max_leverage * (equity if equity > 0.0 else 0.0)
                q1 = sa * max_sv / (sp * (1.0 + maker_fee))
                q2 = max_sv / (sp * (1.0 + maker_fee))
                sell_qty = min(q1, q2)
            if sell_qty < 0.0:
                sell_qty = 0.0
            if sell_qty > 0.0:
                cash += sell_qty * sp * (1.0 - maker_fee)
                inventory -= sell_qty
                num_trades += 1
                sold_this_bar = True
                if inventory <= 0.0:
                    bars_in_position = 0

        if not sold_this_bar and ba > 0.0 and bp > 0.0 and l <= bp * (1.0 - fill_buffer_pct):
            max_bv = max_leverage * (equity if equity > 0.0 else 0.0) - inventory * bp
            if max_bv > 0.0:
                buy_qty = ba * max_bv / (bp * (1.0 + maker_fee))
                if buy_qty > 0.0:
                    cash -= buy_qty * bp * (1.0 + maker_fee)
                    inventory += buy_qty
                    num_trades += 1

        if inventory > 0.0:
            bars_in_position += 1
        else:
            bars_in_position = 0

        equity_curve[i + 1] = cash + inventory * c

    if n_bars > 0 and inventory != 0.0:
        lc = close[n_bars - 1]
        if inventory > 0.0:
            cash += inventory * lc * (1.0 - maker_fee)
        else:
            cash -= (-inventory) * lc * (1.0 + maker_fee)
        inventory = 0.0
        equity_curve[n_bars] = cash

    return {
        "equity_curve": equity_curve.tolist(),
        "total_return": (equity_curve[n_bars] / equity_curve[0] - 1.0) if n_bars > 0 else 0.0,
        "final_equity": equity_curve[n_bars] if n_bars > 0 else initial_cash,
        "num_trades": num_trades,
        "margin_cost_total": margin_cost_total,
        "sortino": py_compute_sortino(equity_curve),
        "max_drawdown": py_compute_max_drawdown(equity_curve),
    }


def py_compute_sortino(equity_curve):
    n_eq = len(equity_curve)
    if n_eq < 2:
        return 0.0
    n_bars = n_eq - 1
    returns = []
    neg_returns = []
    for i in range(n_bars):
        denom = abs(equity_curve[i]) + 1e-10
        r = (equity_curve[i + 1] - equity_curve[i]) / denom
        returns.append(r)
        if r < 0.0:
            neg_returns.append(r)
    mean_ret = sum(returns) / n_bars
    if len(neg_returns) == 0:
        return mean_ret / 1e-10 * np.sqrt(ANNUALIZE_FACTOR)
    mean_neg = sum(neg_returns) / len(neg_returns)
    var_neg = sum(r * r for r in neg_returns) / len(neg_returns) - mean_neg * mean_neg
    if var_neg < 0.0:
        var_neg = 0.0
    std_neg = np.sqrt(var_neg) if len(neg_returns) > 0 else 1e-10
    return (mean_ret / (std_neg + 1e-10)) * np.sqrt(ANNUALIZE_FACTOR)


def py_compute_max_drawdown(equity_curve):
    """Match the canonical positive-magnitude convention used by C
    `compute_max_drawdown` in market_sim.c, by `pufferlib_market.binding_fallback`,
    `pufferlib_market.evaluate_holdout` (median_max_drawdown aggregation),
    `pufferlib_market.evaluate_sliding.compute_calmar`, and the C unit test
    `test_target_weights_max_drawdown_is_positive`.

    The previous version of this helper returned a *signed* (negative) value
    using `rlsys/utils.py:compute_max_drawdown` semantics, which made the
    parity fixture disagree with the C convention on 20/120 cases. The
    underlying PnL math was correct in both code paths — only the metric sign
    differed. Fixed 2026-04-07 during the ctrader audit (alpacaprod.md)."""
    running_max = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > running_max:
            running_max = v
        dd = (running_max - v) / (running_max + 1e-10)
        if dd > max_dd:
            max_dd = dd
    return max_dd


def generate_test_cases(n_cases=20):
    cases = []
    configs = [
        {"name": "base_1x", "max_leverage": 1.0, "can_short": 0, "maker_fee": 0.001,
         "margin_hourly_rate": 0.0, "initial_cash": 10000.0, "fill_buffer_pct": 0.0,
         "min_edge": 0.0, "max_hold_bars": 0, "intensity_scale": 1.0},
        {"name": "leverage_5x", "max_leverage": 5.0, "can_short": 0, "maker_fee": 0.001,
         "margin_hourly_rate": 0.0000071347, "initial_cash": 10000.0, "fill_buffer_pct": 0.0005,
         "min_edge": 0.0, "max_hold_bars": 0, "intensity_scale": 1.0},
        {"name": "short_allowed", "max_leverage": 1.0, "can_short": 1, "maker_fee": 0.001,
         "margin_hourly_rate": 0.0000071347, "initial_cash": 10000.0, "fill_buffer_pct": 0.0,
         "min_edge": 0.0, "max_hold_bars": 0, "intensity_scale": 1.0},
        {"name": "max_hold_6", "max_leverage": 1.0, "can_short": 0, "maker_fee": 0.001,
         "margin_hourly_rate": 0.0, "initial_cash": 10000.0, "fill_buffer_pct": 0.0,
         "min_edge": 0.0, "max_hold_bars": 6, "intensity_scale": 1.0},
        {"name": "min_edge_filter", "max_leverage": 1.0, "can_short": 0, "maker_fee": 0.001,
         "margin_hourly_rate": 0.0, "initial_cash": 10000.0, "fill_buffer_pct": 0.0,
         "min_edge": 0.02, "max_hold_bars": 0, "intensity_scale": 1.0},
        {"name": "intensity_2x", "max_leverage": 1.0, "can_short": 0, "maker_fee": 0.001,
         "margin_hourly_rate": 0.0, "initial_cash": 10000.0, "fill_buffer_pct": 0.0,
         "min_edge": 0.0, "max_hold_bars": 0, "intensity_scale": 2.0},
        {"name": "full_realistic", "max_leverage": 2.0, "can_short": 1, "maker_fee": 0.001,
         "margin_hourly_rate": 0.0000071347, "initial_cash": 10000.0, "fill_buffer_pct": 0.0005,
         "min_edge": 0.0, "max_hold_bars": 6, "intensity_scale": 1.0},
        {"name": "high_fee", "max_leverage": 1.0, "can_short": 0, "maker_fee": 0.01,
         "margin_hourly_rate": 0.0, "initial_cash": 50000.0, "fill_buffer_pct": 0.0,
         "min_edge": 0.0, "max_hold_bars": 0, "intensity_scale": 1.0},
    ]

    for seed in range(n_cases):
        rng = np.random.RandomState(seed)
        n_bars = 50 + rng.randint(0, 150)
        close_start = 50.0 + rng.rand() * 950.0
        rets = rng.randn(n_bars) * 0.015
        closes = close_start * np.cumprod(1.0 + rets)
        highs = closes * (1.0 + rng.rand(n_bars) * 0.025)
        lows = closes * (1.0 - rng.rand(n_bars) * 0.025)

        bp_raw = closes * (1.0 - rng.rand(n_bars) * 0.015)
        sp_raw = closes * (1.0 + rng.rand(n_bars) * 0.015)
        ba_raw = rng.rand(n_bars) * 60.0
        sa_raw = rng.rand(n_bars) * 60.0

        # zero out some bars
        mask_buy = rng.rand(n_bars) < 0.3
        mask_sell = rng.rand(n_bars) < 0.3
        bp_raw[mask_buy] = 0.0
        ba_raw[mask_buy] = 0.0
        sp_raw[mask_sell] = 0.0
        sa_raw[mask_sell] = 0.0

        cfg = configs[seed % len(configs)]

        result = py_simulate(
            high=highs, low=lows, close=closes,
            buy_price=bp_raw, sell_price=sp_raw,
            buy_amount=ba_raw, sell_amount=sa_raw,
            **{k: v for k, v in cfg.items() if k != "name"},
        )

        cases.append({
            "seed": seed,
            "config_name": cfg["name"],
            "n_bars": int(n_bars),
            "config": {k: v for k, v in cfg.items() if k != "name"},
            "high": highs.tolist(),
            "low": lows.tolist(),
            "close": closes.tolist(),
            "buy_price": bp_raw.tolist(),
            "sell_price": sp_raw.tolist(),
            "buy_amount": ba_raw.tolist(),
            "sell_amount": sa_raw.tolist(),
            "expected": {
                "total_return": result["total_return"],
                "final_equity": result["final_equity"],
                "num_trades": result["num_trades"],
                "margin_cost_total": result["margin_cost_total"],
                "sortino": result["sortino"],
                "max_drawdown": result["max_drawdown"],
                "equity_curve": result["equity_curve"],
            },
        })

    return cases


def write_binary_cases(cases, path):
    """Write test cases in binary format for C loading.

    Format per case:
      int32 n_bars
      7 doubles: max_leverage, maker_fee, margin_hourly_rate, initial_cash,
                 fill_buffer_pct, min_edge, intensity_scale
      2 int32s: can_short, max_hold_bars
      7 x double[n_bars]: high, low, close, buy_price, sell_price, buy_amount, sell_amount
      5 doubles: total_return, final_equity, sortino, max_drawdown, margin_cost_total
      int32: num_trades
      double[n_bars+1]: equity_curve
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(cases)))
        for case in cases:
            n = case["n_bars"]
            cfg = case["config"]
            exp = case["expected"]
            f.write(struct.pack("<i", n))
            f.write(struct.pack("<ddddddd",
                                cfg["max_leverage"], cfg["maker_fee"],
                                cfg["margin_hourly_rate"], cfg["initial_cash"],
                                cfg["fill_buffer_pct"], cfg["min_edge"],
                                cfg["intensity_scale"]))
            f.write(struct.pack("<ii", cfg["can_short"], cfg["max_hold_bars"]))
            for arr_name in ["high", "low", "close", "buy_price", "sell_price", "buy_amount", "sell_amount"]:
                arr = case[arr_name]
                f.write(struct.pack(f"<{n}d", *arr))
            f.write(struct.pack("<ddddd",
                                exp["total_return"], exp["final_equity"],
                                exp["sortino"], exp["max_drawdown"],
                                exp["margin_cost_total"]))
            f.write(struct.pack("<i", exp["num_trades"]))
            eq = exp["equity_curve"]
            f.write(struct.pack(f"<{n + 1}d", *eq))


def main():
    print("generating 20 test cases...")
    cases = generate_test_cases(20)

    os.makedirs("ctrader/tests", exist_ok=True)
    bin_path = "ctrader/tests/parity_cases.bin"
    write_binary_cases(cases, bin_path)
    print(f"wrote {len(cases)} cases to {bin_path}")

    json_path = "ctrader/tests/parity_cases.json"
    with open(json_path, "w") as f:
        json.dump(cases, f, indent=1)
    print(f"wrote {len(cases)} cases to {json_path}")

    for i, case in enumerate(cases):
        exp = case["expected"]
        print(f"  case {i:2d} [{case['config_name']:>16s}] "
              f"n={case['n_bars']:3d} trades={exp['num_trades']:3d} "
              f"ret={exp['total_return']:+.6f} sort={exp['sortino']:+.4f} "
              f"dd={exp['max_drawdown']:.4f} margin={exp['margin_cost_total']:.4f}")


if __name__ == "__main__":
    main()
