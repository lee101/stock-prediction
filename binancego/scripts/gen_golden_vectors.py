#!/usr/bin/env python3
"""Generate golden test vectors for Go simulator validation."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from src.metrics_utils import annualized_sortino, compute_step_returns

def gen_c_sim_golden():
    """Generate golden vectors matching market_sim.c simulate() logic."""
    np.random.seed(42)
    n = 100
    # Generate realistic BTC-like price series
    prices = [50000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))

    bars = []
    for i in range(n):
        c = prices[i]
        h = c * (1 + abs(np.random.normal(0, 0.005)))
        l = c * (1 - abs(np.random.normal(0, 0.005)))
        o = c * (1 + np.random.normal(0, 0.002))
        bars.append({"open": o, "high": h, "low": l, "close": c})

    actions = []
    for i in range(n):
        c = bars[i]["close"]
        bp = c * (1 - 0.005)  # buy 0.5% below close
        sp = c * (1 + 0.005)  # sell 0.5% above close
        ba = 50.0
        sa = 80.0
        actions.append({"buy_price": bp, "sell_price": sp, "buy_amount": ba, "sell_amount": sa})

    # Run through C sim logic in Python (matching market_sim.c exactly)
    cfg = {
        "max_leverage": 2.0,
        "can_short": False,
        "maker_fee": 0.001,
        "margin_hourly_rate": 0.0,
        "initial_cash": 10000.0,
        "fill_buffer_pct": 0.0005,
        "min_edge": 0.0,
        "max_hold_bars": 0,
        "intensity_scale": 1.0,
    }

    cash = cfg["initial_cash"]
    inventory = 0.0
    margin_cost = 0.0
    bars_in_pos = 0
    num_trades = 0
    iscale = cfg["intensity_scale"]
    fill_buf = cfg["fill_buffer_pct"]
    eq = [cash]

    for i in range(n):
        c = bars[i]["close"]
        h = bars[i]["high"]
        l = bars[i]["low"]
        bp = actions[i]["buy_price"]
        sp = actions[i]["sell_price"]
        ba = min(actions[i]["buy_amount"] * iscale, 100.0) / 100.0
        sa = min(actions[i]["sell_amount"] * iscale, 100.0) / 100.0

        equity = cash + inventory * c

        # margin interest
        if cash < 0:
            interest = (-cash) * cfg["margin_hourly_rate"]
            cash -= interest
            margin_cost += interest
        if inventory < 0:
            bv = (-inventory) * c
            interest = bv * cfg["margin_hourly_rate"]
            cash -= interest
            margin_cost += interest

        # force close
        if cfg["max_hold_bars"] > 0 and inventory > 0 and bars_in_pos >= cfg["max_hold_bars"]:
            fp = c * 0.999
            cash += inventory * fp * (1 - cfg["maker_fee"])
            num_trades += 1
            inventory = 0
            bars_in_pos = 0
            eq.append(cash)
            continue

        # min edge
        edge = (sp - bp) / bp if bp > 0 and sp > 0 else 0
        if cfg["min_edge"] > 0 and edge < cfg["min_edge"]:
            if inventory > 0:
                bars_in_pos += 1
            eq.append(cash + inventory * c)
            continue

        # sell first
        sold = False
        if sa > 0 and sp > 0 and h >= sp * (1 + fill_buf):
            sell_qty = 0.0
            if inventory > 0:
                sell_qty = sa * inventory
                sell_qty = min(sell_qty, inventory)
            elif cfg["can_short"]:
                eq_pos = max(equity, 0)
                max_sv = cfg["max_leverage"] * eq_pos
                q1 = sa * max_sv / (sp * (1 + cfg["maker_fee"]))
                q2 = max_sv / (sp * (1 + cfg["maker_fee"]))
                sell_qty = min(q1, q2)
            sell_qty = max(sell_qty, 0)
            if sell_qty > 0:
                cash += sell_qty * sp * (1 - cfg["maker_fee"])
                inventory -= sell_qty
                num_trades += 1
                sold = True
                if inventory <= 0:
                    bars_in_pos = 0

        # buy (no same-bar roundtrip)
        if not sold and ba > 0 and bp > 0 and l <= bp * (1 - fill_buf):
            eq_pos = max(equity, 0)
            max_bv = cfg["max_leverage"] * eq_pos - inventory * bp
            if max_bv > 0:
                buy_qty = ba * max_bv / (bp * (1 + cfg["maker_fee"]))
                if buy_qty > 0:
                    cash -= buy_qty * bp * (1 + cfg["maker_fee"])
                    inventory += buy_qty
                    num_trades += 1

        if inventory > 0:
            bars_in_pos += 1
        else:
            bars_in_pos = 0

        eq.append(cash + inventory * c)

    # close remaining
    if n > 0 and inventory != 0:
        lc = bars[n - 1]["close"]
        if inventory > 0:
            cash += inventory * lc * (1 - cfg["maker_fee"])
        else:
            cash -= (-inventory) * lc * (1 + cfg["maker_fee"])
        eq[-1] = cash

    total_return = eq[-1] / eq[0] - 1

    # Canonical sortino from metrics_utils.py
    returns = compute_step_returns(eq)
    sortino = annualized_sortino(returns, periods_per_year=8760)

    # max drawdown
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = (v - peak) / (peak + 1e-10)
        if dd < max_dd:
            max_dd = dd

    return {
        "bars": bars,
        "actions": actions,
        "config": cfg,
        "expected": {
            "total_return": total_return,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "final_equity": eq[-1],
            "num_trades": num_trades,
            "margin_cost_total": margin_cost,
            "equity_curve": eq,
        },
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "internal" / "testdata"
    out_dir.mkdir(parents=True, exist_ok=True)

    golden = gen_c_sim_golden()
    (out_dir / "golden_sim.json").write_text(json.dumps(golden, indent=2))
    print(f"wrote {out_dir / 'golden_sim.json'}")
    print(f"  total_return: {golden['expected']['total_return']:.6f}")
    print(f"  sortino: {golden['expected']['sortino']:.6f}")
    print(f"  max_drawdown: {golden['expected']['max_drawdown']:.6f}")
    print(f"  num_trades: {golden['expected']['num_trades']}")
    print(f"  equity points: {len(golden['expected']['equity_curve'])}")
