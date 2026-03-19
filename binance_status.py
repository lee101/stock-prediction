#!/usr/bin/env python3
"""Single command to print full Binance account status."""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.binan import binance_wrapper as bw

SEP = "=" * 72

def fmt_usd(v): return f"${v:,.2f}"
def fmt_ts(ms): return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%m-%d %H:%M")

def print_table(headers, rows, aligns=None):
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    if not aligns:
        aligns = ["<"] * cols
    hdr = " | ".join(f"{h:{a}{w}}" for h, w, a in zip(headers, widths, aligns))
    print(hdr)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(f"{str(c):{a}{w}}" for c, w, a in zip(r, widths, aligns)))

def main():
    print(f"\n{SEP}")
    print(f"  BINANCE ACCOUNT STATUS  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(SEP)

    # balances
    val = bw.get_account_value_usdt()
    print(f"\n-- BALANCES (total: {fmt_usd(val['total_usdt'])}) --")
    rows = []
    for a in sorted(val["assets"], key=lambda x: -x["value_usdt"]):
        if a["value_usdt"] < 0.01:
            continue
        rows.append((a["asset"], f"{a['amount']:.6f}", fmt_usd(a["value_usdt"])))
    print_table(["Asset", "Amount", "Value"], rows, ["<", ">", ">"])

    # realized trading pnl from state file
    pnl_path = Path(__file__).resolve().parent.parent / "strategy_state" / "binanceneural_pnl_state_live.json"
    if pnl_path.exists():
        pnl_state = json.loads(pnl_path.read_text())
        total_realized = sum(s.get("realized_pnl", 0) for s in pnl_state.values())
        sign = "+" if total_realized >= 0 else ""
        print(f"\n-- REALIZED TRADING PNL: {sign}{fmt_usd(total_realized)} --")
        rows = []
        for sym, s in sorted(pnl_state.items(), key=lambda x: -abs(x[1].get("realized_pnl", 0))):
            rpnl = s.get("realized_pnl", 0)
            pos = s.get("position_qty", 0)
            sign = "+" if rpnl >= 0 else ""
            rows.append((sym, f"{sign}${rpnl:,.2f}", f"{pos:.4f}", s.get("mode", "?")))
        print_table(["Symbol", "Realized", "Position", "Mode"], rows, ["<", ">", ">", "<"])
    else:
        print("\n-- REALIZED TRADING PNL: no state file --")

    # selector state
    state_path = Path(__file__).resolve().parent.parent / "strategy_state" / "selector_state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        sym = st.get("open_symbol", "none")
        if sym and sym != "none":
            price = st.get("open_price", "?")
            ts = st.get("open_ts", "?")
            print(f"\n-- SELECTOR POSITION: {sym} @ ${price} (since {ts}) --")
        else:
            print("\n-- SELECTOR POSITION: flat --")

    # open orders
    orders = bw.get_open_orders()
    print(f"\n-- OPEN ORDERS: {len(orders)} --")
    if orders:
        rows = []
        for o in orders:
            rows.append((o["symbol"], o["side"], o["origQty"], o["price"], o["status"]))
        print_table(["Pair", "Side", "Qty", "Price", "Status"], rows)

    # recent orders across all pairs
    print(f"\n-- RECENT ORDERS --")
    all_orders = []
    for sym in ["BTCFDUSD", "ETHFDUSD", "SOLFDUSD", "BTCU", "ETHU", "SOLU"]:
        try:
            ords = bw.get_all_orders(sym)
            for o in ords[-10:]:
                all_orders.append(o)
        except Exception:
            pass
    all_orders.sort(key=lambda x: x["time"], reverse=True)
    rows = []
    for o in all_orders[:15]:
        rows.append((
            fmt_ts(o["time"]),
            o["symbol"],
            o["side"],
            o["origQty"],
            o["price"],
            o["status"],
            o.get("executedQty", "?"),
        ))
    if rows:
        print_table(["Time", "Pair", "Side", "Qty", "Price", "Status", "Filled"], rows)
    else:
        print("  (none)")

    print(f"\n{SEP}\n")

if __name__ == "__main__":
    main()
