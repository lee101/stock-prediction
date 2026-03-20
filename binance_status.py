#!/usr/bin/env python3
"""Single command to print full Binance account status."""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.binan import binance_wrapper as bw
from src.binan.binance_margin import (
    get_all_margin_orders,
    get_margin_account,
    get_open_margin_orders,
)

SEP = "=" * 72
STABLE_ASSETS = {"USDT", "FDUSD", "BUSD", "USDC"}
SPOT_RECENT_ORDER_SYMBOLS = ["BTCFDUSD", "ETHFDUSD", "SOLFDUSD", "BTCU", "ETHU", "SOLU"]

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


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _asset_price_usdt(asset: str) -> float | None:
    asset = str(asset or "").upper()
    if not asset:
        return None
    if asset in {"USDT", "FDUSD"}:
        return 1.0
    direct_pairs = [f"{asset}USDT", f"{asset}FDUSD", f"{asset}BUSD"]
    inverse_pairs = [f"USDT{asset}", f"FDUSD{asset}"]
    for pair in direct_pairs:
        try:
            price = _safe_float(bw.get_symbol_price(pair), default=0.0)
        except Exception:
            continue
        if price > 0:
            return price
    for pair in inverse_pairs:
        try:
            price = _safe_float(bw.get_symbol_price(pair), default=0.0)
        except Exception:
            continue
        if price > 0:
            return 1.0 / price
    return None


def _load_margin_rows():
    account = get_margin_account()
    total_net_btc = _safe_float(account.get("totalNetAssetOfBtc"), default=0.0)
    btc_price = _safe_float(_asset_price_usdt("BTC"), default=0.0)
    total_net_usdt = total_net_btc * btc_price if btc_price > 0 else None

    rows = []
    for entry in account.get("userAssets", []):
        asset = str(entry.get("asset", "") or "").upper()
        free = _safe_float(entry.get("free"))
        net = _safe_float(entry.get("netAsset"))
        borrowed = _safe_float(entry.get("borrowed"))
        interest = _safe_float(entry.get("interest"))
        if max(abs(free), abs(net), abs(borrowed), abs(interest)) < 1e-8:
            continue
        price = _asset_price_usdt(asset)
        est_value = net * price if price is not None else None
        rows.append(
            {
                "asset": asset,
                "free": free,
                "net": net,
                "borrowed": borrowed,
                "interest": interest,
                "value_usdt": est_value,
            }
        )
    rows.sort(key=lambda item: abs(item["value_usdt"] or 0.0), reverse=True)
    return rows, total_net_usdt


def _normalize_margin_order_symbol(symbol: str) -> str | None:
    value = str(symbol or "").upper().strip()
    if not value:
        return None
    if value.endswith(("USDT", "FDUSD", "BUSD")):
        return value
    if value.endswith("USD") and len(value) > 3:
        return f"{value[:-3]}USDT"
    return None


def _load_recent_spot_orders():
    all_orders = []
    for sym in SPOT_RECENT_ORDER_SYMBOLS:
        try:
            ords = bw.get_all_orders(sym)
        except Exception:
            continue
        all_orders.extend(ords[-10:])
    all_orders.sort(key=lambda row: int(row.get("time", 0) or 0), reverse=True)
    return all_orders


def _margin_recent_order_symbols(margin_rows, pnl_state, open_margin_orders):
    symbols = set()
    for order in open_margin_orders:
        normalized = _normalize_margin_order_symbol(order.get("symbol", ""))
        if normalized:
            symbols.add(normalized)

    for row in margin_rows:
        asset = str(row.get("asset", "") or "").upper()
        if not asset or asset in STABLE_ASSETS:
            continue
        symbols.add(f"{asset}USDT")

    if isinstance(pnl_state, dict):
        for raw_symbol in pnl_state:
            normalized = _normalize_margin_order_symbol(raw_symbol)
            if normalized:
                symbols.add(normalized)

    return sorted(symbols)


def _load_recent_margin_orders(symbols):
    rows = []
    for symbol in symbols:
        try:
            rows.extend(get_all_margin_orders(symbol, limit=10))
        except Exception:
            continue
    rows.sort(key=lambda row: int(row.get("time", 0) or 0), reverse=True)
    return rows


def main():
    print(f"\n{SEP}")
    print(f"  BINANCE ACCOUNT STATUS  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(SEP)

    # spot balances
    val = bw.get_account_value_usdt()
    print(f"\n-- SPOT BALANCES (total: {fmt_usd(val['total_usdt'])}) --")
    rows = []
    for a in sorted(val["assets"], key=lambda x: -x["value_usdt"]):
        if a["value_usdt"] < 0.01:
            continue
        rows.append((a["asset"], f"{a['amount']:.6f}", fmt_usd(a["value_usdt"])))
    if rows:
        print_table(["Asset", "Amount", "Value"], rows, ["<", ">", ">"])
    else:
        print("  (none)")

    # cross-margin balances
    try:
        margin_rows, margin_total = _load_margin_rows()
    except Exception as exc:
        margin_rows, margin_total = [], None
        print(f"\n-- CROSS MARGIN: error loading account ({exc}) --")
    else:
        total_label = fmt_usd(margin_total) if margin_total is not None else "unknown"
        print(f"\n-- CROSS MARGIN (net approx: {total_label}) --")
        if margin_rows:
            table_rows = []
            for row in margin_rows:
                value = fmt_usd(row["value_usdt"]) if row["value_usdt"] is not None else "?"
                table_rows.append(
                    (
                        row["asset"],
                        f"{row['free']:.8f}",
                        f"{row['net']:.8f}",
                        f"{row['borrowed']:.8f}",
                        value,
                    )
                )
            print_table(["Asset", "Free", "Net", "Borrowed", "Est Value"], table_rows, ["<", ">", ">", ">", ">"])
        else:
            print("  (none)")

    combined_total = float(val["total_usdt"]) + float(margin_total or 0.0)
    print(f"\n-- COMBINED APPROX TOTAL: {fmt_usd(combined_total)} --")

    # realized trading pnl from state file
    pnl_path = Path(__file__).resolve().parent.parent / "strategy_state" / "binanceneural_pnl_state_live.json"
    pnl_state = None
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
    spot_orders = bw.get_open_orders()
    margin_orders = get_open_margin_orders()
    combined_open = len(spot_orders) + len(margin_orders)
    print(f"\n-- OPEN ORDERS (combined: {combined_open}, spot: {len(spot_orders)}, margin: {len(margin_orders)}) --")
    if spot_orders:
        print("Spot:")
        rows = []
        for o in spot_orders:
            rows.append((o["symbol"], o["side"], o["origQty"], o["price"], o["status"]))
        print_table(["Pair", "Side", "Qty", "Price", "Status"], rows)
    if margin_orders:
        print("Margin:")
        rows = []
        for o in margin_orders:
            rows.append((
                fmt_ts(o["time"]),
                o["symbol"],
                o["side"],
                o["origQty"],
                o["price"],
                o["status"],
                o.get("executedQty", "?"),
            ))
        print_table(["Time", "Pair", "Side", "Qty", "Price", "Status", "Filled"], rows)
    if not spot_orders and not margin_orders:
        print("  (none)")

    # recent spot orders
    print(f"\n-- RECENT SPOT ORDERS --")
    all_orders = _load_recent_spot_orders()
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

    # recent margin orders
    print(f"\n-- RECENT MARGIN ORDERS --")
    margin_symbols = _margin_recent_order_symbols(margin_rows, pnl_state, margin_orders)
    margin_history = _load_recent_margin_orders(margin_symbols)
    rows = []
    for o in margin_history[:15]:
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
