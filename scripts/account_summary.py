import pytz
from datetime import datetime, timedelta
from loguru import logger
from alpaca_wrapper import get_account_activities, alpaca_api, get_all_positions

def analyze_trading_history():
    """
    A simple Python-based realized P&L calculation for closed trades 
    plus unrealized P&L for currently open positions.
    """

    # 1) Fetch historical FILLs, DIVs, INTs for realized P&L
    activities = get_account_activities(
        alpaca_api, 
        activity_types=['FILL', 'DIV', 'INT'], 
        direction='desc'
    )

    if not activities or len(activities) == 0:
        logger.warning("No trading activities found")
    else:
        # Convert to standardized records plus timestamp
        sorted_activities = []
        for act in activities:
            if act['activity_type'] in ('FILL'):
                stamp = act.get('transaction_time')
            else:  # DIV / INT
                stamp = act.get('date')

            try:
                # Convert from ISO8601 to Python datetime
                stamp_dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except Exception:
                logger.error(f"Could not parse timestamp for activity: {act}")
                continue

            sorted_activities.append({
                'activity_type': act['activity_type'],
                'symbol': act.get('symbol', 'N/A'),
                'side': act.get('side', None),
                'qty': float(act.get('qty', 0.0)),
                'price': float(act.get('price', 0.0)),
                'net_amount': float(act.get('net_amount', 0.0)),
                'timestamp': stamp_dt,
            })

        # Sort ascending by time
        sorted_activities.sort(key=lambda x: x['timestamp'])

        # Track realized P&L
        positions = {}   # symbol => { 'qty': float, 'cost_basis': float }
        pnl_events = []  # each realized event
        symbol_trades = {}  # symbol => { 'total_buy_cost': float, 'realized_pnl': float, 'trade_count': int }
        cumulative_pnl = 0.0

        for act in sorted_activities:
            sym = act['symbol']
            typ = act['activity_type']
            dt = act['timestamp']

            if sym not in positions:
                positions[sym] = {'qty': 0.0, 'cost_basis': 0.0}
            if sym not in symbol_trades:
                symbol_trades[sym] = {
                    'total_buy_cost': 0.0,
                    'realized_pnl': 0.0,
                    'trade_count': 0
                }

            if typ == 'FILL':
                side = act.get('side')
                qty  = act.get('qty', 0.0)
                price = act.get('price', 0.0)

                symbol_trades[sym]['trade_count'] += 1

                if side == 'buy':
                    old_qty = positions[sym]['qty']
                    old_cb  = positions[sym]['cost_basis']
                    new_qty = old_qty + qty
                    if new_qty > 0:
                        new_cb = (old_cb*old_qty + price*qty) / new_qty
                    else:
                        new_cb = price
                    positions[sym]['qty'] = new_qty
                    positions[sym]['cost_basis'] = new_cb

                elif side == 'sell':
                    old_qty = positions[sym]['qty']
                    old_cb  = positions[sym]['cost_basis']
                    if old_qty > 0:
                        shares_sold = min(old_qty, qty)
                        cost_of_shares = old_cb * shares_sold
                        realized = (price - old_cb) * shares_sold

                        symbol_trades[sym]['total_buy_cost']  += cost_of_shares
                        symbol_trades[sym]['realized_pnl']    += realized

                        cumulative_pnl += realized
                        positions[sym]['qty'] = old_qty - shares_sold
                        if positions[sym]['qty'] == 0:
                            positions[sym]['cost_basis'] = 0.0

                        # Store event
                        pnl_events.append({
                            'timestamp': dt,
                            'symbol': sym,
                            'pnl': realized,
                            'cost_basis': cost_of_shares,
                            'type': 'REALIZED_SELL'
                        })

            elif typ in ('DIV','INT'):
                div_int_gain = act['net_amount']
                cumulative_pnl += div_int_gain
                symbol_trades[sym]['realized_pnl'] += div_int_gain

                pnl_events.append({
                    'timestamp': dt,
                    'symbol': sym,
                    'pnl': div_int_gain,
                    'cost_basis': 0.0,
                    'type': typ
                })

        print("\n=== All-Time Realized P&L Summary ===")
        print(f"Total Realized P&L: ${cumulative_pnl:.2f}")

        # Sort P&L events by time and compute a running total
        pnl_events.sort(key=lambda x: x['timestamp'])
        running = 0
        for evt in pnl_events:
            running += evt['pnl']
            evt['cumulative'] = running

        # Last 7 days realized
        one_week_ago = datetime.now(pytz.UTC) - timedelta(days=7)
        last_week_pnl  = sum(e['pnl'] for e in pnl_events if e['timestamp'] >= one_week_ago)
        print("\n=== Last 7 Days Realized P&L ===")
        print(f"Recent Realized P&L: ${last_week_pnl:.2f}")

        # Show P&L by symbol
        sorted_syms = sorted(symbol_trades.items(), key=lambda x: x[1]['realized_pnl'], reverse=True)
        print("\n=== Realized P&L By Symbol (All-Time) ===")
        for sym, data in sorted_syms:
            pnl  = data['realized_pnl']
            cost = data['total_buy_cost']
            tcnt = data['trade_count']
            if cost > 0:
                pct = (pnl / cost)*100
                print(f"{sym}: ${pnl:.2f} ({pct:.2f}% on ${cost:.2f}) [{tcnt} trades]")
            else:
                # Possibly no sells yet => cost=0 or just dividends
                print(f"{sym}: ${pnl:.2f} (N/A% - no sells) [{tcnt} trades]")

        # Weekly by symbol
        weekly_stats = {}
        for evt in pnl_events:
            if evt['timestamp'] >= one_week_ago:
                s = evt['symbol']
                if s not in weekly_stats:
                    weekly_stats[s] = {'pnl': 0.0, 'cost': 0.0, 'count': 0}
                weekly_stats[s]['pnl']  += evt['pnl']
                if evt['type'] == 'REALIZED_SELL':
                    weekly_stats[s]['cost'] += evt['cost_basis']
                    weekly_stats[s]['count'] += 1

        print("\n=== Last 7 Days Realized P&L By Symbol ===")
        if not weekly_stats:
            print("No realized trades/income in the last 7 days.")
        else:
            # Sort by realized P&L
            for sym, vals in sorted(weekly_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                p = vals['pnl']
                c = vals['cost']
                ct = vals['count']
                if c > 0:
                    pct = (p / c)*100
                    print(f"{sym}: ${p:.2f} ({pct:.2f}% on ${c:.2f}) [{ct} sells]")
                else:
                    print(f"{sym}: ${p:.2f} (N/A% - no sells) [{ct} sells]")

    # 2) Now integrate open positions for unrealized P&L
    print("\n=== Open Positions (Unrealized) ===")
    try:
        open_positions = get_all_positions()
        if not open_positions:
            print("No open positions.")
            return
        for pos in open_positions:
            # Each position might have fields like:
            # pos.symbol, pos.qty, pos.avg_entry_price, pos.unrealized_pl
            sym = pos.symbol
            qty = float(pos.qty)
            avg_cost = float(pos.avg_entry_price)
            upl = float(pos.unrealized_pl) if pos.unrealized_pl else (0.0)
            
            # If you want to compute manually:
            # current_price = float(pos.current_price)
            # upl_manual = (current_price - avg_cost) * qty

            print(f"{sym}: {qty} shares, avg cost ${avg_cost:.2f}, unrealized P&L ${upl:.2f}")

    except Exception as ex:
        logger.error(f"Error fetching open positions: {ex}")


if __name__ == "__main__":
    analyze_trading_history()