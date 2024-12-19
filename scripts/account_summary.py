import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from alpaca_wrapper import get_account_activities, alpaca_api
from loguru import logger
import pytz

def analyze_trading_history():
    """
    Analyzes historical executed orders and account activities to compute realized P&L
    and generate visualizations (both $ gains and % gains).
    """
    try:
        # Get all trade activities (FILL = executed trades, DIV/INT = income)
        activities = get_account_activities(
            alpaca_api, 
            activity_types=['FILL', 'DIV', 'INT'],
            direction='desc'
        )
        
        if not activities:
            logger.warning("No trading activities found")
            return
            
        # Convert activities to DataFrame
        activities_data = []
        for activity in activities:
            if activity['activity_type'] == 'FILL':
                activity_data = {
                    'symbol': activity['symbol'],
                    'side': activity['side'],
                    'filled_qty': float(activity['qty']),
                    'filled_avg_price': float(activity['price']),
                    'timestamp': activity['transaction_time'],
                    'type': 'FILL',
                    'total_value': float(activity['qty']) * float(activity['price'])
                }
            elif activity['activity_type'] in ['DIV', 'INT']:
                # Dividends / interest
                activity_data = {
                    'symbol': activity.get('symbol', 'N/A'),
                    'side': 'dividend' if activity['activity_type'] == 'DIV' else 'interest',
                    'filled_qty': float(activity.get('qty', 0)),
                    'filled_avg_price': float(activity.get('per_share_amount', 0)),
                    'timestamp': activity['date'],
                    'type': activity['activity_type'],
                    'total_value': float(activity['net_amount'])
                }
            else:
                continue
            activities_data.append(activity_data)

        df = pd.DataFrame(activities_data)
        if df.empty:
            print("No matching activities.")
            return

        # Make timestamps UTC-aware and sort by time ascending
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # We'll compute realized P&L from buys & sells. Weighted-average cost basis.
        # Also track cost_of_sold_shares: cost basis * shares actually sold
        df['realized_pnl'] = 0.0
        df['cost_of_sold_shares'] = 0.0

        # Store { symbol: { 'qty': current_shares, 'cost_basis': avg_cost_per_share } }
        positions = {}

        for i, row in df.iterrows():
            if row['type'] == 'FILL':
                symbol = row['symbol']
                side = row['side']
                qty = row['filled_qty']
                price = row['filled_avg_price']

                if symbol not in positions:
                    positions[symbol] = {'qty': 0.0, 'cost_basis': 0.0}

                old_qty = positions[symbol]['qty']
                old_cb = positions[symbol]['cost_basis']

                if side == 'buy':
                    # Weighted-average cost update
                    new_qty = old_qty + qty
                    if new_qty > 0:
                        new_cb = (old_cb * old_qty + price * qty) / new_qty
                    else:
                        # Normally shouldn't happen for a buy
                        new_cb = price
                    positions[symbol]['qty'] = new_qty
                    positions[symbol]['cost_basis'] = new_cb

                elif side == 'sell':
                    # Only realize P&L if we have a positive qty (long shares)
                    if old_qty > 0:
                        # If we're selling more than we hold, only compute partial
                        shares_sold = min(qty, old_qty)
                        cost_of_shares_sold = old_cb * shares_sold
                        realized = (price - old_cb) * shares_sold

                        df.at[i, 'realized_pnl'] = realized
                        df.at[i, 'cost_of_sold_shares'] = cost_of_shares_sold

                        positions[symbol]['qty'] = old_qty - shares_sold
                        # cost basis remains the same unless qty is now zero
                        if positions[symbol]['qty'] <= 0:
                            positions[symbol]['qty'] = 0.0

                    # If old_qty == 0, we might be shorting - not handled here
                    # so no realized P&L in that scenario.

            elif row['type'] in ['DIV', 'INT']:
                # Direct gains from dividends/interest
                df.at[i, 'realized_pnl'] = row['total_value']

        # Cumulative realized P&L
        df['cumulative_pnl'] = df['realized_pnl'].cumsum()

        # Save to CSV
        csv_filename = f"trading_history_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(csv_filename, index=False)
        logger.info(f"Trading history saved to {csv_filename}")

        # Plot the realized P&L over time
        plt.figure(figsize=(12,6))
        plt.plot(df['timestamp'], df['cumulative_pnl'], label='Cumulative Realized P&L')
        plt.title('Trading P&L Over Time')
        plt.xlabel('Date')
        plt.ylabel('Realized P&L ($)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = f"pnl_chart_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_filename, bbox_inches='tight')
        logger.info(f"P&L chart saved to {plot_filename}")

        # Overall summary
        print("\n=== All-Time Trading Summary ===")
        print(f"Total Activities: {len(df)}")
        print(f"Total Realized P&L: ${df['realized_pnl'].sum():.2f}")

        # Last 7 days analysis
        one_week_ago = datetime.now(pytz.UTC) - timedelta(days=7)
        week_df = df[df['timestamp'] >= one_week_ago]

        print("\n=== Last 7 Days Summary ===")
        print(f"Recent Activities: {len(week_df)}")
        print(f"Week's Realized P&L: ${week_df['realized_pnl'].sum():.2f}")

        # All-time trades only
        trades_df = df[df['type'] == 'FILL']
        if not trades_df.empty:
            print(f"\n=== All-Time Trade Statistics ===")
            print(f"Total Trades: {len(trades_df)}")
            if 'total_value' in trades_df.columns:
                print(f"Average Trade Value: ${trades_df['total_value'].mean():.2f}")

            print(f"Most Traded Symbol: {trades_df['symbol'].value_counts().index[0]}")

            # Summaries by symbol
            # - Summation of realized_pnl
            # - Summation of cost_of_sold_shares
            # Then compute percentage = (realized_pnl / cost_of_sold_shares)*100
            sum_cols = trades_df.groupby('symbol')[['realized_pnl','cost_of_sold_shares']].sum()
            sum_cols['pct_gain'] = 0.0
            mask_nonzero_cost = (sum_cols['cost_of_sold_shares'] != 0)
            sum_cols.loc[mask_nonzero_cost, 'pct_gain'] = (
                sum_cols.loc[mask_nonzero_cost, 'realized_pnl'] 
                / sum_cols.loc[mask_nonzero_cost, 'cost_of_sold_shares'] 
                * 100
            )
            # Sort by realized_pnl descending
            sum_cols = sum_cols.sort_values('realized_pnl', ascending=False)

            print("\nRealized P&L By Symbol ($ and %):")
            for symbol, rowvals in sum_cols.iterrows():
                # # trades for symbol
                trades_count = len(trades_df[trades_df['symbol'] == symbol])
                print(f"{symbol}: ${rowvals['realized_pnl']:.2f} "
                      f"({rowvals['pct_gain']:.2f}% return) "
                      f"({trades_count} trades)")

            # Last week trades
            week_trades_df = trades_df[trades_df['timestamp'] >= one_week_ago]
            if not week_trades_df.empty:
                print(f"\n=== Last 7 Days Trade Statistics ===")
                print(f"Recent Trades: {len(week_trades_df)}")
                if 'total_value' in week_trades_df.columns:
                    print(f"Week's Average Trade Value: ${week_trades_df['total_value'].mean():.2f}")

                # Same approach for the last 7 days
                week_sum_cols = week_trades_df.groupby('symbol')[['realized_pnl','cost_of_sold_shares']].sum()
                week_sum_cols['pct_gain'] = 0.0
                mask_nonzero_cost = (week_sum_cols['cost_of_sold_shares'] != 0)
                week_sum_cols.loc[mask_nonzero_cost, 'pct_gain'] = (
                    week_sum_cols.loc[mask_nonzero_cost, 'realized_pnl'] 
                    / week_sum_cols.loc[mask_nonzero_cost, 'cost_of_sold_shares'] 
                    * 100
                )
                week_sum_cols = week_sum_cols.sort_values('realized_pnl', ascending=False)

                print("\nRecent Realized P&L By Symbol ($ and %):")
                for symbol, rowvals in week_sum_cols.iterrows():
                    trades_count = len(week_trades_df[week_trades_df['symbol'] == symbol])
                    print(f"{symbol}: ${rowvals['realized_pnl']:.2f} "
                          f"({rowvals['pct_gain']:.2f}% return) "
                          f"({trades_count} trades)")

        # Dividend & Interest
        div_df = df[df['type'] == 'DIV']
        if not div_df.empty:
            print(f"\n=== Dividend Income ===")
            print(f"Total Dividend Income: ${div_df['realized_pnl'].sum():.2f}")
            div_by_symbol = div_df.groupby('symbol')['realized_pnl'].sum().sort_values(ascending=False)
            for symbol, amount in div_by_symbol.items():
                print(f"{symbol}: ${amount:.2f}")

            week_div_df = div_df[div_df['timestamp'] >= one_week_ago]
            if not week_div_df.empty:
                print(f"\nLast 7 Days Dividend Income: ${week_div_df['realized_pnl'].sum():.2f}")

        int_df = df[df['type'] == 'INT']
        if not int_df.empty:
            print(f"\n=== Interest Summary ===")
            print(f"Total Interest: ${int_df['realized_pnl'].sum():.2f}")

            week_int_df = int_df[int_df['timestamp'] >= one_week_ago]
            if not week_int_df.empty:
                print(f"Last 7 Days Interest: ${week_int_df['realized_pnl'].sum():.2f}")

    except Exception as e:
        logger.error(f"Error analyzing trading history: {e}")
        raise

if __name__ == "__main__":
    analyze_trading_history()
