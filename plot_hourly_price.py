#!/usr/bin/env python3
"""Plot hourly price data for crypto symbols."""

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from alpaca.data import TimeFrame, TimeFrameUnit

from alpaca_wrapper import download_symbol_history


def plot_hourly_price(
    symbol: str = "BTCUSD",
    hours: int = 48,
    output_path: str = None,
    include_latest: bool = True,
):
    """
    Plot hourly price data for a symbol.

    Args:
        symbol: Trading symbol (e.g., BTCUSD, ETHUSD)
        hours: Number of hours of historical data to fetch
        output_path: Path to save PNG (default: plots/{symbol}_hourly.png)
        include_latest: Whether to include latest quote data in the chart
    """
    # Fetch data
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    print(f"Fetching {hours} hours of data for {symbol}...")
    df = download_symbol_history(
        symbol=symbol,
        start=start,
        end=end,
        include_latest=include_latest,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour)
    )

    if df.empty:
        print(f"No data found for {symbol}")
        return

    print(f"Retrieved {len(df)} bars")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Plot OHLC as candlesticks
    times = df.index
    opens = df['open']
    highs = df['high']
    lows = df['low']
    closes = df['close']

    # Determine colors for candlesticks
    colors = ['g' if closes.iloc[i] >= opens.iloc[i] else 'r'
              for i in range(len(df))]

    # Plot candlesticks
    for i in range(len(df)):
        # Plot high-low line
        ax1.plot([times[i], times[i]], [lows.iloc[i], highs.iloc[i]],
                color='black', linewidth=0.5, alpha=0.7)

        # Plot open-close body
        body_height = abs(closes.iloc[i] - opens.iloc[i])
        body_bottom = min(opens.iloc[i], closes.iloc[i])

        ax1.bar(times[i], body_height, bottom=body_bottom,
               width=timedelta(minutes=40), color=colors[i], alpha=0.7,
               edgecolor='black', linewidth=0.5)

    # Add close price line
    ax1.plot(times, closes, color='blue', linewidth=1.5, alpha=0.5, label='Close Price')

    # Format price axis
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} Hourly Price Chart (Last {hours} Hours)',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Format x-axis for top chart
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=45)

    # Plot volume
    if 'volume' in df.columns:
        volumes = df['volume']
        vol_colors = ['g' if closes.iloc[i] >= opens.iloc[i] else 'r'
                     for i in range(len(df))]

        ax2.bar(times, volumes, width=timedelta(minutes=40),
               color=vol_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.tick_params(axis='x', rotation=45)

    # Add current price annotation
    latest_price = closes.iloc[-1]
    latest_time = times[-1]
    ax1.annotate(f'Current: ${latest_price:,.2f}',
                xy=(latest_time, latest_price),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Tight layout
    plt.tight_layout()

    # Save figure
    if output_path is None:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        output_path = plots_dir / f"{symbol.lower()}_hourly.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")

    # Print summary stats
    print(f"\nPrice Summary:")
    print(f"  Latest: ${latest_price:,.2f}")
    print(f"  High:   ${highs.max():,.2f}")
    print(f"  Low:    ${lows.min():,.2f}")
    print(f"  Change: ${closes.iloc[-1] - closes.iloc[0]:,.2f} "
          f"({((closes.iloc[-1] / closes.iloc[0]) - 1) * 100:+.2f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot hourly price data for crypto symbols"
    )
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default="BTCUSD",
        help="Trading symbol (default: BTCUSD)"
    )
    parser.add_argument(
        "--hours", "-H",
        type=int,
        default=48,
        help="Number of hours of data to plot (default: 48)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output PNG path (default: plots/{symbol}_hourly.png)"
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Don't include latest quote data"
    )

    args = parser.parse_args()

    plot_hourly_price(
        symbol=args.symbol,
        hours=args.hours,
        output_path=args.output,
        include_latest=not args.no_latest
    )


if __name__ == "__main__":
    main()
