import pandas as pd
from pathlib import Path


def load_daily_bars(symbol: str, data_dir: Path) -> pd.DataFrame:
    for suffix in ["USD", "USDT"]:
        p = data_dir / f"{symbol}{suffix}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)
    return pd.DataFrame()


def load_forecast(symbol: str, forecast_dir: Path, as_of: pd.Timestamp) -> dict:
    for suffix in ["USD", "USDT"]:
        p = forecast_dir / f"{symbol}{suffix}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            mask = df["timestamp"] <= as_of
            if mask.any():
                row = df.loc[mask].iloc[-1]
                return {
                    "close_p50": float(row.get("predicted_close_p50", 0)),
                    "close_p10": float(row.get("predicted_close_p10", 0)),
                    "close_p90": float(row.get("predicted_close_p90", 0)),
                    "high_p50": float(row.get("predicted_high_p50", 0)),
                    "low_p50": float(row.get("predicted_low_p50", 0)),
                }
    return {}


def build_price_table(symbols: list, data_dir: Path, as_of: pd.Timestamp,
                      lookback: int = 7) -> str:
    lines = ["Symbol | Close | 7d_Chg% | 7d_High | 7d_Low | Volume"]
    lines.append("---|---|---|---|---|---")
    for sym in symbols:
        df = load_daily_bars(sym, data_dir)
        if df.empty:
            continue
        df_ts = df[df["timestamp"] <= as_of]
        if len(df_ts) < 2:
            continue
        recent = df_ts.tail(lookback)
        last = recent.iloc[-1]
        first = recent.iloc[0]
        chg = (last["close"] / first["close"] - 1) * 100
        lines.append(
            f"{sym} | {last['close']:.4g} | {chg:+.1f}% | "
            f"{recent['high'].max():.4g} | {recent['low'].min():.4g} | "
            f"{recent['volume'].sum():.0f}"
        )
    return "\n".join(lines)


def build_forecast_table(symbols: list, forecast_dir: Path,
                         as_of: pd.Timestamp) -> str:
    lines = ["Symbol | Forecast_Close | Forecast_Low | Forecast_High | Range%"]
    lines.append("---|---|---|---|---")
    for sym in symbols:
        fc = load_forecast(sym, forecast_dir, as_of)
        if not fc or fc["close_p50"] == 0:
            continue
        rng = 0
        if fc["close_p50"] > 0:
            rng = (fc["close_p90"] - fc["close_p10"]) / fc["close_p50"] * 100
        lines.append(
            f"{sym} | {fc['close_p50']:.4g} | {fc['low_p50']:.4g} | "
            f"{fc['high_p50']:.4g} | {rng:.1f}%"
        )
    return "\n".join(lines)


def build_portfolio_table(positions: dict, current_prices: dict) -> str:
    if not positions:
        return "No current positions."
    lines = ["Symbol | Qty | Entry | Current | PnL%"]
    lines.append("---|---|---|---|---")
    for sym, pos in positions.items():
        cur = current_prices.get(sym, pos["entry_price"])
        pnl = (cur / pos["entry_price"] - 1) * 100
        lines.append(
            f"{sym} | {pos['qty']:.6g} | {pos['entry_price']:.4g} | "
            f"{cur:.4g} | {pnl:+.2f}%"
        )
    return "\n".join(lines)


def build_prompt(symbols: list, data_dir: Path, forecast_dir: Path,
                 as_of: pd.Timestamp, positions: dict,
                 current_prices: dict, capital: float,
                 leverage_limit: float, max_positions: int) -> str:
    price_table = build_price_table(symbols, data_dir, as_of)
    forecast_table = build_forecast_table(symbols, forecast_dir, as_of)
    portfolio_table = build_portfolio_table(positions, current_prices)

    return f"""You are a cryptocurrency portfolio manager on Binance.
Date: {as_of.strftime('%Y-%m-%d')}
Available capital: ${capital:.2f} USDT
Max leverage: {leverage_limit}x (you may use up to {leverage_limit}x total account value)
Max positions: {max_positions}

CURRENT PRICES (last 7 days):
{price_table}

CHRONOS2 24-HOUR FORECASTS (statistical model predictions):
{forecast_table}

CURRENT PORTFOLIO:
{portfolio_table}

TASK: Decide the optimal allocation of capital across these cryptocurrencies for the next 24 hours.
For each position you want, specify:
- allocation_pct: what % of total capital (including leverage) to allocate (all must sum to <=100)
- direction: "long" or "short"
- exit_price: your target exit price within 24h
- stop_price: your stop-loss price

You may hold USDT (cash) by not allocating 100%.
Think about momentum, forecast direction, risk/reward, and diversification.
Aim to maximize risk-adjusted returns (Sortino ratio).

Respond with ONLY a JSON object like:
```json
{{
  "allocations": {{
    "BTC": {{"allocation_pct": 30, "direction": "long", "exit_price": 70000, "stop_price": 65000}},
    "ETH": {{"allocation_pct": 20, "direction": "long", "exit_price": 3500, "stop_price": 3200}}
  }},
  "reasoning": "brief explanation"
}}
```"""
