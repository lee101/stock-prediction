# Hourly Trading Bot PnL Gate

## Overview

The hourly trading bot now includes a PnL-based gate mechanism that tracks trade outcomes and blocks trading on symbol+side+strategy combinations where recent trades have been unprofitable. This helps prevent repeatedly trading losing strategies.

## How It Works

### PnL Tracking

1. **Trade History Storage**: All completed trades are stored in `trade_history_hourly.json` (separate from daily bot's `trade_history.json`)
2. **Recent Trade Analysis**: Before entering a new trade, the system checks the last 1-2 trades for the same symbol, side, and strategy
3. **Blocking Logic**: If the sum of PnL from recent trades is negative, trading is blocked for that combination

### Key Features

- **Automatic Probe Behavior**: If there's no trade history for a symbol+side+strategy, trading is allowed (probe trade)
- **Single Trade Handling**: If only 1 previous trade exists, uses just that trade's PnL
- **Two Trade Window**: If 2+ trades exist, sums the last 2 trades' PnL
- **Strategy-Specific**: Blocking is per strategy (e.g., ETHUSD-buy-maxdiff can be blocked while ETHUSD-buy-highlow is allowed)
- **Data Isolation**: Hourly bot state is completely separate from daily bot via `TRADE_STATE_SUFFIX=hourly`

## Configuration

### Environment Variables

```bash
# Enable/disable PnL gate (default: enabled)
HOURLY_ENABLE_PNL_GATE=1

# Number of recent trades to consider (default: 2)
HOURLY_PNL_GATE_MAX_TRADES=2
```

### Disabling the Gate

To disable the PnL gate:
```bash
export HOURLY_ENABLE_PNL_GATE=0
```

## State File Isolation

The hourly and daily trading bots maintain completely separate state:

### Daily Bot Files
- `strategy_state/trade_history.json`
- `strategy_state/trade_outcomes.json`
- `strategy_state/trade_learning.json`
- `strategy_state/active_trades.json`

### Hourly Bot Files
- `strategy_state/trade_history_hourly.json`
- `strategy_state/trade_outcomes_hourly.json`
- `strategy_state/trade_learning_hourly.json`
- `strategy_state/active_trades_hourly.json`

This is achieved by setting `TRADE_STATE_SUFFIX=hourly` in `trade_stock_e2e_hourly.py:29`.

## Implementation Details

### Core Module: `src/hourly_pnl_gate.py`

Functions:
- `get_recent_trade_pnl()`: Retrieves recent trades and calculates total PnL
- `should_block_trade_by_pnl()`: Determines if trading should be blocked
- `get_pnl_blocking_report()`: Generates a report of blocked symbols (for debugging)

### Integration: `trade_stock_e2e_hourly.py`

The PnL gate is integrated into the trading cycle in `HourlyTradingEngine.run_cycle()`:

1. Build portfolio picks
2. Apply positive forecast filters
3. **Apply PnL gate** (new step)
4. Execute trades

Blocked symbols are logged with warnings:
```
Hourly PnL gate blocked BTCUSD: Last 2 trades for BTCUSD buy (maxdiff) had negative PnL: -15.23
```

## Example Scenarios

### Scenario 1: First Trade (Probe)
- **History**: None
- **Action**: Allow trade (probe)
- **Reason**: No data to base decision on

### Scenario 2: One Losing Trade
- **History**: Trade 1: -$10
- **PnL Sum**: -$10
- **Action**: Block trade
- **Reason**: Recent trade was unprofitable

### Scenario 3: Two Trades, Net Negative
- **History**: Trade 1: +$5, Trade 2: -$20
- **PnL Sum**: -$15
- **Action**: Block trade
- **Reason**: Recent trades sum to negative

### Scenario 4: Two Trades, Net Positive
- **History**: Trade 1: -$5, Trade 2: +$15
- **PnL Sum**: +$10
- **Action**: Allow trade
- **Reason**: Recent trades sum to positive

### Scenario 5: Old Losses, Recent Wins
- **History**: Trade 1: -$50, Trade 2: -$30, Trade 3: +$20, Trade 4: +$25
- **Check Last 2**: Trade 3: +$20, Trade 4: +$25
- **PnL Sum**: +$45
- **Action**: Allow trade
- **Reason**: Only last 2 trades are considered, which are profitable

## Testing

Comprehensive test suite in `tests/test_hourly_pnl_gate.py`:
- No history allows trading (probe scenario)
- Positive PnL allows trading
- Negative PnL blocks trading
- Mixed PnL with negative sum blocks
- Single negative trade blocks
- Only recent trades are considered
- Strategy-specific blocking
- State file isolation

Run tests:
```bash
source .venv/bin/activate
python -m pytest tests/test_hourly_pnl_gate.py -v
```

## Monitoring

Watch logs for PnL gate activity:
```bash
tail -f logs/trade_stock_e2e_hourly.log | grep "PnL gate"
```

Example output:
```
2025-01-13 10:30:00 WARNING Hourly PnL gate blocked BTCUSD: Last 2 trades for BTCUSD buy (maxdiff) had negative PnL: -15.23
2025-01-13 11:30:00 WARNING Hourly PnL gate blocked ETHUSD: Last 1 trade for ETHUSD sell had negative PnL: -8.50
```

## Benefits

1. **Risk Management**: Prevents repeated trading of losing strategies
2. **Capital Preservation**: Reduces losses from bad market conditions or poor strategy fit
3. **Automatic Recovery**: Once a strategy becomes profitable again, it's automatically unblocked
4. **Strategy-Specific**: Good strategies continue while bad ones are blocked
5. **Zero Interference**: Daily bot is completely unaffected

## Future Enhancements

Potential improvements:
- Configurable PnL threshold (e.g., block if < -$50, not just < $0)
- Time-based cooldowns (e.g., block for N hours after losses)
- Win rate tracking (e.g., block if win rate < 40%)
- Drawdown-based blocking (e.g., block if recent drawdown > -10%)
