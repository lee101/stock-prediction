# neuraldailyv3timed - Explicit Time-Based Exit Neural Trading

## Problem with V2

The current v2 system has no explicit exit timing:
- Model outputs: buy_price, sell_price, trade_amount, confidence
- Exit watchers wait indefinitely for take-profit price
- Positions held for 30+ days when market moves against them
- No stop-loss, no time-based exit, no risk control

**Result**: BTC/ETH positions stuck underwater for weeks, no buying power for new trades.

## V3 Timed Solution

### Core Idea
Make exit timing an explicit, learned output. The model predicts:
1. Entry price (when to buy)
2. Exit price (take profit target)
3. **Exit days (1-10 days maximum hold time)**
4. Trade amount (position size)

At inference: position closes at EITHER take-profit OR exit_days deadline, whichever comes first.

### Key Principle: Training = Inference
The simulation during training MUST match real-world execution:
- If model says exit_days=3, simulation forces close on day 3
- Model learns the consequences of different hold durations
- No gap between training assumptions and live behavior

---

## Architecture Changes

### 1. Model Outputs (model.py)

**Current V2 (4 outputs):**
```python
# Output head: buy_price, sell_price, trade_amount, confidence
self.head = nn.Linear(config.hidden_dim, 4, bias=False)
```

**V3 Timed (5 outputs):**
```python
# Output head: buy_price, sell_price, trade_amount, confidence, exit_days
self.head = nn.Linear(config.hidden_dim, 5, bias=False)
```

**Exit days decoding:**
```python
# exit_days: sigmoid gives 0-1, scale to 1-10 days
exit_days_raw = torch.sigmoid(outputs["exit_days_logits"]).squeeze(-1)
exit_days = 1.0 + exit_days_raw * 9.0  # Range [1, 10] days
```

### 2. TradingPlan (runtime.py)

```python
@dataclass
class TradingPlan:
    symbol: str
    timestamp: pd.Timestamp
    buy_price: float
    sell_price: float
    trade_amount: float
    exit_days: float      # NEW: 1-10 days max hold
    exit_timestamp: pd.Timestamp  # NEW: computed absolute exit time
    confidence: float = 1.0
    asset_class: float = 0.0
```

### 3. Simulation (simulation.py) - MAJOR CHANGE

**New Simulation Approach: "Trade Episode" Model**

Instead of continuous inventory tracking, each prediction is a complete trade:
1. Entry attempt at buy_price
2. Hold for up to exit_days
3. Exit at take_profit OR forced exit at deadline

```python
def simulate_trade_episode(
    *,
    # OHLC data for T+0 through T+max_hold_days
    highs: torch.Tensor,      # (batch, max_hold_days)
    lows: torch.Tensor,       # (batch, max_hold_days)
    closes: torch.Tensor,     # (batch, max_hold_days)

    # Model predictions
    buy_price: torch.Tensor,   # (batch,) entry limit
    sell_price: torch.Tensor,  # (batch,) take profit
    exit_days: torch.Tensor,   # (batch,) max hold 1-10
    trade_amount: torch.Tensor, # (batch,) position size

    config: SimulationConfig,
    temperature: float = 0.0,
) -> TradeEpisodeResult:
    """
    Simulate a single trade episode:
    1. Day 0: Check if entry fills (low <= buy_price)
    2. Days 1 to exit_days: Check if take profit fills (high >= sell_price)
    3. Day exit_days: If no TP, force close at close price

    Returns PnL, whether TP was hit, actual hold time, etc.
    """
```

**Simulation Logic:**
```python
# Day 0: Entry
entry_filled = (lows[:, 0] <= buy_price)
entry_price = torch.where(entry_filled, buy_price, closes[:, 0])

# Days 1 to max_hold: Check take profit
for day in range(1, max_hold_days + 1):
    # Mask: only check days <= exit_days
    active = (day <= exit_days) & entry_filled & ~tp_hit

    # Take profit check
    day_tp_hit = highs[:, day] >= sell_price
    tp_hit = tp_hit | (active & day_tp_hit)
    exit_price = torch.where(tp_hit, sell_price, exit_price)
    actual_hold = torch.where(tp_hit, day, actual_hold)

# Forced exit for positions that didn't hit TP
forced_exit = entry_filled & ~tp_hit
exit_idx = exit_days.long().clamp(1, max_hold_days)
forced_exit_price = closes.gather(1, exit_idx.unsqueeze(1)).squeeze(1)
exit_price = torch.where(forced_exit, forced_exit_price, exit_price)
actual_hold = torch.where(forced_exit, exit_days, actual_hold)

# PnL calculation
gross_pnl = (exit_price - entry_price) * trade_amount
fees = (entry_price + exit_price) * trade_amount * config.maker_fee
net_pnl = gross_pnl - fees
```

### 4. Data Loading (data.py)

**Extended lookahead for simulation:**
```python
# Need sequence_length history + max_hold_days future
total_length = sequence_length + max_hold_days  # 256 + 10 = 266

# Features from [0, sequence_length)
# Simulation data from [sequence_length, sequence_length + max_hold_days)
```

### 5. Training Loop (trainer.py)

```python
# Forward pass
outputs = model(features)
actions = model.decode_actions(outputs, reference_close, chronos_high, chronos_low)

# Simulate trade episodes
result = simulate_trade_episode(
    highs=future_highs,      # Days T+1 to T+10
    lows=future_lows,
    closes=future_closes,
    buy_price=actions["buy_price"][:, -1],   # Last timestep prediction
    sell_price=actions["sell_price"][:, -1],
    exit_days=actions["exit_days"][:, -1],
    trade_amount=actions["trade_amount"][:, -1],
    config=sim_config,
    temperature=temperature,
)

# Loss: maximize risk-adjusted return
loss = -result.sharpe  # or sortino, or custom objective
```

### 6. Config Changes (config.py)

```python
@dataclass
class SimulationConfigV3:
    max_hold_days: int = 10           # Maximum position hold time
    min_hold_days: int = 1            # Minimum (always at least 1 day)
    forced_exit_slippage: float = 0.001  # 10bps slippage on forced exits
    maker_fee: float = 0.0008
    # ... rest same as v2
```

---

## Inference / Live Trading

### Position Opening
```python
# When model says "buy BTCUSD":
plan = TradingPlan(
    symbol="BTCUSD",
    buy_price=92000,
    sell_price=94000,
    exit_days=3.5,
    exit_timestamp=now + timedelta(days=3.5),  # Computed deadline
    trade_amount=0.5,
)

# Save exit_timestamp to position state
position_db.save_exit_deadline(symbol, exit_timestamp)
```

### Exit Enforcement
New daemon/cron that checks deadlines:
```python
# Every minute:
for position in get_open_positions():
    if now >= position.exit_deadline:
        # Force close at market
        close_position_at_market(position.symbol)
        log.info(f"Forced exit {position.symbol} - deadline reached")
```

### Exit Watcher Updates
Modify maxdiff_cli.py to check BOTH:
1. Take profit price hit
2. Exit deadline reached

```python
while True:
    now = datetime.now()

    # Check deadline first
    if now >= exit_deadline:
        logger.info(f"Exit deadline reached for {symbol}")
        close_at_market(symbol)
        break

    # Check take profit
    if current_price >= takeprofit_price:
        close_at_takeprofit(symbol, takeprofit_price)
        break

    time.sleep(poll_seconds)
```

---

## Training Run

### Data Requirements
- Need 10 extra days of lookahead per sample
- Modify DataLoader to provide future OHLC for simulation

### Expected Behavior
Model learns to:
- Set tight exit_days (1-2) for high volatility / uncertain trades
- Set longer exit_days (7-10) for high-confidence trends
- Balance take-profit probability vs forced-exit risk

### Loss Function
```python
def compute_loss(result: TradeEpisodeResult) -> torch.Tensor:
    # Maximize expected return
    return_loss = -result.net_pnl.mean()

    # Penalize trades that hit forced exit (didn't reach TP)
    forced_exit_penalty = 0.1 * result.forced_exit_rate

    # Risk adjustment (variance of returns)
    risk_penalty = 0.05 * result.net_pnl.std()

    return return_loss + forced_exit_penalty + risk_penalty
```

---

## Directory Structure

```
neuraldailyv3timed/
├── __init__.py
├── config.py          # V3 configs with max_hold_days
├── model.py           # 5-output model (add exit_days)
├── simulation.py      # Trade episode simulation
├── data.py            # Extended lookahead data loading
├── trainer.py         # Training loop
├── runtime.py         # TradingPlan with exit_timestamp
└── checkpoints/
```

---

## Implementation Steps

1. **Copy v2 to v3timed**
   ```bash
   cp -r neuraldailyv2 neuraldailyv3timed
   ```

2. **Modify config.py**
   - Add max_hold_days, min_hold_days
   - Add forced_exit_slippage

3. **Modify model.py**
   - Add exit_days output head (5th output)
   - Add exit_days decoding logic

4. **Rewrite simulation.py**
   - Replace step-by-step with trade episode model
   - Handle entry fill, TP check, forced exit

5. **Modify data.py**
   - Extend samples to include 10-day lookahead
   - Provide future OHLC for simulation

6. **Modify trainer.py**
   - Use new simulation function
   - Adjust loss to reward proper exit timing

7. **Modify runtime.py**
   - Add exit_days, exit_timestamp to TradingPlan

8. **Create neural_trade_stock_live_v3timed.py**
   - New live trading script
   - Tracks exit deadlines per position
   - Enforces time-based exits

9. **Modify maxdiff_cli.py (or create v3 version)**
   - Add deadline checking to exit watchers

---

## Expected Outcomes

1. **No more stuck positions**: Every trade has a deadline
2. **Learnable risk control**: Model learns optimal hold durations
3. **Consistent training/inference**: Same rules in simulation and live
4. **Better capital efficiency**: Positions close, freeing buying power
5. **Reduced losses**: Forced exits prevent unlimited downside

---

## Questions to Resolve

1. Should exit_days be discrete (1,2,3...10) or continuous?
   - Continuous is more differentiable for training
   - Could round to nearest day at inference

2. How to handle partial fills at forced exit?
   - Use market orders with slippage assumption

3. Should the model also learn stop-loss price?
   - Could add as 6th output later
   - Start simple with time-based exits only

4. How to handle overnight/weekend gaps?
   - Count calendar days or trading days?
   - Probably trading days for stocks, calendar for crypto
