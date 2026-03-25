# Alpaca Progress 6 — 2026-03-25

## Current Production State

### Running processes (GPU)
| PID | Process | Status |
|-----|---------|--------|
| 3178825 → 1161308 | `trade_unified_hourly_meta.py` | LIVE stocks, supervisor `unified-stock-trader` |
| 1157793 | `trade_daily_stock_prod.py --paper` | PAPER daily RL, systemd `daily-rl-trader` |
| hourlyv5-stocks.service | `trade_hourlyv5_multi.py --asset-class stocks --daemon` | PAPER stocks (old checkpoint) |
| 754212 | crypto15 s78 long retrain | Training (not prod) |
| 893475 | crypto70 s72 sweep | Training (not prod) |

### Live Alpaca positions (as of 2026-03-25 01:37 UTC)
| Symbol | Qty | Value | Unrealized | Notes |
|--------|-----|-------|------------|-------|
| ABEV | 4459 | $12,396 | +$222.95 | **force_close order placed 01:42 UTC, fills at market open** |
| AVAXUSD | ~0 | ~$0 | ~$0 | Dust from old trade, can't close |
| BTCUSD | ~0 | ~$0 | ~$0 | Dust |
| ETHUSD | 0.000000967 | ~$0.002 | ~$0 | Dust — this is the "no exit for ETH" the user saw |
| LTCUSD | ~0 | ~$0 | ~$0 | Dust |
| SOLUSD | ~0 | ~$0 | ~$0 | Dust |

### Paper Alpaca positions (orphaned from old hourlyv5 runs)
| Symbol | Qty | Value | Issue |
|--------|-----|-------|-------|
| ETHUSD | 1.58 | $3,424 | No exit manager (hourlyv5 now runs `--asset-class stocks`) |
| BTCUSD | 0.079 | $5,576 | No exit manager |
| UNIUSD | 10,943 | $39,496 | No exit manager |
| SOLUSD | 35.4 | $3,232 | No exit manager |
| AAPL | 0.667 | $168 | From daily-rl-trader paper |
| COUR | -7 | -$42 | Old short from paper trading |

---

## Bugs Found and Fixed (2026-03-25)

### Bug 1: ABEV stuck in pending_close with no exit order (CRITICAL — FIXED)

**Root cause**: `force_close_position` was called immediately after `cancel_symbol_orders`.
Alpaca hadn't processed the cancel yet when the new order arrived, so it rejected with:
```
{"code": 40310000, "message": "insufficient qty available for order (requested: 4459, available: 0)",
 "held_for_orders": "4459", "related_orders": ["ee917e61-..."]}
```
The force_close failed and ABEV was moved to `pending_close`. But `pending_close` only prevented further close attempts — it never retried.

**Timeline**:
- 2026-03-20: ABEV entered @ $2.73, exit_price=$2.83 (TP)
- 2026-03-22/23: Protective exits submitted at $2.83 every cycle but never filled (stock at ~$2.78)
- 2026-03-24 10:13 UTC: Hold timeout (5 NYSE hrs). cancel + force_close. Force_close FAILS (race condition).
- 2026-03-24 → 2026-03-25 01:42: ABEV in `pending_close`, 0 open orders, $12k exposed

**Fix 1a**: Added `pending_close_retry` — in the `still_pending` loop, if a position is untracked and has no exit order, re-issue `force_close_position`. Applied 01:42 UTC, first cycle confirmed `force_close_submit_succeeded`.

**Fix 1b**: Added `time.sleep(0.75)` after `cancel_symbol_orders` before any subsequent order submission (hold_timeout, missing_exit_price, and protective_exit paths). Prevents future race conditions.

### Bug 2: `abs(qty) < 1` treats crypto fractional positions as closed (FIXED)

**Location**: `trade_unified_hourly.py:1031`

**Problem**: A tracked position with `qty < 1` was treated as "position closed, cleaning state". Correct for stocks (fractional shares < 1 → effectively zero). Wrong for crypto: 0.5 ETH = ~$1000.

Without this fix, a tracked ETHUSD position with 0.5 ETH would be cleaned from state → added to pending_close → stuck forever (no close order, the original "closed" check was wrong).

**Fix**: For crypto, check `notional < $1.0` using current price rather than raw qty threshold.

```python
if is_crypto_symbol(symbol):
    position_is_zero = abs_qty_check < 1e-8 or (cur_price > 0 and notional < 1.0)
else:
    position_is_zero = abs_qty_check < 1
```

### Bug 3: daily-rl-trader using wrong model (FIXED)

Service was running with stale process using `stocks12_daily_tp05_longonly/best.pt` (old bad model, -2.55% median OOS). Code already had `DEFAULT_CHECKPOINT = stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt` (3-model ensemble, 0/50 neg).
Fix: restarted service. Now uses correct 3-model ensemble (s123+s15+s36, med=+47.30%).

---

## Simulator vs Live Gaps

### 1. market_hours_between used for crypto hold timeout
`manage_positions` uses `market_hours_between()` (NYSE 9:30-4:00 Mon-Fri) for ALL symbols including crypto. If/when crypto is added to the unified hourly bot:
- Crypto position entered Friday 4pm → hold time doesn't accrue until Monday 9:30am
- A 5-hour hold limit effectively becomes 5 NYSE hours = ~1 trading day
- In simulation: hold measured in calendar hours → shorter actual hold
- Fix needed: `calendar_hours_between` for crypto

### 2. force_close uses limit 0.3% below market (DAY TIF)
If market is closed when timeout triggers, the DAY order is queued. If price gaps down at open by more than 0.3%, the order may not fill immediately. Not a simulator/live gap but a slippage risk.

### 3. Limit exit orders may expire without filling
The protective exit is a limit sell at the signal's `sell_price`. If the stock never reaches that price within the hold window, the limit expires. `force_close` kicks in, but as seen with ABEV, can fail silently.

**Proposed improvement**: Use `TimeInForce.GTC` for protective exits (not DAY) so the TP order persists across days.

### 4. Entry timing mismatch: market order vs hourly bar close
- Simulator: fills at bar open (T+1 open price)
- Live: market order fills at ask price at ~9:35am ET (after bot runs at 9:35am)
- Usually very close but diverges during volatile opens

### 5. Paper crypto positions (hourlyv5) have no exit management
When `hourlyv5-stocks.service` was switched to `--asset-class stocks`, orphaned ETHUSD/BTCUSD/UNIUSD/SOLUSD paper positions have no exit manager. These will sit indefinitely in the paper account.
- **Not a live money issue** — all paper
- **Action**: Either run a manual paper close, or switch service to `--asset-class all`

---

## Consolidation Plan: Stocks + Crypto in One Algorithm

### Current fragmentation
1. `unified-stock-trader` — best algorithm, stocks only, LIVE
2. `hourlyv5-stocks` — older checkpoint, stocks only, PAPER
3. `daily-rl-trader` — pufferlib PPO daily signals, stocks only, PAPER
4. Binance worksteal — rule-based crypto dip-buying, LIVE (separate machine concern)
5. Binance hybrid-spot — BROKEN (no Gemini key), crypto, LIVE

### Why consolidate to unified hourly meta
- The meta-selector (`trade_unified_hourly_meta.py`) is the most sophisticated
- It already supports crypto symbols via `is_crypto_symbol()` throughout the code
- The fix to `abs(qty) < 1` makes it safe for crypto fractional positions
- `trade_unified_hourly.py` already has crypto-aware: place_exit_order, force_close, notional checks

### What needs to be added for Alpaca crypto
1. **market_hours_between for crypto** — need `calendar_hours_between` function, use it when `is_crypto_symbol(symbol)`
2. **Chronos2 caches for crypto** — verify BTCUSD/ETHUSD/SOLUSD have up-to-date hourly caches in `trainingdatahourly/crypto/`
3. **Minimum order size** — Alpaca crypto minimum is $1 (already handled by notional check)
4. **Add crypto symbols to supervisor command** — add `--crypto-symbols BTCUSD,ETHUSD,SOLUSD` to `unified-stock-trader` once above are done

### Stopping redundant services
- `hourlyv5-stocks.service`: **STOPPED AND DISABLED** 2026-03-25 (paper only, older model, no stop-loss — see analysis below)
- `daily-rl-trader.service`: **STOPPED AND DISABLED** 2026-03-25 (was using wrong checkpoint; see bug 3 above)

---

## Paper Bot Analysis (2026-03-25)

Both paper bots stopped and `strategy_state/trade_outcomes.json` analyzed (30 completed trades).

### Overall results
| Metric | Value |
|--------|-------|
| Completed trades | 30 |
| Total realized PnL | **-$3,781** |
| Win rate | 33% (10/30) |
| Top winner | QUBT:sell +$5,548 |
| Top loser | ADSK:sell -$5,079 |

### Active paper positions at shutdown
| Symbol | Qty | Value | Unrealized | Opened |
|--------|-----|-------|------------|--------|
| UNIUSD | 10,943 | $39,496 | +$536 | Nov 2025 |
| ETHUSD | 1.58 | $3,424 | +$2,155 | Nov 2025 |
| BTCUSD | 0.079 | $5,576 | -$592 | Nov 2025 |
| SOLUSD | 35.4 | $3,232 | -$14 | Nov 2025 |
Total unrealized: **+$2,092**

### Key findings

**1. Long-only crypto without stop-loss is catastrophic during downtrends**
- UNIUSD, ETHUSD, BTCUSD, SOLUSD all entered Nov 2025. Crypto market dropped 30–45% through Feb-Mar 2026.
- Positions remained open for 4+ months accumulating unrealized losses before partial recovery.
- Without stop-loss or max-hold, a buy signal → indefinite hold through the drawdown.
- **Lesson**: Do NOT add crypto to unified-meta without stop-loss / max-hold. The 5h max_hold in unified-meta is critical protective guard.

**2. "not_in_portfolio" close reason means positions held with no exit management**
- 21 of 30 completed trades (70%) closed via `not_in_portfolio` — model rotation removed the symbol, triggering exit.
- These exits are delayed by days/weeks until the model rotates again. In that window: no TP, no stop.
- **Lesson**: All open positions need an active exit order at all times (unified-meta does this; hourlyv5 did not).

**3. Short signals outperformed long signals dramatically**
- QUBT:sell +$5,548, INTC:sell +$3,561 — the two biggest winners were short/sell entries.
- Long crypto entries (UNIUSD -$4,912, BTCUSD -$1,109, ETHUSD -$1,910) were large losers.
- ADSK:sell -$5,079 was a failed short (stock rallied against position).
- **Lesson**: Market direction matters. The model's short calls were better than long crypto in this period.

**4. The unified-meta architecture is far superior**
- Explicit TP order + 5h max_hold + force_close retry = clean exits with bounded exposure.
- hourlyv5 had none of this: open-ended holds, no stop, model-rotation-only exits.
- **Conclusion**: hourlyv5 was running a worse algorithm with worse checkpoints. Stopping it was correct.

### What to carry forward
- Cryptos on unified-meta ONLY after: (a) `calendar_hours_between` fix for 24/7 hold timing, (b) paper test ≥2 weeks
- The 5h max_hold + explicit TP + force_close retry pattern is validated as essential
- Short signals from the Chronos2 model appear accurate; don't suppress them for crypto

---

## Next Experiments

### Priority 1: Validate ABEV close
At 2026-03-25 market open (~13:30 UTC), ABEV force_close order (limit sell ~$2.77) should fill.
Check: `PAPER=0 python -c "from env_real import *; from alpaca.trading.client import TradingClient; api = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False); [print(p.symbol, p.qty, p.unrealized_pl) for p in api.get_all_positions()]"`

### Priority 2: Fix market_hours_between for crypto
Add `calendar_hours_between` to `trade_unified_hourly.py`, use it when `is_crypto_symbol()`. Test: enter a crypto position in paper, verify hold timeout uses wall-clock hours not NYSE hours.

### Priority 3: Add Alpaca crypto to unified hourly (after priority 2)
- Verify Chronos2 caches current for BTC/ETH/SOL
- Add crypto symbols to supervisor command
- Run paper test with crypto for at least 2 weeks before live

### Priority 4: RunPod training experiments
- Use RunPod GPU pool for crypto70/stocks12 sweeps in parallel with local
- Gives more seed coverage per day
- See `feedback_runpod_ssh.md` for SSH setup notes (use `cloudType=COMMUNITY`)

### Priority 5: Investigate early-stopping behavior of meta-selector
The unified-stock-trader logs show many `run_portfolio_simulation early stopping` messages.
These are the per-strategy daily-return simulations used for meta-selection.
High early-stop rate (~95%) may indicate the strategies are mostly underperforming in recent 14-day lookback window.
Worth checking: what `meta_winner_selected` shows — which strategy is winning and whether it's genuinely better.

---

## Monitor Commands

```bash
# Live account positions
PAPER=0 .venv313/bin/python -c "import alpaca_wrapper; [print(p.symbol, p.qty, p.unrealized_pl) for p in alpaca_wrapper.get_all_positions()]"

# Unified hourly meta logs
sudo tail -f /var/log/supervisor/unified-stock-trader.log

# ABEV event history
grep "ABEV" strategy_state/stock_event_log.jsonl | python3 -c "import sys,json; [print(json.loads(l).get('logged_at','')[:19], json.loads(l).get('event_type'), json.loads(l).get('action','')) for l in sys.stdin if 'logged_at' in l]" | tail -20

# Bot state
python3 -c "import json; d=json.load(open('strategy_state/stock_portfolio_state.json')); print('tracked:', list(d.get('positions',{}).keys())); print('pending_close:', d.get('pending_close',[]))"

# Training progress
ps aux | grep "pufferlib_market.train" | grep -v grep | awk '{print $2, substr($0, index($0,"--seed")), " | " substr($0, index($0,"--checkpoint-dir"))}'
```
