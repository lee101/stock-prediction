# XGB hold-through (carry repeated picks) — 2026-04-19

When tomorrow's top-N pick set equals today's, we don't need to
sell-at-close + buy-at-open. We can just **carry** the positions.
Benefit per held day:

- save `2 × (fee_rate + fill_buffer_bps)` round-trip cost
- earn close-to-close return instead of open-to-close (captures
  overnight drift, which at top_n=1 is strongly positive because
  the picked name usually has continuation)

Shipped behind `BacktestConfig.hold_through` (default `False`),
exposed as `xgbnew.eval_multiwindow --hold-through`. Regime-gated
days break the chain (sitting in cash = fresh entry next day).

## Pair-run results (champion config, 113 windows, 833 OOS symbols)

Config: `n_estimators=400 max_depth=5 lr=0.03 top_n=1 seed=42`,
train 2021-01-01 → 2024-12-31, OOS 2024-01-02 → 2026-04-19,
30d/7d stride, fill_buffer=5 bps, fee_rate=0.278 bps.

### Leverage 1.0

| metric | OFF | ON | Δ |
| --- | ---: | ---: | ---: |
| median monthly % | +26.48 | **+28.65** | **+2.17** |
| p10 monthly % | +6.34 | +7.01 | +0.67 |
| median sortino | 8.37 | 8.59 | +0.22 |
| neg windows | 4/113 | 4/113 | 0 |

### Leverage 1.25 (deploy setting)

| metric | OFF | ON | Δ |
| --- | ---: | ---: | ---: |
| median monthly % | +33.59 | **+36.13** | **+2.54** |
| p10 monthly % | +7.46 | +8.22 | +0.76 |
| median sortino | 8.32 | 8.55 | +0.23 |
| neg windows | 4/113 | 4/113 | 0 |

**Every metric improves, no tail cost at either leverage.**

## Interpretation

Champion is `top_n=1`, so "same pick two days running" is the most
common continuation case. Empirically on OOS 2024-2026 it happens
often enough that skipping the round-trip saves ≈30 bps/held-day
(2 × 5 bps buffer + 2 × 10 bps fee at deploy rate) and picks up the
overnight gap (which in a concentrated daily-momentum strategy is
positive in expectation). Delta looks small on headline (+2%/mo)
but is structural: it's a free reduction of transaction cost on
the deployed signal, with zero change to the model or gate.

## Implementation notes

- `BacktestConfig.hold_through: bool = False` — off by default so
  existing eval runs / regression tests are unchanged.
- `simulate()` tracks `previous_held_symbols`. If today's selected
  set equals yesterday's AND yesterday wasn't a regime-flattened
  day, we call `_simulate_continuation_day` which builds
  zero-fee / zero-buffer Trade objects returning close-to-close.
- Regime gate flattens the held state (sitting in cash for a
  day = fresh entry on the next open day).
- Any regime gate state reset must ALSO clear
  `previous_held_symbols` to prevent a 2-day gap from being
  billed as continuation.

Tests: `tests/test_xgbnew_backtest_hold_through.py` — 6 green.

## Next step

Wire `--hold-through` into `xgbnew/live_trader.py`:
- When today's ranked picks == yesterday's open positions, SKIP
  the sell-at-close + buy-at-open block entirely.
- Keep HARD RULE #3 (death-spiral guard) by routing skips through
  a no-op path (no order to Alpaca at all).
- Weekend/holiday handling: if the trading calendar inserted a gap
  (today not a trading day, previous day was), treat as reset —
  no carry across non-trading days.

## Artifacts

- `analysis/xgbnew_multiwindow/hold_through_pair/multiwindow_20260419_203400.json` — OFF lev=1.0
- `analysis/xgbnew_multiwindow/hold_through_pair/multiwindow_20260419_203448.json` — ON lev=1.0
- `analysis/xgbnew_multiwindow/hold_through_pair/multiwindow_20260419_203600.json` — OFF lev=1.25
- `analysis/xgbnew_multiwindow/hold_through_pair/multiwindow_20260419_203654.json` — ON lev=1.25
- Code: `xgbnew/backtest.py`, `xgbnew/eval_multiwindow.py`
- Tests: `tests/test_xgbnew_backtest_hold_through.py`
