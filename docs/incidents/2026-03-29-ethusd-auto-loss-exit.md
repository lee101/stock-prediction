# 2026-03-29 ETHUSD Auto-Loss Exit Incident

## Summary

On 2026-03-29 the live Alpaca crypto stack repeatedly opened `ETHUSD` and then closed it shortly after at a loss. The observed live pattern was:

- limit buy near the top of the hour
- automatic sell posted almost immediately after
- realized exit occurred below fee-aware breakeven

This was unacceptable because the live system had no hard invariant preventing automatic loss-taking exits on fresh crypto entries.

## Confirmed Runtime Context

At investigation time on 2026-03-29:

- host: `leaf-gpu-dedicated-server`
- live crypto service: `unified-orchestrator.service`
- live stock service: `daily-rl-trader.service`
- duplicate-order guard: `alpaca-cancel-multi-orders.service`

Additional operational issue discovered during the incident:

- after key rotation, `alpaca_wrapper` calls to Alpaca `account`, `positions`, `orders`, and `clock` were returning `401 unauthorized`
- this specifically broke the duplicate-order guard and any other `alpaca_wrapper`-based paths

## Root Cause

The live crypto stack was missing multiple checks that should have made the bad behavior impossible:

1. Trailing-stop exits were allowed before the position had ever reached fee-aware profitability.
2. Passive crypto exit orders were not forced above a round-trip fee-aware breakeven floor.
3. The intra-hour watcher could post a sell order using a target that was too close to entry.
4. The LLM prompt had incorrect fee context for Alpaca crypto (`BTC/ETH` were described as `0 bps`).

The resulting failure mode was:

- enter a fresh crypto long
- arm an automatic exit path too early
- submit a sell order that could close the trade below true breakeven

## Safeguards Added

The code now enforces the following:

1. Fee-aware crypto profit floor
   - all passive crypto sells are raised to at least:
   - entry price + conservative round-trip fees + extra safety buffer

2. Trailing-stop arming rule
   - trailing stops do not arm until the trade's peak price has first cleared the fee-aware profit floor

3. Loss-block on automatic trailing exits
   - if a trailing-stop exit would place a sell below the profit floor, the order is clamped up to the profit floor instead of submitting a guaranteed-loss automatic exit

4. Watcher sell-floor enforcement
   - the watcher now raises post-fill sell targets to the same fee-aware floor

5. Prompt fee correction
   - Alpaca crypto is now described with non-zero per-side fees so the model does not optimize around false zero-fee assumptions

## Tests Added

Regression coverage was added for:

- TP sells being raised to the fee-aware floor
- trailing stops being disarmed before a position ever reaches profit
- trailing-stop exits being clamped to the profit floor when armed
- watcher-posted sells being raised to the fee-aware floor

Targeted validation run:

```bash
.venv313/bin/pytest -q \
  tests/test_unified_orchestrator_execute_crypto.py \
  tests/test_unified_orchestrator_watcher.py \
  tests/test_orchestrator_order_mgmt.py
```

Result on 2026-03-29:

- `48 passed`

## Follow-Up

Still recommended after this patch:

- rotate all plaintext secrets committed into `env_real.py` and systemd unit files
- remove hardcoded secrets from the repo entirely
- restart only after the live credential path is corrected and verified
- verify the live broker account state directly before re-enabling automated trading
