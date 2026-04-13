# Alpaca Deployment Runbook

This runbook is for the Alpaca live writers in this repo, especially the
supervisor-managed `llm-stock-trader` stock planner and the systemd-managed
`daily-rl-trader`.

## Preconditions

- Source secrets first:
  ```bash
  source ~/.secretbashrc
  source .venv313/bin/activate
  ```
- Read the live rules in [AGENTS.md](../AGENTS.md) and the current-prod ledger
  in [alpacaprod.md](../alpacaprod.md).
- Do not restart a live writer blindly from a dirty repo. Run the preflight and
  inspect the blocker list first.

## 1. Inspect current live state

Check the running process directly:

```bash
ps -ef | rg "unified_orchestrator\\.orchestrator|trade_daily_stock_prod.py"
```

Check repo-side preflight for the stock LLM service:

```bash
python scripts/alpaca_deploy_preflight.py --service llm-stock-trader
```

Useful live files:

- Cycle log: `strategy_state/orchestrator_cycle_events.jsonl`
- Stock event log: `strategy_state/stock_event_log.jsonl`
- Supervisor stdout: `/var/log/supervisor/llm-stock-trader.log`
- Supervisor stderr: `/var/log/supervisor/llm-stock-trader-error.log`

Important nuance: outside US stock hours the stock orchestrator can still be
healthy while reporting `Regime: CRYPTO_ONLY`. That confirms the daemon is up,
not that the stock branch traded in that moment.

## 2. Run focused validation before deploy

Targeted tests for the stock orchestrator path:

```bash
.venv313/bin/pytest -q \
  tests/test_unified_orchestrator_prompt_builder.py \
  tests/test_unified_orchestrator_stock_execution.py \
  tests/test_unified_orchestrator_llm_runtime.py \
  tests/test_unified_orchestrator_orchestrator.py \
  tests/test_alpaca_deploy_preflight.py \
  tests/test_symbol_conflict.py
```

Optional direct stock signal dry-run against live Alpaca history with a cheap
Gemini model:

```bash
.venv313/bin/python - <<'PY'
from unified_orchestrator.orchestrator import get_stock_signals
from unified_orchestrator.state import UnifiedPortfolioSnapshot

snapshot = UnifiedPortfolioSnapshot(
    alpaca_cash=28_679.0,
    alpaca_buying_power=57_358.0,
    regime="STOCK_HOURS",
    market_is_open=True,
    minutes_to_close=180,
)

signals = get_stock_signals(
    ["NET", "COIN"],
    snapshot,
    model="gemini-3.1-flash-lite-preview",
    thinking_level="LOW",
    dry_run=True,
)

for sym, plan in signals.items():
    print(sym, plan)
PY
```

Use this as a prompt/runtime smoke test, not as a historical deployment proof.

## 3. Deploy `llm-stock-trader`

The checked-in launcher is:

- [deployments/llm-stock-trader/launch.sh](../deployments/llm-stock-trader/launch.sh)

Repo-side target config as of 2026-04-12:

- model: `gemini-3.1-pro-preview`
- thinking level: `HIGH`
- lock: `llm_stock_writer`
- symbols: `YELP NET DBX OPTX PDYN COIN CRWD`
- marketsim/backtest model: keep `gemini-3.1-flash-lite-preview`; only live production should use Pro

Restart via supervisor on the host:

```bash
sudo supervisorctl restart llm-stock-trader
```

If the restarted PID is still on the old model, inspect the installed
supervisor program directly:

```bash
sed -n '1,200p' /etc/supervisor/conf.d/llm-stock-trader.conf
```

On this host, supervisor was using a root-owned inline `command=...` rather
than delegating to the repo `launch.sh`, so changing the repo launcher alone was
not sufficient.

Then verify:

```bash
ps -ef | rg "unified_orchestrator\\.orchestrator"
tail -n 40 strategy_state/orchestrator_cycle_events.jsonl
tail -n 120 /var/log/supervisor/llm-stock-trader.log
tail -n 80 /var/log/supervisor/llm-stock-trader-error.log
```

You want to see the new model and PID in the cycle log:

- `model: gemini-3.1-pro-preview`
- `reprompt_passes: 1`
- expected stock symbol set

## 4. Deploy `daily-rl-trader`

Current live command pattern:

```bash
.venv313/bin/python -u trade_daily_stock_prod.py --daemon --live --allocation-pct 12.5
```

Current repo defaults as of 2026-04-13:

- keep the conservative daily-RL open gate on by default
- `DEFAULT_MIN_OPEN_CONFIDENCE=0.20`
- `DEFAULT_MIN_OPEN_VALUE_ESTIMATE=0.0`

If you want to test a gate-off variant, pass it explicitly on the CLI or in a
local experiment. Do not leave that change in the production default path
without a qualifying positive replay.

Restart:

```bash
sudo systemctl restart daily-rl-trader.service
sudo systemctl status daily-rl-trader.service --no-pager
```

## 5. Update the production ledger

Whenever a live config changes or a live restart happens:

1. Update [alpacaprod.md](../alpacaprod.md) with the exact model, symbol set,
   launch path, timestamp, and any validation evidence.
2. If replacing an older "current" snapshot, archive the previous state under
   `old_prod/`.

## Known limitations

- The stock executor currently opens new long positions and manages exits for
  held longs. The LLM can still emit `short` plans, which are logged but not
  opened as new short positions by `execute_stock_signals(...)`.
- `python scripts/alpaca_deploy_preflight.py --service llm-stock-trader` will
  refuse `--apply` from a dirty repo outside the service watchlist.
