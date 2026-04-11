# Binance Production Systems

## CRITICAL (2026-04-07): Lag=0 Validation Is Untrustworthy
`robust_champion` was briefly deployed after a "+216% Sort=25.06" holdout — this was run at `decision_lag=0` which embeds lookahead bias (agent sees bar t features and trades at bar t close, impossible live). At realistic `decision_lag=2` on the SAME `mixed23_latest_val` data (50 windows x 30h):

| Checkpoint | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| robust_champion (lag=0 claim: +12.4%/+3.35) | -2.8% | -0.50 | -14.0% |
| robust_reg_tp005_dd002 | +0.0% | +0.76 | -26.9% |
| robust_reg_tp005_ent | -5.5% | -2.09 | -21.1% |

**PROTOCOL**: Never trust a holdout unless run with `--decision-lag 2` AND a slippage sweep (0/5/10/20 bps). `evaluate_holdout.py` now has `--slippage-bps`. The C training env has a built-in `t_obs = t-1` single-bar lag, but production exhibits >=2 bars lag (forecast build + Gemini call + order placement). Training sim vs production gap is the ROOT CAUSE of why every deploy since 2026-03 looks great in holdout and drifts flat/negative live.

Launch.sh reverted to dd002 pending supervisor restart:
`sudo supervisorctl restart binance-hybrid-spot`

Artifacts: `analysis/robust_champion_validation/slip{0,5,10,20}.json`

## Audit (2026-04-09): Live-Like Six-Symbol Replay Is Still Not Deployable

Artifacts: `analysis/current_binance_prod_eval_20260409/`

### Reproduced current live config

Live source of truth remains:

- launch: `deployments/binance-hybrid-spot/launch.sh`
- symbols: `BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD`
- leverage: `0.5x`
- model: `gemini-3.1-flash-lite-preview`
- RL checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`

### Runtime drift confirmed on machine

Follow-up machine audit on 2026-04-09 showed the launch file is **not** what the running hybrid PID is executing:

- launch target checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- running hybrid checkpoint: `pufferlib_market/checkpoints/a100_scaleup/robust_champion/best.pt`
- running hybrid PID: `1712424`

The process audit was also tightened in `src/binance_live_process_audit.py` so cache-only research helpers no longer count as live-writer conflicts.

After that fix, the remaining true live-writer conflicts on the machine are:

- `binanceleveragesui.trade_margin_meta`
- `binance_worksteal.trade_live`

So the current production risk is not just model quality; it is also checkpoint drift plus shared-account contamination from multiple real live writers.

Production-faithful replay settings:

- data: `pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin`
- `decision_lag=2`
- `slippage_bps=5`
- `fill_buffer_bps=5`
- `fee_rate=0.001`
- long-only
- tradable symbols masked to the six live Binance symbols

Reproduced 30-step / 50-window baseline:

| Checkpoint | med_ret | med_sort | p10_ret | med_dd |
|---|---:|---:|---:|---:|
| live `robust_reg_tp005_dd002` | `-2.90%` | `-3.84` | `-5.94%` | `4.63%` |

### Candidate sweep under the same live constraints

Best six-symbol 30-step candidates tested on 2026-04-09:

| Checkpoint | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| `gspo_like_drawdown_mix15_tp01` | `-1.26%` | `-1.48` | `-10.25%` |
| `gspo_like_drawdown_mix15` | `-1.38%` | `-2.47` | `-6.30%` |
| `reg_combo_2` | `-2.36%` | `-2.54` | `-11.74%` |
| `gspo_like_smooth_mix15` | `-2.38%` | `-2.05` | `-9.29%` |
| live `robust_reg_tp005_dd002` | `-2.90%` | `-3.84` | `-5.94%` |

None of the tested six-symbol configurations came out positive on the 30-step median.

### Full-span check with early-stop disabled

To avoid the holdout early-exit heuristic hiding losses, the top candidates were re-run across the full 179-step slice with `--no-early-stop`:

| Checkpoint | total_ret | sortino | max_dd |
|---|---:|---:|---:|
| `gspo_like_smooth_mix15` | `-5.75%` | `-0.18` | `18.84%` |
| `gspo_like_drawdown_mix15` | `-20.12%` | `-2.21` | `24.64%` |
| `gspo_like_drawdown_mix15_tp01` | `-21.04%` | `-1.83` | `28.61%` |
| live `robust_reg_tp005_dd002` | `-24.07%` | `-2.19` | `33.65%` |
| `reg_combo_2` | `-29.75%` | `-1.90` | `38.17%` |

`gspo_like_smooth_mix15` is the least bad checkpoint on the full span, but still not good enough to promote as-is.

### Subset search result

The six-symbol basket is hurting `gspo_like_smooth_mix15`. Searching all subsets of the six live symbols at the live `0.5x` leverage found:

| Tradable subset | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| `ETHUSD` | `-0.06%` | `-3.46` | `-2.10%` |
| `BTCUSD,ETHUSD` | `-0.19%` | `-3.86` | `-2.53%` |
| `BTCUSD` | `-0.26%` | `-3.68` | `-1.64%` |

These are much better than the live six-symbol basket, but still slightly negative on median 30-step replay.

### Leverage simulator bug fixed

`pufferlib_market/hourly_replay.py` was silently disabling long leverage above `1.0x` for limit entries by clamping long notional back to available cash inside `_open_long_limit`.

Fix applied on 2026-04-09:

- leveraged long limit entries now allow negative cash, matching the existing equity accounting
- regression test added in `tests/test_pufferlib_market_hourly_replay.py`

Relevant test command:

```bash
source .venv313/bin/activate
pytest -q tests/test_pufferlib_market_hourly_replay.py tests/test_evaluate_holdout.py
```

Post-fix full-span result for the best concentrated candidate (`gspo_like_smooth_mix15`, `ETHUSD` only):

| leverage | total_ret | sortino | max_dd |
|---|---:|---:|---:|
| `0.5x` | `+2.87%` | `1.90` | `2.18%` |
| `1.0x` | `+5.61%` | `1.91` | `4.35%` |
| `2.0x` | `+10.68%` | `1.91` | `8.67%` |

But the same ETH-only setup remains negative on the 30-step median:

| leverage | med_ret | p10_ret |
|---|---:|---:|
| `0.5x` | `-0.06%` | `-2.10%` |
| `1.0x` | `-0.11%` | `-4.21%` |
| `2.0x` | `-0.23%` | `-8.41%` |

### Current recommendation

- Do **not** promote a new live checkpoint yet.
- The best paper candidate to keep evaluating is:
  - checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/gspo_like_smooth_mix15/best.pt`
  - tradable subset: `ETHUSD` only
  - leverage to study: `1.0x` to `2.0x`
- The next training loop should optimize directly for the concentrated `BTC/ETH` or `ETH`-only deployment profile, because the mixed 23-symbol models improve materially once masked down to that subset.

### Follow-up candidate ranking (2026-04-09 later)

Artifacts:

- `analysis/binance_prod_live6_23sym_candidates_20260409/`
- `analysis/per_symbol_all23_holdout_20260409/`
- `analysis/live6_top_singletons_fullspan_20260409/`
- `analysis/per_symbol_all23_fullspan_2x_20260409/`

Fresh six-symbol 30-step / 50-window rerun across the main 23-symbol-compatible checkpoints:

| Checkpoint | med_ret | med_sort | p10_ret | med_dd |
|---|---:|---:|---:|---:|
| `per_env_adv_smooth` | `-2.08%` | `-2.39` | `-10.23%` | `5.92%` |
| `reg_combo_2` | `-2.36%` | `-2.54` | `-11.74%` | `6.85%` |
| `gspo_like_smooth_mix15` | `-2.38%` | `-2.05` | `-9.29%` | `6.31%` |
| launch `robust_reg_tp005_dd002` | `-2.90%` | `-3.84` | `-5.94%` | `4.63%` |
| actual running `robust_champion` | `-4.48%` | `-6.17` | `-8.57%` | `5.17%` |

Important read:

- `per_env_adv_smooth` was the least-bad six-symbol basket on median return, but it did **not** produce a deployable positive basket.
- the actual running checkpoint `robust_champion` is materially worse than the launch target on this production-shaped slice

Best meaningful concentrated full-span results at `2.0x` leverage:

| Checkpoint | subset | total_ret | sortino | max_dd |
|---|---|---:|---:|---:|
| `gspo_like_smooth_mix15` | `ETHUSD` | `+10.68%` | `1.91` | `8.67%` |
| `gspo_like_smooth_mix15` | `BTCUSD` | `+7.19%` | `1.03` | `15.66%` |
| `gspo_like_smooth_mix15` | `DOGEUSD` | `+5.76%` | `0.86` | `43.27%` |

Counterexamples from the same pass:

- launch `robust_reg_tp005_dd002` on `AAVEUSD`: `-10.36%`, Sortino `-1.14`, DD `17.65%` at `0.5x`
- `robust_champion` on `BTCUSD`: flat median on the short-window sweep was a no-trade artifact, and full-span `2.0x` still came out `-10.06%`

Bottom line:

- no existing model/subset combination found here is close to `27%` monthly PnL under the live-like lag/slippage rules
- the best current paper lane is still concentrated `gspo_like_smooth_mix15` on `ETHUSD`, studied at `1.0x` to `2.0x`, but it remains a research candidate rather than a live promotion

## Active Services

### 1. binance-hybrid-spot
- **Supervisor**: `binance-hybrid-spot` (auto-restart)
- **Entry**: `rl_trading_agent_binance/trade_binance_live.py`
- **Launch**: `deployments/binance-hybrid-spot/launch.sh`
- **Config**: `deployments/binance-hybrid-spot/supervisor.conf`
- **Python**: `.venv313`
- **Symbols**: BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD + ADA XRP SUI (positions held)
- **Mode**: Cross-margin, 0.5x leverage, hourly cycles
- **Launch target RL model**: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- **Actual running RL model on 2026-04-09**: `pufferlib_market/checkpoints/a100_scaleup/robust_champion/best.pt`
- **Previous model**: `robust_reg_tp005_ent` (switched 2026-03-31, dd002 has better 120d: +54.6% Sort=2.85 vs +42.4% Sort=2.71)
- **LLM**: Gemini 3.1 Flash Lite (thinking=HIGH)
- **Equity**: ~$3,025 (2026-03-31)
- **Logs**: `/var/log/supervisor/binance-hybrid-spot-error.log`

**Cycle flow**:
1. Fetch portfolio + margin balances
2. Compute Chronos2 h24 forecasts
3. Run RL inference (pufferlib checkpoint)
4. Build prompt -> call Gemini for allocation
5. Place/cancel margin limit orders
6. Sleep 3600s

**Current behavior** (2026-03-31):
- Gemini very defensive: 65-77% cash allocation
- RL consistently signals LONG_XRP and LONG_ADA
- DOGE position being sold down via existing sell order
- Stale orders auto-cancelled after 2h
- **Model switch** (2026-03-31): `robust_reg_tp005_dd002` deployed
  - 120d backtest: +54.6% ret, Sort=2.85, DD=26.9%, 50 trades, 54% WR
  - Exec param sweep: slippage insensitive, fill buffer insensitive
  - Better than ent at all periods except 30d

### 2. binance-worksteal-daily
- **Supervisor**: `binance-worksteal-daily` (auto-restart)
- **Entry**: `binance_worksteal/trade_live.py`
- **Launch**: `deployments/binance-worksteal-daily/launch.sh`
- **Config**: `deployments/binance-worksteal-daily/supervisor.conf`
- **Python**: `.venv313`
- **Universe**: `binance_worksteal/universe_live.yaml` (15 symbols)
- **Mode**: Spot+margin, dip-buying with SMA-20 filter
- **Equity**: ~$3,044 (2026-03-31)
- **Logs**: `/var/log/supervisor/binance-worksteal-daily-error.log`
- **State**: `binance_worksteal/live_state.json`
- **Trades**: `binance_worksteal/trade_log.jsonl`

**Parameters** (from launch.sh):
- `--max-position-pct 0.10`
- `--rebalance-seeded-positions`
- dip_pct=0.20, tp=20%, sl=15%, trail=3%, maxpos=5, maxhold=14d, sma=20

**Current positions** (2026-03-31):
- ADAUSD: entry $0.2726 (Mar 25), target $0.27, stop $0.23
- ZECUSD: entry $238.05 (Mar 25), target $285.66, stop $202.34
- XRPUSD: entry $1.36 (Mar 26), target $1.42, stop $1.16

## Monitoring

```bash
# Check service status
sudo supervisorctl status

# View recent logs
tail -100 /var/log/supervisor/binance-hybrid-spot-error.log
tail -100 /var/log/supervisor/binance-worksteal-daily-error.log

# Check running processes
ps aux | grep trade | grep -v grep

# Restart services
sudo supervisorctl restart binance-hybrid-spot
sudo supervisorctl restart binance-worksteal-daily

# View worksteal positions
python3 -m json.tool binance_worksteal/live_state.json
```

## Deployment

```bash
# After code changes, restart to pick up new code:
sudo supervisorctl restart binance-worksteal-daily
sudo supervisorctl restart binance-hybrid-spot

# Or kill the process (supervisor auto-restarts):
kill -TERM $(pgrep -f "trade_live.*--live")
kill -TERM $(pgrep -f "trade_binance_live")
```

## Known Issues (2026-03-31)

### Fixed: LOT_SIZE sell failures (worksteal)
- **Symptom**: Positions stuck 6 days, all sell orders rejected by Binance
- **Root cause**: Running process (started Mar 26) predated `_prepare_limit_order` quantization
- **Fix**: Restarted process to pick up code with exchange filter quantization
- **Prevention**: `_prepare_limit_order` now quantizes qty to stepSize before submission

### Fixed: Invalid symbols in universe_v2.yaml
- HYPEUSDT, KASUSDT removed (not valid Binance margin symbols)
- Was causing APIError -1121 every scan cycle

### Fixed: Margin 15% limit-price spam
- ENJUSD deep-dip buy orders were being submitted every 4h and rejected (code -3064)
- Added pre-check in `_prepare_limit_order`: rejects orders >14% from index price

## Market Simulator Validation

```bash
# Validate pufferlib model with binary fills (ground truth):
python -m pufferlib_market.evaluate_fast \
  --checkpoint <path> \
  --data-path pufferlib_market/data/crypto15_daily_val.bin \
  --eval-hours 720 --n-windows 100 \
  --fee-rate 0.001 --fill-slippage-bps 5

# Validate worksteal strategy:
python -m binance_worksteal.backtest --days 30

# Multi-period eval:
python -m pufferlib_market.evaluate_multiperiod \
  --checkpoint <path> \
  --data-path <bin> \
  --periods 1d,7d,30d --json
```

## Realism Parameters
- fee=10bps, margin=6.25% annual, fill_buffer=5bps, max_hold=6h
- validation_use_binary_fills=True
- fill_temperature=0.01
- decision_lag>=2
- Test at slippage 0/5/10/20 bps before deploying

## Runtime Audit (2026-04-09)

The recent "market was good but hybrid still lost" question is not explainable from the hybrid model alone because the Binance margin account is not isolated.

Active live Binance traders on this machine at audit time:

- `rl_trading_agent_binance/trade_binance_live.py --live --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD --execution-mode margin --leverage 0.5`
- `python -m binanceleveragesui.trade_margin_meta ... --doge-checkpoint ... --aave-checkpoint ... --max-leverage 2.30`
- `python -m binanceexp1.trade_binance_selector --symbols BTCUSD,ETHUSD,SOLUSD ...`
- `python -m binanceexp1.trade_binance_hourly --symbols SOLUSD ...`

That means the hybrid bot is sharing a live Binance account with other active strategies, including a margin strategy that explicitly trades `AAVEUSDT` and `DOGEUSDT`.

### What the hybrid cycle snapshots showed

Window audited:

- `2026-04-08T00:00:00+00:00` to `2026-04-08T22:00:00+00:00`

Command:

```bash
source .venv313/bin/activate
python -m src.binance_hybrid_runtime_audit \
  --launch-script deployments/binance-hybrid-spot/launch.sh \
  --trace-dir strategy_state/hybrid_trade_cycles \
  --start 2026-04-08T00:00:00+00:00 \
  --end 2026-04-08T22:00:00+00:00 \
  --output-json analysis/current_binance_prod_eval_20260409/runtime_audit_20260408.json
```

Result:

- `22` allocation snapshots
- `12` snapshots with unexpected borrowed quote despite a `0.5x` launch
- `AAVEUSD` flagged as oversized in `6` managed-position snapshots
- `AAVEUSD` flagged in `15` oversized managed-order buckets
- foreign symbol/order activity also present: `DOTUSD`, `XRPUSDT`

The most concrete bad state was:

- `2026-04-08T19:09:05Z`
- account equity about `$3078`
- borrowed quote about `$6265`
- `AAVE` position `79.21747868`
- planned AAVE sleeve only about `$231`

So the account was not simply "missing an up day." It was carrying a massively stale / foreign AAVE exposure that the hybrid loop then tried to unwind with large sell orders.

### Why the recent PnL looked wrong

- the local market data over the last completed 24 hourly bars was broadly positive:
  - `BTC +2.96%`
  - `ETH +4.46%`
  - `AAVE +4.18%`
  - `LINK +2.98%`
  - `SOL +1.65%`
  - `DOGE +1.10%`
- but the hybrid account state was not the clean equal-weight basket assumed by the replay
- the live account was being mutated by other bots and by leftover margin debt / working orders
- therefore the "market simulator says it should be up" vs "production equity is down" comparison was contaminated at the account layer, not just at the model layer

### Code support added

`src/binance_hybrid_runtime_audit.py` now flags:

- borrowed quote exposure beyond the launch leverage
- managed positions far larger than their planned sleeve
- managed orders far larger than their planned sleeve

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_runtime_audit.py
```

Result: `9 passed`

### Practical conclusion

Do not use current live PnL as a clean readout of the hybrid strategy until the Binance account is isolated from:

- `binanceleveragesui.trade_margin_meta`
- `binanceexp1.trade_binance_selector`
- `binanceexp1.trade_binance_hourly`

If we want simulator-vs-production comparisons to mean anything, the next production step is account/process isolation first, not another checkpoint swap.

### Live fail-safe added

The highest-impact code change after the contamination audit was to harden the live hybrid loop itself.

`rl_trading_agent_binance/trade_binance_live.py` now evaluates the live account before any context fetch, Gemini call, or order placement and blocks live trading with status `account_guard_blocked` when it sees:

- borrowed quote inconsistent with the configured leverage
- foreign positions or working orders outside the configured launch symbol set
- single-symbol live sleeves or working orders larger than the hybrid gross budget implied by `effective_leverage`

This matters because the prior failure mode was not just "bad forecast quality"; it was the hybrid loop continuing to trade on top of a contaminated shared margin account. The new guard turns that into an explicit unhealthy runtime state instead of letting the bot compound the mismatch.

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_hybrid_allocation_prompt.py tests/test_binance_hybrid_prod_verify.py
```

Result: `57 passed`

### Deploy preflight added

There was still a clear gap after the live cycle guard: the deploy gate and autoresearch auto-deploy path could approve a candidate even while the machine was already running other Binance live writers.

That is now hardened with a process-isolation preflight:

- new audit module: `src/binance_live_process_audit.py`
- detects the Binance writer processes we actually observed on this machine:
  - `rl_trading_agent_binance/trade_binance_live.py`
  - `binanceleveragesui.trade_margin_meta`
  - `binanceexp1.trade_binance_selector`
  - `binanceexp1.trade_binance_hourly`
  - `binance_worksteal.trade_live`
- `deploy_candidate_checkpoint()` now fails closed before touching supervisor if the process set is not isolated
- `gate_deploy_candidate()` supports `require_process_isolation=True`
- `binance_autoresearch_forever.py` now uses that isolation requirement on the real auto-deploy path

This matters because the prior deploy logic could still say "candidate passes deploy gate" even when the live account was already contaminated by concurrent Binance writers. That was the highest-value remaining production gap after the runtime guard.

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_deploy_gate.py tests/test_binance_autoresearch_forever.py tests/test_search_binance_prod_config_grid.py
```

Result: `171 passed`

### Live-process drift audit

The next concrete gap I observed on the machine was launch/process drift:

- `deployments/binance-hybrid-spot/launch.sh` points at `mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- the actual running hybrid PID was still `1712424`
- that PID was running `a100_scaleup/robust_champion/best.pt`

That means "launch file says X" and "actual live process is X" were not equivalent, which makes both production debugging and automated deployment decisions less trustworthy.

Implemented:

- extracted a shared `trade_binance_live.py` command parser into `src/binance_hybrid_launch.py`
- added `src/binance_hybrid_process_audit.py` to compare the running hybrid process command against the launch config
- `scripts/binance_autoresearch_forever.py` now blocks real deploy attempts when:
  - Binance live-writer isolation fails
  - or the currently running hybrid process does not match the current launch script

This closes the specific failure mode where an automated rollout could proceed while the machine was already in a stale live state.

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_process_audit.py tests/test_binance_autoresearch_forever.py tests/test_evaluate_binance_hybrid_prod.py tests/test_binance_deploy_gate.py
```

Result: `167 passed`

Machine sanity check:

```bash
source .venv313/bin/activate
python - <<'PY'
from src.binance_hybrid_process_audit import audit_running_hybrid_process_matches_launch
print(audit_running_hybrid_process_matches_launch('deployments/binance-hybrid-spot/launch.sh').reason)
PY
```

Observed result: `running hybrid process does not match launch config: rl_checkpoint`

### Manual deploy preflight alignment

Review checkpoint follow-up:

- `scripts/deploy_crypto_model.sh` still did not enforce the same live-process preflight as `scripts/binance_autoresearch_forever.py`
- that meant a manual deployment could still restart into the exact conflict/drift state the autoresearch path would reject

Fixed:

- tightened `src/binance_live_process_audit.py` so it only classifies actual Python writer processes instead of matching raw substrings in arbitrary commands
- added the same live-process preflight to `scripts/deploy_crypto_model.sh`
  - blocks when multiple Binance live writers are active
  - blocks when the currently running hybrid process command does not match `launch.sh`

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_hybrid_process_audit.py tests/test_deploy_crypto_model.py
```

Result: `39 passed`

### Production eval now audits machine state

The next important gap was that `scripts/evaluate_binance_hybrid_prod.py` still treated the "current production baseline" as if runtime snapshots were the whole truth. That missed two machine-level failures we had already observed directly:

- multiple Binance live writers on the same account
- running hybrid process checkpoint drift versus `launch.sh`

Implemented:

- added `src/binance_hybrid_machine_audit.py`
- `scripts/evaluate_binance_hybrid_prod.py` now records:
  - `current_machine_audit`
  - `current_machine_audit_issues`
  - `current_machine_health_issues`
- for the default production launch, those machine issues are merged into the existing manifest baseline issue fields, so deploy gating treats them as part of the production baseline honesty check
- `src/binance_deploy_gate.py` also explicitly rejects manifests that already admit machine drift/health issues

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_machine_audit.py tests/test_evaluate_binance_hybrid_prod.py tests/test_binance_deploy_gate.py
```

Result: `64 passed`

Real machine dry-run:

```bash
source .venv313/bin/activate
python scripts/evaluate_binance_hybrid_prod.py --dry-run
```

Now reports both layers:

- runtime contamination from borrowed quote / foreign symbol activity
- machine contamination from `hourly`, `meta_margin`, `selector`, and `worksteal_daily`
- running hybrid process drift: `robust_champion` vs launch `robust_reg_tp005_dd002`

### Post-deploy verification now checks live machine state too

Another gap was that `src.binance_hybrid_prod_verify` only trusted:

- launch file contents
- supervisor log startup text
- recent live cycle snapshots

That was not enough for this machine. We had already observed that a restart can still leave the wrong hybrid process or extra Binance writers active, so deploy verification needed to check the actual process set after restart, not just the artifacts around it.

Implemented:

- `src.binance_hybrid_prod_verify.py` now supports `--require-machine-state-match`
- when enabled, deploy verification also runs the machine audit:
  - process isolation via `src.binance_live_process_audit.py`
  - running hybrid process vs `launch.sh` via `src.binance_hybrid_process_audit.py`
- `scripts/deploy_crypto_model.sh` now enables that flag for both RL deploys and `--remove-rl` deploys

This closes the remaining trust gap where preflight could pass but post-restart verification could still miss:

- extra Binance writers appearing after restart
- the wrong hybrid checkpoint continuing to run after restart

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_prod_verify.py tests/test_deploy_crypto_model.py
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_hybrid_process_audit.py
```

Results:

- `54 passed`
- `8 passed`
