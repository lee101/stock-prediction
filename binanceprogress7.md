# Binance Model Refresh Pass (2026-03-23)

## Runtime Findings

- Current hybrid live allocator is effectively **not trading** on `2026-03-23`.
  - Evidence: `strategy_state/hybrid_trade_cycles/hybrid-cycle_20260323.jsonl`
  - Repeated status: `invalid_allocation_plan_rl_flat`
  - Root cause in log payload:
    - Gemini API error `403 PERMISSION_DENIED`
    - Message: API key was reported as leaked
  - Result: no orders placed, allocation falls back to cash.

- Legacy `binanceneural` live state did lose money before that handoff.
  - Evidence: `strategy_state/binanceneural_pnl_state_live.json`
  - Realized PnL total: `-1232.47`
  - Biggest losers:
    - `DOGEUSDT`: `-670.91`
    - `SOLUSD`: `-391.06`
    - `AAVEUSDT`: `-267.57`
    - `BTCUSD`: `-260.92`

## Validation Harness Work

### Code changes

- `binanceneural.marketsimulator`
  - Added seeded-start support via:
    - `initial_inventory_by_symbol`
    - `initial_cost_basis_by_symbol`
  - Works in both single-symbol and shared-cash simulation.

- New script:
  - `scripts/run_binanceneural_robustness_sweep.py`
  - Evaluates checkpoint candidates across:
    - multiple windows
    - seeded start states
    - decision lags
    - fill-buffer / slippage assumptions
    - action intensity / offset overrides

- `scripts/run_deep_binanceneural_sweep.py`
  - Added modern architecture flags:
    - `--model-arch`
    - `--num-memory-tokens`
    - `--dilated-strides`
    - `--attention-window`
    - `--use-compile`
    - `--use-vectorized-sim`
    - `--accumulation-steps`
    - `--decision-lag-range`
    - `--forecast-horizons`

- `binanceneural/benchmark_training.py`
  - Added matching architecture flags for throughput probes.
  - Enabled TF32 matmul precision so benchmark matches real trainer settings more closely.

### Tests

- `pytest tests/test_marketsimulator.py tests/test_run_binanceneural_robustness_sweep.py -q`
- Result: `56 passed`

## Robustness Baseline

Command run:

```bash
source .venv313/bin/activate
python scripts/run_binanceneural_robustness_sweep.py \
  --checkpoints binanceneural/checkpoints/crypto_portfolio_6sym \
  --sample-epochs 12 \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD \
  --sequence-length 48 \
  --window-hours 168,336,720 \
  --decision-lag-list 1,2 \
  --fill-buffer-bps-list 0,5,10 \
  --output-dir analysis/binanceneural_robustness_crypto_portfolio_6sym
```

Artifacts:

- Summary: `analysis/binanceneural_robustness_crypto_portfolio_6sym/summary.csv`
- Scenarios: `analysis/binanceneural_robustness_crypto_portfolio_6sym/scenarios.csv`

Result for `crypto_portfolio_6sym/epoch_012.pt`:

- `selection_score`: `-133.56`
- `return_mean_pct`: `-8.64%`
- `return_worst_pct`: `-17.60%`
- `max_drawdown_worst_pct`: `18.63%`
- `negative_return_rate`: `1.0`

Key scenario read:

- Worst cluster is the `168h` window with `lag=2` and `fill_buffer_bps=10`.
- Even the best cluster stayed negative:
  - around `-0.36%` return with `lag=1`, `fill_buffer_bps=0`
- Conclusion:
  - this checkpoint is not a promotion candidate
  - do not use it as the “best algorithm” baseline

## Training Efficiency Read

Local GPU: `RTX 3090 Ti 24GB`

Benchmark commands:

```bash
source .venv313/bin/activate
python binanceneural/benchmark_training.py \
  --steps 20 --warmup 5 \
  --batch-size 16 --seq-len 72 \
  --hidden-dim 256 --num-layers 4 --num-heads 8 \
  --use-fast-sim --use-compile

python binanceneural/benchmark_training.py \
  --steps 20 --warmup 5 \
  --batch-size 16 --seq-len 72 \
  --hidden-dim 256 --num-layers 4 --num-heads 8 \
  --model-arch nano --num-memory-tokens 4 \
  --dilated-strides 1,4,24 \
  --use-fast-sim --use-compile

python binanceneural/benchmark_training.py \
  --steps 20 --warmup 5 \
  --batch-size 8 --seq-len 96 \
  --hidden-dim 384 --num-layers 6 --num-heads 8 \
  --model-arch nano --num-memory-tokens 8 \
  --dilated-strides 1,4,24,72 --attention-window 96 \
  --use-fast-sim --use-compile
```

Observed throughput:

| Config | Steps/s | Samples/s | Peak VRAM |
|--------|--------:|----------:|----------:|
| classic `256x4` | `8.13` | `130.14` | `139.1 MB` |
| nano `256x4` + mem4 + dilated | `7.89` | `126.18` | `34.9 MB` |
| nano `384x6` + mem8 + dilated | `5.84` | `46.73` | `71.2 MB` |

Read:

- `256` width is the current sweet spot.
- `384x6` is too expensive for the first search pass.
- Nano is slightly slower than classic at this size on the 3090 Ti, but the VRAM footprint is much lower.
- That means the next sweep should focus on:
  - `192`
  - `256`
  - `320`
  - not `384x6` yet

## Local Proof Sweep

Command running:

```bash
source .venv313/bin/activate
python scripts/run_deep_binanceneural_sweep.py \
  --symbols SOLUSD \
  --dims 192 256 384 \
  --wds 0.03 \
  --lrs 1e-4 \
  --forecast-horizons 1,24 \
  --sequence-length 96 \
  --batch-size 16 \
  --transformer-layers 4 \
  --transformer-heads 8 \
  --validation-days 90 \
  --model-arch nano \
  --num-memory-tokens 4 \
  --dilated-strides 1,4,24 \
  --attention-window 96 \
  --return-weight 0.08 \
  --fill-buffer-pct 0.0005 \
  --decision-lag-bars 1 \
  --decision-lag-range 0,1,2 \
  --loss-type sortino \
  --use-compile \
  --use-vectorized-sim \
  --epochs 4 \
  --patience 2 \
  --results-dir analysis/solusd_modern_size_sweep_local \
  --no-wandb
```

Current status at last check:

- process still active locally
- no checkpoint files written yet
- first config had reached:
  - `Training: 2043 train batches, 136 val batches`

## Remote Status

Attempted remote connectivity checks from this shell did not return usable output:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 \
  'cd /nvme0n1-disk/code/stock-prediction && source .venv313/bin/activate && python -V'
```

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no administrator@93.127.141.100 'echo ok'
```

Result:

- remote training was **not launched** from this shell
- no reproducible remote run id / log path exists yet
- next step is to resolve the SSH/auth/connectivity issue, then launch the `192/256/320` trio sweep remotely

## Audit (2026-04-09)

### Goal

Re-audit what is actually running in Binance production, replay it under production-faithful assumptions, and see whether any existing RL checkpoint or simple production-shape variant can move toward the monthly PnL target.

### What is live right now

- hybrid service: `deployments/binance-hybrid-spot/launch.sh`
- live symbols: `BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD`
- live leverage: `0.5x`
- LLM: `gemini-3.1-flash-lite-preview`
- live RL checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`

### Machine reality check

The launch file is **not** what the running hybrid process is currently executing:

- launch target checkpoint: `mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- running hybrid checkpoint: `a100_scaleup/robust_champion/best.pt`
- running hybrid PID: `1712424`

I also tightened `src/binance_live_process_audit.py` so cache-only helper daemons no longer count as conflicting live writers. After that fix, the remaining real live-writer conflicts on this machine are:

- `binanceleveragesui.trade_margin_meta`
- `binance_worksteal.trade_live`

So production is currently contaminated by both:

- checkpoint drift (`robust_champion` still running instead of launch `dd002`)
- shared-account live-writer overlap

### Replay protocol used

All meaningful comparisons in this pass used:

- `decision_lag=2`
- `slippage_bps=5`
- `fill_buffer_bps=5`
- `fee_rate=0.001`
- long-only
- tradable action mask restricted to the actual live symbol set or tested subset
- data: `pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin`

For multi-month runs, `--no-early-stop` was used to avoid the drawdown-vs-profit early-exit heuristic hiding losses.

### Current production replay

Current live config reproduced as:

| config | med_ret | med_sort | p10_ret | med_dd |
|---|---:|---:|---:|---:|
| live `dd002`, 30-step / 50-window | `-2.90%` | `-3.84` | `-5.94%` | `4.63%` |

Full-span no-early-stop:

| config | total_ret | sortino | max_dd |
|---|---:|---:|---:|
| live `dd002` | `-24.07%` | `-2.19` | `33.65%` |

### Candidate checkpoint sweep

Best six-symbol 30-step candidates in this pass:

| checkpoint | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| `gspo_like_drawdown_mix15_tp01` | `-1.26%` | `-1.48` | `-10.25%` |
| `gspo_like_drawdown_mix15` | `-1.38%` | `-2.47` | `-6.30%` |
| `reg_combo_2` | `-2.36%` | `-2.54` | `-11.74%` |
| `gspo_like_smooth_mix15` | `-2.38%` | `-2.05` | `-9.29%` |
| live `dd002` | `-2.90%` | `-3.84` | `-5.94%` |

Best full-span no-early-stop checkpoint in this pass:

| checkpoint | total_ret | sortino | max_dd |
|---|---:|---:|---:|
| `gspo_like_smooth_mix15` | `-5.75%` | `-0.18` | `18.84%` |

Conclusion: no existing six-symbol checkpoint is close to a deployable `27%` monthly target.

### Subset search

Searching all subsets of the six live symbols on `gspo_like_smooth_mix15` showed the mixed basket is the main problem:

| subset | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| `ETHUSD` | `-0.06%` | `-3.46` | `-2.10%` |
| `BTCUSD,ETHUSD` | `-0.19%` | `-3.86` | `-2.53%` |
| `BTCUSD` | `-0.26%` | `-3.68` | `-1.64%` |

That does not get the 30-step median positive, but it is a large improvement versus the live six-symbol basket.

`followup_tp01` did not beat this on subset search. Its best subset was `DOGEUSD,AAVEUSD`, still negative at `-0.77%` median.

### Leverage replay bug found and fixed

Bug:

- `pufferlib_market/hourly_replay.py`
- `_open_long_limit` was clamping long notional back to available cash
- any `max_leverage > 1.0` was effectively ignored for long-only limit-entry runs

Fix:

- remove the long-side cash clamp
- keep negative cash as borrowed margin, which already matches the equity accounting path
- add regression coverage in `tests/test_pufferlib_market_hourly_replay.py`

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_pufferlib_market_hourly_replay.py tests/test_evaluate_holdout.py
```

Result: `30 passed`

### Best concentrated result after the leverage fix

Checkpoint:

- `pufferlib_market/checkpoints/mixed23_a40_sweep/gspo_like_smooth_mix15/best.pt`

Tradable subset:

- `ETHUSD`

Full-span no-early-stop:

| leverage | total_ret | sortino | max_dd |
|---|---:|---:|---:|
| `0.5x` | `+2.87%` | `1.90` | `2.18%` |
| `1.0x` | `+5.61%` | `1.91` | `4.35%` |
| `2.0x` | `+10.68%` | `1.91` | `8.67%` |

30-step / 50-window robustness:

| leverage | med_ret | p10_ret |
|---|---:|---:|
| `0.5x` | `-0.06%` | `-2.10%` |
| `1.0x` | `-0.11%` | `-4.21%` |
| `2.0x` | `-0.23%` | `-8.41%` |

Interpretation:

- this is the best production-shaped result found in this session
- it is still nowhere near `27%` monthly
- full-span PnL turns positive, but the short-window median stays negative and worsens with leverage

### Follow-up candidate sweep (2026-04-09 later)

Artifacts:

- `analysis/binance_prod_live6_23sym_candidates_20260409/`
- `analysis/per_symbol_all23_holdout_20260409/`
- `analysis/live6_top_singletons_fullspan_20260409/`
- `analysis/per_symbol_all23_fullspan_2x_20260409/`

Fresh six-symbol 30-step / 50-window rerun across the main 23-symbol-compatible checkpoints:

| checkpoint | med_ret | med_sort | p10_ret | med_dd |
|---|---:|---:|---:|---:|
| `per_env_adv_smooth` | `-2.08%` | `-2.39` | `-10.23%` | `5.92%` |
| `reg_combo_2` | `-2.36%` | `-2.54` | `-11.74%` | `6.85%` |
| `gspo_like_smooth_mix15` | `-2.38%` | `-2.05` | `-9.29%` | `6.31%` |
| launch `dd002` | `-2.90%` | `-3.84` | `-5.94%` | `4.63%` |
| actual running `robust_champion` | `-4.48%` | `-6.17` | `-8.57%` | `5.17%` |

Takeaways:

- `per_env_adv_smooth` slightly improved the six-symbol median, but still stayed negative and is not a promotion candidate
- the actual running checkpoint `robust_champion` is worse than the launch target on the current production-shaped slice

Best meaningful concentrated full-span results at `2.0x` leverage:

| checkpoint | subset | total_ret | sortino | max_dd |
|---|---|---:|---:|---:|
| `gspo_like_smooth_mix15` | `ETHUSD` | `+10.68%` | `1.91` | `8.67%` |
| `gspo_like_smooth_mix15` | `BTCUSD` | `+7.19%` | `1.03` | `15.66%` |
| `gspo_like_smooth_mix15` | `DOGEUSD` | `+5.76%` | `0.86` | `43.27%` |

Important negatives from the same pass:

- launch `dd002` on `AAVEUSD`: `-10.36%`, Sortino `-1.14`, DD `17.65%` at `0.5x`
- `robust_champion` on `BTCUSD`: flat short-window median was a no-trade artifact; full-span `2.0x` still came out `-10.06%`

Conclusion: the best paper lane remains concentrated `gspo_like_smooth_mix15` on `ETHUSD`, but nothing tested here is remotely close to the `27%` monthly target under live-like constraints.

### Next loop

1. Train and validate specifically for `ETHUSD` or `BTCUSD,ETHUSD` deployment masks instead of hoping a mixed 23-symbol policy survives masking.
2. Keep `decision_lag=2` and live symbol masking as hard constraints in every gate.
3. Compare any new candidate against:
   - live `dd002`
   - `gspo_like_smooth_mix15`
   - `gspo_like_smooth_mix15` masked to `ETHUSD`
4. Do not promote any checkpoint to live unless both:
   - 30-step median replay is clearly positive
   - full-span no-early-stop replay remains positive under the same mask and leverage

## Audit (2026-04-09)

Investigated the recent complaint that Binance production was losing money even though the market had been broadly up.

Conclusion:

- this was not a clean "hybrid model underperformed an up tape" case
- the Binance margin account is currently shared across multiple live trading processes
- hybrid runtime snapshots show account contamination severe enough to invalidate direct sim-vs-live comparison

Live processes found:

- `rl_trading_agent_binance/trade_binance_live.py --live ... --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD --execution-mode margin --leverage 0.5`
- `python -m binanceleveragesui.trade_margin_meta ... --doge-checkpoint ... --aave-checkpoint ... --max-leverage 2.30`
- `python -m binanceexp1.trade_binance_selector --symbols BTCUSD,ETHUSD,SOLUSD ...`
- `python -m binanceexp1.trade_binance_hourly --symbols SOLUSD ...`

Runtime audit over `2026-04-08T00:00:00Z` to `2026-04-08T22:00:00Z`:

- `22` allocation snapshots
- `12` snapshots with unexpected borrowed quote on a `0.5x` launch
- `AAVEUSD` oversized in `6` managed-position snapshots
- `AAVEUSD` oversized in `15` managed-order buckets
- unexpected foreign activity also present: `DOTUSD`, `XRPUSDT`

Concrete contaminated snapshot:

- `2026-04-08T19:09:05Z`
- equity about `$3078`
- borrowed quote about `$6265`
- `AAVE` position `79.21747868`
- hybrid target sleeve for `AAVEUSD` only about `$231`

Interpretation:

- the live account was carrying stale / foreign leveraged AAVE exposure while the hybrid loop was trying to rebalance an equal-weight basket
- the market itself was positive over the last completed 24 hourly bars (`BTC +2.96%`, `ETH +4.46%`, `AAVE +4.18%`, `LINK +2.98%`, `SOL +1.65%`, `DOGE +1.10%`)
- therefore the observed production drawdown was dominated by shared-account state and execution contamination, not just forecast quality

Code change made:

- extended `src/binance_hybrid_runtime_audit.py` to flag:
  - borrowed quote exposure beyond launch leverage
  - oversized managed positions versus planned sleeve
  - oversized managed orders versus planned sleeve

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_runtime_audit.py
```

Result: `9 passed`

Next action:

- isolate the Binance account / processes before trusting any further simulator-vs-production PnL comparison

## Live guard added

Implemented the next highest-value production improvement in `rl_trading_agent_binance/trade_binance_live.py`:

- added an account contamination guard ahead of context collection / Gemini / order placement
- blocks live cycles with `status=account_guard_blocked` when the account shows:
  - borrowed quote beyond the configured leverage budget
  - foreign positions or foreign working orders outside the configured symbol set
  - single-symbol exposure larger than the hybrid gross budget implied by `effective_leverage`

Why this matters:

- the main observed production failure mode was shared-account contamination, not just weak signal quality
- without a live guard, the hybrid loop could keep placing new orders on top of an already-invalid state
- with the guard, the runtime now fails closed and the production verifier/runtime audit treat that state as unhealthy

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_hybrid_allocation_prompt.py tests/test_binance_hybrid_prod_verify.py tests/test_binance_hybrid_runtime_audit.py tests/test_evaluate_binance_hybrid_prod.py
```

Result: `87 passed`

## Deploy-process isolation guard

Next high-value gap observed after the runtime contamination audit:

- the live hybrid loop could now block on a dirty account
- but the deploy/autodeploy path could still approve a checkpoint while other Binance live writers were already running on the same machine

Implemented:

- added `src/binance_live_process_audit.py`
- audits the specific live Binance writers actually observed here:
  - hybrid allocator
  - meta-margin writer
  - selector writer
  - hourly writer
  - worksteal daily writer
- `src/binance_deploy_gate.py` now supports `require_process_isolation=True`
- `scripts/binance_autoresearch_forever.py` now:
  - requires process isolation when auto-deploy is active
  - blocks `deploy_candidate_checkpoint()` before it calls the deploy shell script if conflicting Binance writers are already running

Why this matters:

- without it, the system could still decide "deploy this winner" in exactly the contaminated multi-writer environment that invalidated the original sim-vs-live read
- with it, deploy decisions now fail closed at preflight time instead of waiting for the runtime guard to block after launch

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_deploy_gate.py tests/test_binance_autoresearch_forever.py tests/test_search_binance_prod_config_grid.py
```

Result: `171 passed`

## Running hybrid process drift

Next high-value issue observed directly on the machine:

- the running hybrid PID did not actually match `deployments/binance-hybrid-spot/launch.sh`
- launch file checkpoint: `mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- running hybrid checkpoint: `a100_scaleup/robust_champion/best.pt`

That means "launch config" and "actual live process" had drifted apart, which undermines both diagnosis and deployment safety.

Implemented:

- extracted a reusable parser for `trade_binance_live.py` command lines in `src/binance_hybrid_launch.py`
- added `src/binance_hybrid_process_audit.py`
- `scripts/binance_autoresearch_forever.py` now blocks real deploy attempts if the current running hybrid process does not match the launch script

Why this matters:

- previously, auto-deploy could operate on the assumption that the current launch file described the live system
- on this box that assumption was false
- now automated deploys fail closed when the live hybrid process is already stale

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_process_audit.py tests/test_binance_autoresearch_forever.py tests/test_evaluate_binance_hybrid_prod.py tests/test_binance_deploy_gate.py
```

Result: `167 passed`

Machine sanity check with the new audit returned:

- `running hybrid process does not match launch config: rl_checkpoint`

## Manual deploy preflight aligned

Review pass found that `scripts/deploy_crypto_model.sh` still bypassed the new live-process safety checks.

Fixed:

- `src/binance_live_process_audit.py` now classifies only actual Python writer processes instead of doing broad substring matches
- `scripts/deploy_crypto_model.sh` now runs the same live-process preflight before a real deploy
  - blocks on conflicting Binance live writers
  - blocks on running-hybrid-process drift from `launch.sh`

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_hybrid_process_audit.py tests/test_deploy_crypto_model.py
```

Result: `39 passed`

## Production baseline now includes machine-state audit

Next high-value issue:

- `scripts/evaluate_binance_hybrid_prod.py` only captured runtime snapshot drift/health
- it did not record the machine-level failures we already knew were happening:
  - concurrent Binance live writers
  - running hybrid process checkpoint drift from `launch.sh`

Implemented:

- added `src/binance_hybrid_machine_audit.py`
- production eval now writes:
  - `current_machine_audit`
  - `current_machine_audit_issues`
  - `current_machine_health_issues`
- for the real default launch, those machine issues are merged into the manifest’s baseline issue fields so downstream deploy gating sees the real baseline, not just the snapshot layer
- `src/binance_deploy_gate.py` now also blocks manifests that declare machine baseline issues

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_machine_audit.py tests/test_evaluate_binance_hybrid_prod.py tests/test_binance_deploy_gate.py
```

Result: `64 passed`

Real machine dry-run now reports:

- runtime drift from borrowed quote / foreign symbol contamination
- machine health issues from `hourly`, `meta_margin`, `selector`, and `worksteal_daily`
- hybrid process checkpoint drift from launch config

## Deploy verification now enforces actual post-restart machine state

Observed gap:

- deploy preflight already blocked conflicting writers and launch/process drift
- but post-deploy verification still only trusted launch text, supervisor logs, and snapshots
- on this machine that was not enough, because a restart can still leave the wrong process set active

Implemented:

- `src.binance_hybrid_prod_verify.py` now supports `--require-machine-state-match`
- when enabled it runs the machine audit during verification and blocks on:
  - conflicting Binance live writers
  - running hybrid process drift vs `launch.sh`
- `scripts/deploy_crypto_model.sh` now uses this stricter verification path for both normal deploys and `--remove-rl`

Added coverage for:

- verifier pass with matching machine state
- verifier block on checkpoint drift after deploy
- verifier block on conflicting Binance writers after deploy
- deploy-script rollback when preflight passes but post-restart machine state becomes contaminated

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_prod_verify.py tests/test_deploy_crypto_model.py
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_hybrid_process_audit.py
```

Results:

- `54 passed`
- `8 passed`
