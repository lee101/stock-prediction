# 2026-05-04 XGB Stock Optimization + Codex Monitor Test

## Scope

- Trading target is Alpaca stocks, not Binance.
- Production deploy gate remains median monthly return >= 27% on honest
  heldout/stress simulation with realistic fills and drawdown/tail controls.
- No live writer changes or order placement are allowed from this experiment
  unless a candidate clearly clears the gate.
- Existing dirty file `manifest_stocks_drytest_001.json` is unrelated and left
  untouched.

## 2026-05-04 Run Log

### Initial state

- Starting from commit `b7f9d01b` (`Guard aggressive XGB live buy sizing`).
- Active production remains documented in `alpacaprod.md`; no strategy
  promotion has happened in this run.
- Planned experiment tracks:
  - Stress-first hyperparameter sweeps around the top-1/top-N XGB stock family.
  - Correlation-aware and work-stealing allocation checks.
  - Codex scheduled production monitor wrapper smoke test.

### Sweep 1 partial: packing + work-stealing allocation

- Command family:
  `xgbnew.sweep_ensemble_grid` on `stocks_wide_1000_v1`,
  `heldout2025h2_xgb` 5-seed ensemble, OOS `2025-07-01 -> 2026-04-17`,
  `stress36x`, 10 bps fill buffer, hold-through, top-N `{1,2,3}`,
  leverage `{1.5,2.0,2.25}`, min score `{0.55,0.60}`, allocation modes
  `{equal,score_norm,softmax,worksteal}`, overnight cap `2.0`.
- Artifact:
  `analysis/xgbnew_daily/alpacaprod8_pack_worksteal_20260504/sweep_20260504_115149.partial.json`
- Stopped after 200 completed cells because all tested cells had already
  fail-fast pruned; no cell survived the stress gate.
- Best partial median cell was `top_n=3`, `leverage=1.5`, `min_score=0.60`,
  `allocation=worksteal`, `max_vol_20d=1.5`: median `+11.22%/mo`,
  p10 `-10.65%`, worst monthly `-18.16%`, worst DD `25.95%`,
  failed after one negative window.
- Interpretation: naive packing is not enough. Work-stealing improves top-2/3
  versus equal sizing, but the tail window is still far below deployable.

### Sweep 2: correlation-aware work-stealing packing

- Artifact:
  `analysis/xgbnew_daily/alpacaprod8_corr_worksteal_small_20260504/sweep_20260504_122103.json`
- Grid: top-N `{2,3}`, leverage `{1.5,1.75}`, work-stealing allocation,
  skew-gated regime (`regime_cs_skew_min=0.5`), `regime_cs_iqr_max=0.06`,
  and trailing correlation filters (`corr_window_days in {0,40}`,
  `corr_max_signed in {0.50,0.75}`).
- Result: `0/16` cells survived fail-fast.
- Best median cell: `top_n=3`, `leverage=1.75`, median `+10.37%/mo`,
  p10 `-5.57%` without the correlation filter and p10 `-3.51%` with
  40-day correlation filtering; worst DD about `14.2%`.
- Interpretation: correlation filtering reduced the bad tail on the packed
  top-3 case, but median remains far below the 27% target and still has a
  negative window.

### Sweep 3: opportunistic work-stealing entries

- Artifact:
  `analysis/xgbnew_daily/alpacaprod8_opportunistic_20260504/sweep_20260504_123042.json`
- Grid: top-N `{1,2}`, leverage `{1.75,2.0}`, watch-N `{3,5,10}`,
  entry discount `{20,40}` bps below open, work-stealing allocation,
  skew/IQR regime filters, `stress36x`, 10 bps fill buffer.
- Result: `0/24` cells survived fail-fast.
- Best median cell: `top_n=1`, `leverage=2.0`, watch-N `3/5/10`,
  20 bps entry discount, median `+16.90%/mo`, p10 `-5.27%`,
  worst monthly `-11.66%`, worst DD `17.40%`.
- Interpretation: opportunistic limit-style entry improved the median versus
  correlation-packed work-stealing while keeping drawdown reasonable, but it
  still failed the negative-window gate and remains below the deploy target.

### Deploy decision after sweeps

- No strategy candidate cleared the production gate.
- No XGB deployment or threshold/leverage promotion from these sweeps.
- Best direction for the next round is the opportunistic top-1 path, but it
  needs a tail-risk/regime fix before it is production material.

### Monitoring prompt fix

- `monitoring/codex_prod_check_prompt.md` had stale assumptions from the
  older XGB-live layout. It now defers to `alpacaprod.md` and explicitly
  accepts the current manual `trading-server` + `daily-rl-trader` production
  mode, with XGB intentionally inert.
- Upgraded local Codex CLI from `0.56.0` to `0.128.0` so `gpt-5.5` is
  accepted.
- Updated `monitoring/codex_prod_check.sh` and `monitoring/monitor_agent.sh`
  for the current CLI flag:
  `--dangerously-bypass-approvals-and-sandbox` instead of obsolete `--yolo3`.
- Re-authenticated headless Codex with the machine `OPENAI_API_KEY` because
  the prior ChatGPT session returned 401 from the new CLI.
- Made both monitor wrappers source `~/.secretbashrc` under `set +e` so the
  profile does not break strict-mode wrapper execution.

### Real Codex monitor test

- Command: `CODEX_BIN=/home/administrator/.bun/bin/codex monitoring/codex_prod_check.sh`
- Successful wrapper artifact:
  `monitoring/logs/codex_prod_20260504T124713Z.log`
- Machine-readable current status:
  `monitoring/logs/codex_current.log` now reports `status=OK rc=0`.
- The scheduled Codex agent appended:
  `monitoring/logs/codex_prod_20260504.log`
- Agent verdict: `YELLOW`.
- Production checks from the agent:
  - `trading-server` pid `4130990` owns the singleton lock as
    `alpaca_wrapper_4130990`.
  - `daily-rl-trader` pid `4131404` is attached through `127.0.0.1:8050`.
  - Alpaca account active, `trading_blocked=false`, equity/cash `$19,799.53`,
    buying power `$39,599.06`.
  - `0` open orders and `0` material positions.
  - No services were started/stopped, no orders submitted, and no
    model/config/leverage/threshold changes made.
- Follow-up health check:
  `.venv313/bin/python monitoring/health_check.py --json` exits `0`;
  scheduled-audits is now OK for Codex. Remaining warnings are optional
  services and disk (`/` 94%, `/nvme0n1-disk` 86%).
