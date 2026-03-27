# Binance Pure Crypto Optimization (2026-03-15)

## Objective

Find the **highest Sortino, lowest max-drawdown, best annualized PnL** strategy for pure crypto trading on Binance. This will run live on Binance — every decision here must be validated in a market simulator that closely matches real execution.

## 2026-03-26 Replay-Combo Ranking Pass

This pass tightened the short-budget search loop around the actual decision
criteria we care about for Binance deployment: daily profitability is not
enough; the candidate also needs to survive hourly replay with acceptable
drawdown and a smoother equity path.

### What changed

- `pufferlib_market.replay_eval` now exports per-section:
  - `pnl_smoothness`
  - `ulcer_index`
  - `goodness_score`
- `pufferlib_market.autoresearch_rl` now derives `replay_combo_score` from:
  - daily replay return / Sortino / max drawdown / smoothness
  - hourly replay return / Sortino / max drawdown / smoothness
  - optional hourly-policy stress metrics when present
- `--rank-metric replay_combo_score` is now available, and `--rank-metric auto`
  prefers it whenever replay metrics exist.

### Repro command

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD/PufferLib:$PYTHONPATH
python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 180 --max-trials 2 \
  --descriptions reg_combo_2,gspo_like_drawdown_mix15 \
  --periods-per-year 365 --max-steps-override 90 \
  --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
  --holdout-eval-steps 90 --holdout-n-windows 12 \
  --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5 \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-fill-buffer-bps 5 \
  --rank-metric replay_combo_score \
  --leaderboard analysis/binance_replay_combo_probe_20260326.csv \
  --checkpoint-root pufferlib_market/checkpoints/binance_replay_combo_probe_20260326
```

### Artifacts

- `analysis/binance_replay_combo_probe_20260326.csv`
- `pufferlib_market/checkpoints/binance_replay_combo_probe_20260326/reg_combo_2/replay_eval.json`
- `pufferlib_market/checkpoints/binance_replay_combo_probe_20260326/gspo_like_drawdown_mix15/replay_eval.json`
- `pufferlib_market/checkpoints/binance_replay_combo_probe_20260326/*/holdout_summary.json`

### Results

| Config | Replay Combo Score | Holdout Robust | Daily Replay Return | Hourly Replay Return | Daily MaxDD | Hourly MaxDD |
|--------|-------------------:|---------------:|--------------------:|---------------------:|------------:|-------------:|
| `gspo_like_drawdown_mix15` | `-130.97` | `-126.29` | `-14.69%` | `-15.21%` | `29.82%` | `33.14%` |
| `reg_combo_2` | `-201.97` | `-213.00` | `-37.07%` | `-32.23%` | `41.47%` | `43.54%` |

### Current read

- The new replay-combo metric did its job: it ranked the GSPO-style drawdown
  branch above `reg_combo_2` because it was materially less bad on both replay
  horizons and drawdown.
- Neither candidate is promotable:
  - both remained negative on daily replay
  - both remained negative on hourly replay
  - both posted large replay drawdowns
- Practical takeaway: keep using **short 3-minute bounded runs**, but rank by
  `replay_combo_score` instead of raw hourly return so the search loop stops
  over-favoring strategies that only look good on one horizon.
- Immediate next branch to test:
  - keep the GSPO family
  - push more friction / downside control
  - only continue candidates that improve the replay-combo score while getting
    hourly replay return back toward non-negative territory

## 2026-03-26 GSPO Follow-Up Sweep

The next pass stayed inside the same GSPO drawdown family and changed one knob
at a time, so the failures would be interpretable.

### Repro command

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD/PufferLib:$PYTHONPATH
python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 180 --max-trials 5 \
  --descriptions gspo_like_drawdown_mix15_slip12,gspo_like_drawdown_mix15_tp01,gspo_like_drawdown_mix15_dd03,gspo_like_drawdown_mix15_sds03,gspo_like_drawdown_mix15_h512 \
  --periods-per-year 365 --max-steps-override 90 \
  --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
  --holdout-eval-steps 90 --holdout-n-windows 12 \
  --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5 \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-fill-buffer-bps 5 \
  --rank-metric replay_combo_score \
  --leaderboard analysis/binance_replay_combo_followup_20260326.csv \
  --checkpoint-root pufferlib_market/checkpoints/binance_replay_combo_followup_20260326
```

### Artifacts

- `analysis/binance_replay_combo_followup_20260326.csv`
- `pufferlib_market/checkpoints/binance_replay_combo_followup_20260326/*/replay_eval.json`
- `pufferlib_market/checkpoints/binance_replay_combo_followup_20260326/*/holdout_summary.json`

### Results

| Config | Replay Combo Score | Val Return | Holdout Robust | Daily Replay Return | Hourly Replay Return | Daily MaxDD | Hourly MaxDD |
|--------|-------------------:|-----------:|---------------:|--------------------:|---------------------:|------------:|-------------:|
| `gspo_like_drawdown_mix15_dd03` | `+7.94` | `-7.96%` | `-172.59` | `+8.05%` | `+20.32%` | `15.21%` | `17.53%` |
| `gspo_like_drawdown_mix15_tp01` | `-14.37` | `+11.22%` | `-146.12` | `+6.51%` | `+8.85%` | `23.59%` | `28.59%` |
| `gspo_like_drawdown_mix15_slip12` | `-89.39` | `-1.41%` | `-222.13` | `-23.69%` | `+41.08%` | `44.84%` | `22.50%` |
| `gspo_like_drawdown_mix15_h512` | `-126.64` | `-0.65%` | `-77.77` | `-15.18%` | `-11.45%` | `28.58%` | `31.54%` |
| `gspo_like_drawdown_mix15_sds03` | `-151.73` | `+12.00%` | `-179.49` | `-20.96%` | `-21.23%` | `32.76%` | `36.37%` |

### Read

- Two branches are worth keeping:
  - `gspo_like_drawdown_mix15_dd03`: best deployment-shaped replay score so far
  - `gspo_like_drawdown_mix15_tp01`: cleaner balanced branch with positive val and positive replay on both horizons
- `slip12` improved hourly replay aggressively but broke the daily replay and
  holdout shape too much to stand alone.
- `sds03` and `h512` should not be prioritized further.

## 2026-03-26 GSPO Combo Sweep

After the follow-up pass, the only justified combo was to merge the two knobs
that had actually helped: `tp01` and `dd03`, then test a slightly higher
friction version.

### Repro command

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD/PufferLib:$PYTHONPATH
python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 180 --max-trials 2 \
  --descriptions gspo_like_drawdown_mix15_tp01_dd03,gspo_like_drawdown_mix15_tp01_dd03_slip10 \
  --periods-per-year 365 --max-steps-override 90 \
  --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
  --holdout-eval-steps 90 --holdout-n-windows 12 \
  --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5 \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-fill-buffer-bps 5 \
  --rank-metric replay_combo_score \
  --leaderboard analysis/binance_replay_combo_combo_20260326.csv \
  --checkpoint-root pufferlib_market/checkpoints/binance_replay_combo_combo_20260326
```

### Artifacts

- `analysis/binance_replay_combo_combo_20260326.csv`
- `pufferlib_market/checkpoints/binance_replay_combo_combo_20260326/*/replay_eval.json`
- `pufferlib_market/checkpoints/binance_replay_combo_combo_20260326/*/holdout_summary.json`

### Results

| Config | Replay Combo Score | Val Return | Holdout Robust | Daily Replay Return | Hourly Replay Return | Daily MaxDD | Hourly MaxDD |
|--------|-------------------:|-----------:|---------------:|--------------------:|---------------------:|------------:|-------------:|
| `gspo_like_drawdown_mix15_tp01_dd03_slip10` | `+19.91` | `+10.06%` | `-183.88` | `+26.66%` | `+19.98%` | `29.83%` | `28.48%` |
| `gspo_like_drawdown_mix15_tp01_dd03` | `-165.43` | `+3.66%` | `-243.32` | `-26.33%` | `-26.20%` | `31.47%` | `34.33%` |

### Current read

- Best new branch so far: `gspo_like_drawdown_mix15_tp01_dd03_slip10`
  - first positive replay-combo score with both replay horizons strongly positive
  - also keeps validation positive
- Key lesson: `tp01` + `dd03` **needs extra friction**. The plain combo failed
  badly, while `slip10` turned it into the best replay-ranked candidate of the
  day.
- Remaining weakness:
  - holdout robustness is still too negative for promotion
- Next sensible direction:
  - keep the `tp01_dd03_slip10` branch
  - test nearby friction / holdout stabilizers rather than broad random search

## 2026-03-16 Mixed23 Daily Robustness Pass

This pass focused on the **daily mixed23 RL stack** under more realistic replay
semantics, not the older pure-crypto hourly plan above.

### What changed

- Daily replay now requires bars to trade **through** a limit by `5bp` before fill.
- The same fill rule is wired through `replay_eval`, `evaluate_holdout`,
  `evaluate_tail`, `evaluate_multiperiod`, and `autoresearch_rl`.
- Added `pufferlib_market/meta_replay_eval.py` to test adaptive checkpoint
  switching without lookahead.

### Repro commands

#### 1. Re-export the latest 60-day mixed23 validation window

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -m pufferlib_market.export_data_daily \
  --symbols AAPL,NFLX,NVDA,ADBE,ADSK,COIN,GOOG,MSFT,PYPL,SAP,TSLA,BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,AAVEUSD,UNIUSD,DOTUSD,SHIBUSD,XRPUSD \
  --output /tmp/mixed23_val_60d_20251208_20260205.bin \
  --start-date 2025-12-08 --end-date 2026-02-05 --min-days 60
```

#### 2. Replay the current checkpoints with 5bp fill-through

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -m pufferlib_market.replay_eval \
  --checkpoint pufferlib_market/checkpoints/mixed23_fresh_replay/ent_anneal/best.pt \
  --daily-data-path /tmp/mixed23_val_60d_20251208_20260205.bin \
  --hourly-data-root trainingdatahourly \
  --start-date 2025-12-08 --end-date 2026-02-05 \
  --max-steps 59 --fill-buffer-bps 5 --deterministic --cpu \
  --output-json pufferlib_market/replay_eval_5bp_60d/ent_anneal.json
```

Repeat for:
- `pufferlib_market/checkpoints/mixed23_fresh_targeted/reg_combo_2/best.pt`
- `pufferlib_market/checkpoints/mixed23_fresh_replay/wd_01/best.pt`
- `pufferlib_market/checkpoints/mixed23_fresh_replay/clip_vloss/best.pt`

#### 3. Test the best current-window adaptive selector

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -m pufferlib_market.meta_replay_eval \
  --checkpoint pufferlib_market/checkpoints/mixed23_fresh_replay/ent_anneal/best.pt \
  --checkpoint pufferlib_market/checkpoints/mixed23_fresh_targeted/reg_combo_2/best.pt \
  --checkpoint pufferlib_market/checkpoints/mixed23_fresh_replay/wd_01/best.pt \
  --labels ent_anneal,reg_combo_2,wd_01 \
  --daily-data-path /tmp/mixed23_val_60d_20251208_20260205.bin \
  --hourly-data-root trainingdatahourly \
  --start-date 2025-12-08 --end-date 2026-02-05 \
  --max-steps 59 --fill-buffer-bps 5 \
  --lookback-days 14 --metric return \
  --selection-mode sticky --switch-margin 0.01 \
  --recency-halflife-days 5 \
  --deterministic --cpu \
  --output-json pufferlib_market/meta_replay_5bp_60d/sticky_return_lb14_hl5_sm001.json
```

## 2026-03-26 Replay Robust-Start + GPU Pool Batch Wiring

This pass tightened the infrastructure around the actual Binance decision loop:
the replay scorer can now stress a candidate from non-flat starting inventory,
and the GPU-pool launcher can submit train-then-replay batches on the same
entrypoint instead of forcing a separate manual replay step.

### What changed

- `pufferlib_market.hourly_replay` now supports explicit initial portfolio
  states for:
  - `simulate_daily_policy`
  - `replay_hourly_frozen_daily_actions`
  - `simulate_hourly_policy`
- `pufferlib_market.replay_eval` now supports:
  - `--robust-start-states`
  - automatic `max_steps` clamp to dataset length
  - `robust_start_summary` output covering worst/median replay behavior
- `pufferlib_market.autoresearch_rl` now:
  - parses robust replay summary fields into leaderboard metrics
  - can rank on worst robust replay returns when requested
  - clamps replay-eval `--max-steps` to the replay dataset length
  - creates nested leaderboard parent directories automatically
- `pufferlib_market.gpu_pool_rl` now forwards replay-eval settings directly to
  autoresearch runs, including:
  - replay data path
  - hourly root
  - replay date range
  - hourly-policy replay stress mode
  - robust start states
  - replay fill buffer and periods/year

### Why this matters

- The current replay path previously assumed a flat start, which can overstate
  deployment quality for a live bot that often begins the next cycle already
  holding risk.
- The GPU-pool launcher previously only handled training-side search knobs.
  That was not enough for the intended workflow of:
  - short train
  - replay/market-sim rank
  - multi-seed compare
  - longer follow-up only for the strongest branch
- The local `hftraining/` stack is still a custom “HF-style” trainer rather
  than a real `transformers.Trainer` entrypoint. I checked the current official
  Hugging Face Trainer/examples path and the right next step is to add a clean
  compare harness instead of mutating the wrong legacy script in place.

### Verified locally

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH
pytest -q \
  tests/test_pufferlib_market_replay_eval.py \
  tests/test_pufferlib_market_hourly_replay_initial_state.py \
  tests/test_pufferlib_market_autoresearch_rl.py \
  tests/test_gpu_pool_rl.py
```

Result:

- `65 passed`

### GPU pool dry-run repro

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH
python -m pufferlib_market.gpu_pool_rl run \
  --dry-run \
  --experiment binance_batch_probe \
  --gpu a100 \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 300 \
  --max-trials 4 \
  --rank-metric replay_combo_score \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-run-hourly-policy \
  --replay-eval-robust-start-states flat,long:BTCUSD:0.25 \
  --replay-eval-fill-buffer-bps 5 \
  --replay-eval-hourly-periods-per-year 8760
```

Dry-run result:

- accepted all new replay arguments
- estimated A100 spend for one 5-minute proof batch: about `$0.69`

### Safe git-sync status

- `git fetch --all --prune` completed
- local branch state after fetch:
  - `main` is `ahead 1, behind 49`
- a plain `git pull` is not safe in this checkout because the worktree already
  has active Binance/RL modifications plus untracked audit/test files
- practical takeaway:
  - use this checkout for the current work
  - integrate `origin/main` as a deliberate merge/rebase task after these local
    Binance changes are either committed or split into a worktree

### Next concrete batch

- Use the updated GPU-pool entrypoint for a short replay-ranked proof batch.
- Run 3 seeds around the current best replay branch plus 1 conservative branch.
- Prefer robust replay ranking for live-shape filtering:
  - `replay_combo_score` for general compare
  - `replay_hourly_policy_robust_worst_return_pct` when inventory stress should
    dominate the shortlist
- Only after one branch survives robust replay plus holdout should we spend
  time wiring the true latest Hugging Face Trainer compare path into the same
  experiment table.

#### 4. Re-test that same selector on earlier 60-day windows

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -m pufferlib_market.export_data_daily \
  --symbols AAPL,NFLX,NVDA,ADBE,ADSK,COIN,GOOG,MSFT,PYPL,SAP,TSLA,BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,AAVEUSD,UNIUSD,DOTUSD,SHIBUSD,XRPUSD \
  --output /tmp/mixed23_val_60d_20251009_20251207.bin \
  --start-date 2025-10-09 --end-date 2025-12-07 --min-days 60

PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -m pufferlib_market.export_data_daily \
  --symbols AAPL,NFLX,NVDA,ADBE,ADSK,COIN,GOOG,MSFT,PYPL,SAP,TSLA,BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,AAVEUSD,UNIUSD,DOTUSD,SHIBUSD,XRPUSD \
  --output /tmp/mixed23_val_60d_20250810_20251008.bin \
  --start-date 2025-08-10 --end-date 2025-10-08 --min-days 60
```

Then rerun the same `meta_replay_eval` command with:
- `--daily-data-path /tmp/mixed23_val_60d_20251009_20251207.bin --start-date 2025-10-09 --end-date 2025-12-07`
- `--daily-data-path /tmp/mixed23_val_60d_20250810_20251008.bin --start-date 2025-08-10 --end-date 2025-10-08`

### Latest-window results (`2025-12-08..2026-02-05`)

| Strategy | Daily Return | Daily MaxDD | Hourly Replay Return | Hourly Replay MaxDD |
|----------|--------------|-------------|----------------------|---------------------|
| `ent_anneal` | `+62.37%` | `15.54%` | `+8.02%` | `34.77%` |
| `reg_combo_2` | `+30.24%` | `21.59%` | `+27.53%` | `34.21%` |
| `wd_01` | `+13.25%` | `20.19%` | `+11.11%` | `26.52%` |
| `clip_vloss` | `-23.73%` | `27.01%` | `-11.76%` | `22.01%` |
| `meta sticky return 14/5 + sm=0.01` | `+74.26%` | `13.83%` | `+41.27%` | `24.86%` |

### 3-window robustness check

Saved artifacts:
- `pufferlib_market/replay_eval_5bp_60d/*.json`
- `pufferlib_market/meta_replay_5bp_60d/*.json`
- `pufferlib_market/meta_replay_5bp_3window_sweep.csv`
- `pufferlib_market/mixed23_3window_strategy_summary.csv`

| Strategy | `2025-12-08..2026-02-05` Hourly | `2025-10-09..2025-12-07` Hourly | `2025-08-10..2025-10-08` Hourly |
|----------|----------------------------------|----------------------------------|----------------------------------|
| `ent_anneal` | `+8.02%` | `-42.43%` | `-9.51%` |
| `reg_combo_2` | `+27.53%` | `+29.23%` | `+21.33%` |
| `wd_01` | `+11.11%` | `-15.01%` | `-42.47%` |
| `meta sticky return 14/5 + sm=0.01` | `+41.27%` | `-47.79%` | `-4.75%` |

### Current read

- Best **current slice**: `meta sticky return 14/5 + sm=0.01`
- Best **3-window hourly robustness**: `reg_combo_2`
- Best **latest-window pure daily PnL**: `ent_anneal`
- Decision: **no live promotion yet**. The meta selector is too regime-sensitive, and no new selector from the 3-window sweep stayed positive on hourly replay across all tested windows.

## 2026-03-16 Robust Daily Variant Sweep

This pass extended the daily RL sweep around the only mixed23 family that had
stayed positive on all three 60-day hourly replay windows: `reg_combo_2`.

### What changed

- `pufferlib_market.train` now exposes `--smoothness-penalty`.
- `pufferlib_market.autoresearch_rl` now supports and records
  `drawdown_penalty`, `smooth_downside_penalty`, and `smoothness_penalty`.
- Added a targeted robust-daily batch around `reg_combo_2`:
  - `robust_reg_wd02`
  - `robust_reg_tp005`
  - `robust_reg_tp01`
  - `robust_reg_tp005_sds02`
  - `robust_reg_tp005_dd002`
  - `robust_reg_tp005_sm001`
  - `robust_reg_tp005_ent`
  - `robust_reg_h512_tp005`

### Repro command

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 180 --max-trials 8 \
  --descriptions robust_reg_wd02,robust_reg_tp005,robust_reg_tp01,robust_reg_tp005_sds02,robust_reg_tp005_dd002,robust_reg_tp005_sm001,robust_reg_tp005_ent,robust_reg_h512_tp005 \
  --periods-per-year 365 --max-steps-override 90 \
  --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
  --holdout-eval-steps 90 --holdout-n-windows 20 \
  --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5 \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-fill-buffer-bps 5 \
  --rank-metric replay_hourly_return_pct \
  --leaderboard pufferlib_market/autoresearch_mixed23_fresh_robust_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/mixed23_fresh_robust
```

### Full fresh-val sweep result (`2025-06-01..2026-02-05`)

Artifacts:
- `pufferlib_market/autoresearch_mixed23_fresh_robust_leaderboard.csv`
- `pufferlib_market/checkpoints/mixed23_fresh_robust/*/best.pt`

| Config | Replay Hourly Return | Val Return | Holdout Robust | Notes |
|--------|---------------------:|-----------:|---------------:|-------|
| `robust_reg_tp005_sds02` | `+7.31%` | `-5.59%` | `-147.26` | Best replay, but daily/holdout weak |
| `robust_reg_tp005_ent` | `-1.97%` | `-7.69%` | `-191.95` | Near-flat replay, weak elsewhere |
| `robust_reg_tp01` | `-3.57%` | `+64.43%` | `-95.92` | Best balanced new candidate |
| `robust_reg_tp005_sm001` | `-8.83%` | `+22.74%` | `-140.66` | Smoothness helped less than trade penalty |
| `robust_reg_h512_tp005` | `-18.92%` | `+49.42%` | `-71.64` | Best holdout score, unstable replay |
| `robust_reg_tp005_dd002` | `-26.84%` | `+35.45%` | `-95.24` | Drawdown penalty hurt replay |

### 3-window replay retest

Exact per-window outputs:
- `pufferlib_market/mixed23_robust_3window_results.csv`
- `pufferlib_market/mixed23_robust_3window_summary.csv`

Tested set:
- existing baselines: `reg_combo_2`, `ent_anneal`
- new finalists: `robust_reg_tp01`, `robust_reg_tp005_sds02`, `robust_reg_tp005_ent`, `robust_reg_h512_tp005`

| Strategy | Mean Hourly Return | Worst Hourly Return | Mean Hourly MaxDD | Worst Hourly MaxDD |
|----------|--------------------:|--------------------:|------------------:|-------------------:|
| `reg_combo_2` | `+26.03%` | `+21.33%` | `26.98%` | `34.21%` |
| `robust_reg_tp01` | `+6.39%` | `-15.24%` | `28.34%` | `30.97%` |
| `robust_reg_h512_tp005` | `+21.79%` | `-48.55%` | `40.78%` | `53.87%` |
| `robust_reg_tp005_ent` | `-14.69%` | `-23.22%` | `37.16%` | `41.80%` |
| `ent_anneal` | `-14.64%` | `-42.43%` | `37.61%` | `49.40%` |
| `robust_reg_tp005_sds02` | `-23.33%` | `-44.64%` | `36.28%` | `46.92%` |

### Current read

- No new daily checkpoint beat `reg_combo_2` on 3-window hourly robustness.
- `robust_reg_tp01` is the most credible new secondary candidate:
  - current 60-day hourly replay: `+4.01%`
  - current 60-day hourly max drawdown: `25.80%`
  - versus `reg_combo_2`: `+27.53%` return, `34.21%` max drawdown
- That means `robust_reg_tp01` buys lower drawdown on the latest window, but it
  gives up too much return and fails the older window.
- Decision: keep `reg_combo_2` as the best robustness anchor, keep
  `robust_reg_tp01` as the main “lower-DD” branch for follow-up variants, and do
  not promote any new checkpoint to live trading yet.

## 2026-03-16 Symbol Expansion Reality Check

Looked at the simpler Chronos2-style flow in `../btcmarketsbot` for pair ideas.
Useful signal from that codebase:

- `../btcmarketsbot/docs/trainingprocesses.md` still treats Chronos post-processing
  as a mostly universal calibration problem, not a per-pair tuning problem.
- That supports using the simpler bot for **pair discovery**, but keeping the
  unified RL stack as the actual selection layer.

### What the current mixed23 crypto sleeve already has

The current fresh mixed23 daily binaries already include many of the names that
show up as strong on the simpler Chronos leaderboard:

- already present: `SOLUSD`, `AVAXUSD`, `LINKUSD`, `UNIUSD`, `DOTUSD`, `SHIBUSD`
- also already present from the existing crypto sleeve: `BTCUSD`, `ETHUSD`,
  `LTCUSD`, `DOGEUSD`, `AAVEUSD`, `XRPUSD`

### Candidate adds from the pasted list

Missing from mixed23 but locally visible somewhere in the repo:

- `ADAUSD`
- `TRXUSD`
- `APTUSD`

### Data readiness check

Using the actual daily source root (`trainingdata/`) that aligns with the fresh
mixed23 binaries:

- `trainingdata/ADAUSD.csv`: `2021-08-27 .. 2025-08-26`
- `trainingdata/TRXUSD.csv`: `2022-04-01 .. 2023-04-19`
- `trainingdata/APTUSD.csv`: missing

Using the current hourly collector tree (`trainingdatahourly/`):

- `ADAUSD`: only `48` hourly bars (`2026-02-04 .. 2026-02-06`)
- `APTUSD`: only `48` hourly bars (`2026-02-04 .. 2026-02-06`)
- `TRXUSD`: stale (`2022-04-01 .. 2023-04-19`)

### Implication

- A clean `mixed26` export for the current fresh window is **not possible yet**.
- The blocker is data freshness / backfill, not model code.
- For the current daily unified RL stack, the next universe-expansion task is:
  1. repair/backfill `ADAUSD`
  2. repair/backfill `APTUSD`
  3. decide whether `TRXUSD` is still worth adding after fresh history exists
  4. only then train a `mixed26` or `mixed25` variant

### Practical read

- The symbol list from the simpler Chronos bot is still useful as a discovery
  queue.
- But right now, most of the high-signal names from that queue are either already
  inside mixed23 or not fresh enough locally to support the current daily RL
  evaluation window.
- So the next win is not “throw more symbols into mixed23 immediately”; it is
  “fix the data for the next 2-3 candidate symbols and then retest the expanded
  universe under the same 5bp replay gate.”

## Previous Key Findings (binanceprogress1-5 + dailyvshourly.md)

| Finding | Source | Impact |
|---------|--------|--------|
| Daily RL 3.3x better than hourly | dailyvshourly.md | +108% vs +32.5% annualized OOS |
| trade_penalty=0.05 is the daily winner | autoresearch_daily | +20% OOS, 100% profitable, Sortino 1.76 |
| trade_penalty=0.10 also strong daily | sweep_daily_combos | +8.86% OOS, 96% profitable, Sortino 1.24 |
| slip_5bps is the hourly winner | autoresearch_hourly | +5.3% OOS, 96% profitable, Sortino 1.62 |
| 5-min timeboxed training prevents overfitting | OOS eval | 200M+ steps → -3% to -16% OOS |
| 5x leverage optimal for hourly LLM | binanceprogress4 | Sortino 74.69, MaxDD 1.157% |
| Cross-learning improves forecasts 42% | binanceprogress5 | BTC MAE 0.405% → 0.234% |
| 1x leverage >> 2x for RL (48% profitable at 2x) | memory | Risk explodes with leverage on RL alone |
| Live-backtest gap ~2-3% per trade | binanceprogress5 | Limit fills at market ≠ bar touch |
| FDUSD = 0 fee, USDT = 10bps fee | Binance tier | Fee structure is a MAJOR strategic lever |

## Binance-Specific Fee Structure

This is the **single most important** detail differentiating our Binance strategy from generic backtesting:

| Quote Asset | Maker Fee | Taker Fee | Available Pairs | Notes |
|-------------|-----------|-----------|-----------------|-------|
| **FDUSD** | **0%** | **0%** | BTC, ETH, SOL, BNB | Promotional zero-fee. Use LIMIT orders exclusively |
| **USDT** | **0.1%** | **0.1%** | All pairs | Standard fee. 10bps round-trip = 20bps per trade |
| **U (futures)** | 0% | varies | BTC, ETH etc | Futures. Different account type |

**Strategic implication**: With FDUSD, the **only cost** is spread/slippage (~2-5bps on majors). This makes hourly trading FAR more attractive on FDUSD than previously tested (where we assumed 10bps fee). Daily's fee advantage shrinks dramatically when fees are zero.

## Experiment Design: Daily vs Hourly on Binance

### Phase 1: Data Preparation

**FDUSD-3 symbols**: BTCUSD, ETHUSD, SOLUSD (BNB lacks sufficient historical data — only 49 hourly bars)
- Hourly train: 37,060 bars (~4.2 years, Mar 2021 - Jun 2025)
- Hourly val: 6,868 bars (~9.5 months, Jun 2025 - Mar 2026)
- Daily train: 1,210 days (Feb 2022 - Jun 2025)
- Daily val: 286 days (Jun 2025 - Mar 2026)
- Hourly uses price-only features (no Chronos2 dependency) via `export_data_hourly_priceonly.py`
- Daily uses price-only features via `export_data_daily.py`

**USDT-extended symbols** (for diversification sweep): LTCUSD, AVAXUSD, DOGEUSD, LINKUSD, AAVEUSD
- These pay 10bps fee → fee_rate=0.001 in C env
- Only worthwhile if edge > 20bps round-trip
- All have 37,000+ hourly bars and 1,400+ daily bars

### Phase 2: Core Autoresearch Sweeps

Run full 35-config autoresearch on EACH combination:

| Experiment | Timeframe | Fee Rate | Symbols | Episode Length |
|------------|-----------|----------|---------|---------------|
| **fdusd_hourly** | Hourly | 0.0 | 4 FDUSD | 720h (30d) |
| **fdusd_daily** | Daily | 0.0 | 4 FDUSD | 90d |
| **usdt_hourly** | Hourly | 0.001 | 4-6 USDT | 720h (30d) |
| **usdt_daily** | Daily | 0.001 | 4-6 USDT | 90d |

All evaluations use 8bps fill slippage (conservative for Binance majors).

### Phase 3: Leverage Sweep

On best configs from Phase 2, sweep leverage:

| Leverage | Max Leverage | Short Borrow APR | Notes |
|----------|-------------|-------------------|-------|
| 1x | 1.0 | 0% | Spot long-only (current prod) |
| 2x | 2.0 | 6.25% | Cross margin |
| 3x | 3.0 | 6.25% | Cross margin |
| 5x | 5.0 | 6.25% | Max cross margin |
| 1x+short | 1.0 | 6.25% | Spot + short capability |
| 3x+short | 3.0 | 6.25% | Full margin both directions |

**Critical**: 1x was 100% profitable; 2x was 48%. Leverage amplifies both returns AND drawdowns. Need Sortino-focused reward shaping to handle leverage.

### Phase 4: Reward Shaping for Low-Drawdown

For leveraged strategies specifically:
- `downside_penalty`: [0.0, 0.2, 0.5, 1.0] — penalize negative returns
- `smooth_downside_penalty`: [0.0, 0.3, 0.5] — smooth version (differentiable)
- `drawdown_penalty`: [0.0, 0.1, 0.3] — penalize equity drops from peak
- `smoothness_penalty`: [0.0, 0.1, 0.3] — penalize return volatility
- Combined with `trade_penalty` sweep [0.01, 0.03, 0.05, 0.08, 0.10]

### Phase 5: Signal Stacking

Once best base models identified:
1. **Chronos2 forecasts** → embed h1/h24 predictions as features in daily data
2. **Gemini 3.1 Flash overlay** → direction filter + entry price refinement
3. **Ensemble daily+hourly** → daily sets direction, hourly times entry
4. **SMA filter** → suppress longs below SMA-24 (already proven in prod)

### Phase 6: Market Simulator Validation

Final candidates run through `marketsimulator.py` with Binance-realistic parameters:
- `maker_fee=0.0` (FDUSD) or `maker_fee=0.001` (USDT)
- `max_hold_hours=6` (hourly) or `max_hold_hours=72` (daily, 3-day max hold)
- Shared cash simulation across symbols
- Slippage: 5bps for BTC/ETH, 8bps for SOL/BNB, 12bps for alts
- Short borrow fee: 6.25% APR (Binance cross margin rate)

## Annualized PnL Comparison Framework

All results will be reported as **annualized** for fair comparison:

```
annualized_return = (1 + period_return) ^ (365 / period_days) - 1
```

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Annualized Return | >100% | >30% |
| Sortino Ratio | >2.0 | >1.0 |
| Max Drawdown | <5% | <15% |
| Win Rate | >55% | >50% |
| Profitable Episodes | >95% | >80% |

## Current Baseline Performance

### Daily RL (trade_pen_05, fee=10bps)
- **Annualized OOS**: +108.1%
- **Sortino**: 1.76
- **MaxDD**: <7% (worst 90d episode: +6.84%)
- **Profitable**: 100% (500/500)
- **Fee drag**: ~3% annual

### Hourly RL (slip_5bps, fee=10bps)
- **Annualized OOS**: +32.5% (250-day eval)
- **Sortino**: 1.10
- **Profitable**: 79.3%
- **Fee drag**: ~10% annual

### Hourly LLM (Gemini, position_context, 5x leverage)
- **3-day backtest**: +13.2%, Sortino 81.85, MaxDD 1.4%
- **Annualized (extrapolated)**: ~1600%+ (unreliable, small sample)
- **Live reality**: -8.9% in 48h (execution gap)

## Experiments To Run

### Experiment 1: FDUSD Zero-Fee Hourly RL
**Hypothesis**: Zero fees should dramatically improve hourly RL (previous winner was held back by 10bps fee drag).

```bash
# Export FDUSD-4 hourly data
python -m pufferlib_market.export_data \
    --symbols BTCUSD,ETHUSD,SOLUSD,BNBUSD \
    --output-train pufferlib_market/data/fdusd4_hourly_train.bin \
    --output-val pufferlib_market/data/fdusd4_hourly_val.bin

# Run autoresearch with fee=0
python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/fdusd4_hourly_train.bin \
    --val-data pufferlib_market/data/fdusd4_hourly_val.bin \
    --time-budget 300 --max-trials 50 \
    --leaderboard pufferlib_market/autoresearch_fdusd_hourly_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_fdusd_hourly
```

### Experiment 2: FDUSD Zero-Fee Daily RL
**Hypothesis**: Daily with zero fees may not improve as much (already few trades), but still worth testing since fee_2x was #3 on daily.

```bash
# Export FDUSD-4 daily data
python -m pufferlib_market.export_data_daily \
    --symbols BTCUSD,ETHUSD,SOLUSD,BNBUSD \
    --output-train pufferlib_market/data/fdusd4_daily_train.bin \
    --output-val pufferlib_market/data/fdusd4_daily_val.bin

# Run autoresearch daily with fee=0
python -u -m pufferlib_market.autoresearch_rl \
    --train-data pufferlib_market/data/fdusd4_daily_train.bin \
    --val-data pufferlib_market/data/fdusd4_daily_val.bin \
    --time-budget 300 --max-trials 50 \
    --periods-per-year 365 --max-steps-override 90 \
    --leaderboard pufferlib_market/autoresearch_fdusd_daily_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_fdusd_daily
```

### Experiment 3: Leverage Sweep on Best Configs
**Hypothesis**: With Sortino-focused reward shaping, moderate leverage (2-3x) may beat 1x without the 48% profitability collapse seen before.

Requires adding `--max-leverage`, `--short-borrow-apr`, and potentially `--disable-shorts` flags to autoresearch, then sweeping:

```bash
# For each leverage level [1, 2, 3, 5]:
python -u -m pufferlib_market.train \
    --data-path pufferlib_market/data/fdusd4_hourly_train.bin \
    --max-leverage $LEV --short-borrow-apr 0.0625 \
    --trade-penalty 0.05 --fee-rate 0.0 \
    --total-timesteps 999999999 --anneal-lr \
    --hidden-size 1024 --max-steps 720 \
    --checkpoint-dir pufferlib_market/checkpoints/leverage_sweep/lev${LEV}
```

### Experiment 4: Sortino-Optimized Reward Shaping
**Hypothesis**: Adding downside/drawdown penalties during training produces models that trade more conservatively but with better risk-adjusted returns.

Sweep matrix (on best fee/timeframe from Experiments 1-2):
- `trade_penalty` × `downside_penalty` × `leverage`
- Focus: Sortino > 2.0, MaxDD < 10%

### Experiment 5: Daily+Hourly Ensemble
**Hypothesis**: Daily model sets direction (long/short/flat), hourly model times entry within the day. Combined should beat either alone.

Architecture:
```
Daily model (midnight UTC):
  → Direction: LONG BTC, SHORT ETH, FLAT SOL, LONG BNB

Hourly model (each hour):
  → Only act if direction matches daily signal
  → Time entry/exit within the daily window
  → Trailing stop: 0.3% (proven in prod)
```

### Experiment 6: Add Chronos2 Forecasts to Daily
**Hypothesis**: Daily models currently use pure technicals (16 features). Adding Chronos2 h24 forecast deltas should improve signal quality significantly (42% MAE improvement from cross-learning).

Requires modifying `export_data_daily.py` to embed daily Chronos2 forecast features.

### Experiment 7: Gemini LLM Direction Filter
**Hypothesis**: LLM overlay with position context improved Sortino from ~30 to 78 on hourly. Should also help daily.

Test on best daily+hourly models with:
- Gemini 2.5 Flash (production model)
- h1_only horizon (outperformed h24 for Sortino)
- Position context prompt (proven superior)

## Implementation Plan

### Step 1: Fee-Rate Support in Autoresearch
Add `--fee-rate` CLI flag to `autoresearch_rl.py` so we can sweep fee=0 vs fee=10bps.

### Step 2: Export FDUSD-Specific Data
Create hourly + daily binary exports for BTC/ETH/SOL/BNB FDUSD pairs.

### Step 3: Run FDUSD Sweeps (Experiments 1 & 2)
~35 configs × 2 timeframes × 5 min each = ~6 hours total. Can run in parallel on GPU.

### Step 4: Leverage Sweep Script
Create `sweep_leverage.py` that tests leverage=[1,2,3,5] × best configs × fee tiers.

### Step 5: Market Simulator Validation
Run top-3 candidates through full `marketsimulator.py` with Binance-realistic params.

### Step 6: Signal Stacking (Chronos2 + Gemini)
Layer on forecast features and LLM overlay on best base models.

### Step 7: Paper Trade 30 Days
Deploy winner on Binance testnet before real capital.

## Market Simulator Realism Checklist

| Aspect | Current Sim | Binance Reality | Gap |
|--------|-------------|----------------|-----|
| Maker fee FDUSD | 0.1% default | **0%** | **Must fix** |
| Maker fee USDT | 0.1% | 0.1% | OK |
| Taker fee | N/A | 0.1% | Use limit orders only |
| Slippage | 5-8bps | 2-5bps majors, 8-15bps alts | OK (conservative) |
| Short borrow | configurable | ~6.25% APR (varies) | OK |
| Leverage | configurable | up to 5x cross | OK |
| Fill probability | configurable | ~95-99% for majors | Set 0.95-0.98 |
| Order book depth | N/A | Deep for majors | N/A (limit orders) |
| Max hold enforcement | configurable | Manual via bot | OK |
| Position sizing | % of cash | Min notional varies | Need to check |

## Early Results: Slippage Sensitivity Analysis

Before the FDUSD-specific sweeps complete, we can evaluate existing models at FDUSD-realistic slippage (3bps instead of 8bps):

| Strategy | 8bps Slippage | 3bps Slippage | Improvement |
|----------|--------------|--------------|-------------|
| **Daily trade_pen_05** | +108.1% ann, Sortino 1.76 | **+132.9% ann, Sortino 1.90** | +23% |
| **Hourly slip_5bps** | +32.5% ann, Sortino 1.10 | **+84.8% ann, Sortino 1.61** | +161% (2.6x!) |

**Critical finding**: Hourly trading benefits FAR more from lower slippage (2.6x improvement) because it trades ~6x more often. At FDUSD-realistic slippage, hourly narrows the gap to 1.6x (vs 3.3x at 8bps). But daily still wins.

Note: These use the 5-symbol crypto6 data (includes LTC/AVAX which would be USDT 10bps in reality). The FDUSD-3 specific sweeps are running now.

## FDUSD-3 Sweep Results (Experiments 1 & 2 — IN PROGRESS)

### Critical Finding: 3 Symbols Is Not Enough

**ALL 35 configs are negative OOS on both hourly and daily** for the FDUSD-3 dataset (BTC, ETH, SOL only). This contrasts sharply with the 5-symbol crypto6 dataset where trade_pen_05 got +20% OOS daily and slip_5bps got +5.3% OOS hourly.

**FDUSD Daily (fee=0, 3 symbols, 16/35 trials done):**
| Rank | Config | OOS Return | Sortino | Profitable% |
|------|--------|-----------|---------|-------------|
| 1 | trade_pen_01 | -23.5% | -0.91 | 0% |
| 2 | trade_pen_05 | -26.1% | -1.02 | 0% |
| 3 | ent_anneal | -29.7% | -1.36 | 0% |

**FDUSD Hourly (fee=0, 3 symbols, 13/35 trials done):**
| Rank | Config | OOS Return | Sortino | Profitable% |
|------|--------|-----------|---------|-------------|
| 1 | wd_005 | -54.4% | -12.63 | 0% |
| 2 | slip_10bps | -56.1% | -12.50 | 0% |
| 3 | slip_5bps | -58.1% | -13.74 | 0% |

**Why 3 symbols fails while 5 succeeds:**
1. **Fewer opportunities**: RL agent has only 3 choices vs 5 — less ability to rotate between uncorrelated moves
2. **BTC/ETH/SOL are highly correlated** (0.84-0.91) — diversification is minimal
3. **The agent learned to exploit LTC/AVAX decorrelation** in the 5-symbol data — removing them eliminates this edge
4. This confirms the memory finding: "more symbols = better"

**Implication for Binance deployment:**
- Pure FDUSD-3 (0% fee) RL alone won't work — need additional signal (Chronos2 forecasts, LLM overlay)
- **SOLUTION: Train on 8+ symbols, execute FDUSD-3 at 0% fee (hybrid approach) — CONFIRMED WORKING**
- The LLM overlay + Chronos2 forecasts may provide additional edge

## 8-Symbol Daily Sweep Results (KEY FINDING)

Training on 8 diverse crypto symbols (BTC, ETH, SOL, LTC, LINK, UNI, DOGE, AAVE) with fee=0 produces **strong positive OOS returns**. This is the winning approach.

| Rank | Config | OOS Return (90d) | Sortino | Profitable% | Annualized |
|------|--------|-----------------|---------|-------------|-----------|
| **1** | **clip_anneal** | **+18.84%** | **1.85** | **100%** | **~+108%** |
| 2 | slip_10bps | +14.35% | 1.71 | 100% | ~+80% |
| 3 | wd_05 | +8.73% | 1.40 | 100% | ~+40% |
| 4 | wd_01 | +1.66% | 1.20 | 75% | ~+7% |
| 5 | fee_2x | -1.29% | 1.11 | 34% | — |
| ... | trade_pen_01 | -11.4% | 0.61 | 0% | — |
| ... | trade_pen_05 | -17.2% | 0.24 | 0% | — |

**Critical insight: trade_penalty is COUNTERPRODUCTIVE with zero fees!**
- On 5-symbol 10bps-fee data: trade_pen_05 was #1 (+20% OOS)
- On 8-symbol 0-fee data: trade_pen_05 is #13 (-17.2% OOS)
- With zero fees, there's no reason to penalize trading — more trading = more opportunity
- clip_anneal (annealing PPO clip epsilon 0.2→0.05) is the new champion

**Why 8 symbols succeeds where 3 fails:**
- BTC/ETH/SOL correlation: 0.84-0.91 (too similar)
- Adding LTC/LINK/UNI/DOGE/AAVE: correlation 0.74-0.86 with BTC (more decorrelated)
- Agent can rotate between 8 opportunities vs 3 → more edges found
- Training return 110x (8-sym) vs 23x (3-sym) → more signal to learn from

## Expected Outcome

Based on prior findings, the prediction is:

1. **FDUSD hourly RL should dramatically improve** — removing 10bps fee removes the main drag. Predicted: +60-80% annualized (up from +32.5%).
2. **FDUSD daily RL should stay similar** — it already makes few trades. Predicted: +100-120% annualized.
3. **Moderate leverage (2-3x) with Sortino shaping should work** — previous 2x failure was without reward shaping. Predicted: 2x → +150-200% annualized with Sortino > 1.5.
4. **Best combination will be**: FDUSD daily direction + hourly entry timing + 2-3x leverage + Gemini filter + trailing stop. Target: **+200-300% annualized, Sortino > 2.0, MaxDD < 10%**.

## Decision Criteria

Deploy the strategy that maximizes:

```
score = annualized_return × min(sortino / 2.0, 1.0) × (1 - max_drawdown / 0.15)
```

This penalizes high-return strategies with poor risk management. A +200% return with Sortino 1.0 and 15% MaxDD scores lower than +150% with Sortino 2.5 and 5% MaxDD.

## Risk Management for Live Deployment

| Control | Setting | Rationale |
|---------|---------|-----------|
| Max position size | 20% of account per symbol | No single-symbol blow-up |
| Max leverage | 3x (even if 5x available) | Leave margin buffer |
| Trailing stop | 0.3% from peak | Proven in hourly prod |
| Daily loss limit | -3% of account | Stop trading for the day |
| Weekly loss limit | -8% of account | Stop trading, review model |
| Max hold (hourly) | 6 hours | Proven optimal |
| Max hold (daily) | 72 hours (3 days) | Prevents multi-day underwater |
| SMA filter | Price > SMA-24 for longs | Suppress trades in downtrends |
| Execution | Limit orders ONLY | Capture maker fee tier |
| Kill switch | Manual abort | Always available |

---

*Status: Experiment design complete. Implementation starting.*
*Data: 5 crypto symbols × hourly+daily × train+val exports needed*
*Compute: ~12 GPU-hours for full sweep*
*Target: Identify production candidate within 24h*

## 2026-03-16 Remote Chronos2 -> RL pipeline

Added a reproducible remote-first hourly pipeline:

- launcher: `scripts/launch_remote_hourly_chronos_rl.py`
- tagged LoRA batch runner: `scripts/run_crypto_lora_batch.py`
- shared plan/window helpers: `src/remote_training_pipeline.py`

The launcher now:

- computes the latest shared hourly train/val window locally
- writes `analysis/remote_runs/<run_id>/launch_manifest.json`
- launches a detached remote pipeline on `administrator@93.127.141.100:/nvme0n1-disk/code/stock-prediction`
- runs Chronos2 LoRA batch -> promotion -> forecast caches -> forecast-feature MKTD export -> `pufferlib_market.autoresearch_rl`

Important fix from the first live probe:

- remote pipeline shell used `set -u` and failed when `PYTHONPATH` was unset
- fixed by exporting `PYTHONPATH="$PWD:$PWD/PufferLib:${PYTHONPATH:-}"`

### Live probe

Run id:

- `probe_hourly_remote_live_fix_20260316`

Launch command:

```bash
source .venv313/bin/activate
python scripts/launch_remote_hourly_chronos_rl.py \
  --run-id probe_hourly_remote_live_fix_20260316 \
  --symbols BTCUSD,ETHUSD \
  --preaugs baseline \
  --context-lengths 128 \
  --learning-rates 5e-5 \
  --num-steps 120 \
  --train-hours 336 \
  --val-hours 72 \
  --time-budget 300 \
  --max-trials 1 \
  --descriptions baseline_anneal_lr \
  --no-sync
```

Artifacts/logs:

- local manifest: `analysis/remote_runs/probe_hourly_remote_live_fix_20260316/launch_manifest.json`
- remote pid: `3795881`
- remote log: `analysis/remote_runs/probe_hourly_remote_live_fix_20260316/pipeline.log`

Current status:

- LoRA batch completed successfully for `BTCUSD` and `ETHUSD`
- promotion into `hyperparams/chronos2/hourly/{BTCUSD,ETHUSD}.json` completed
- pipeline has advanced into `scripts/build_hourly_forecast_caches.py`

Probe LoRA metrics from `analysis/remote_runs/probe_hourly_remote_live_fix_20260316/lora_results/probe_hourly_remote_live_fix_20260316_batch_summary.csv`:

| Symbol | Preaug | Ctx | LR | Val MAE% | Test MAE% | Val Consistency | Test Consistency | Elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTCUSD | baseline | 128 | 5e-5 | 3.2096 | 2.4037 | 5.7406 | 3.2260 | 33.4s |
| ETHUSD | baseline | 128 | 5e-5 | 2.9631 | 2.4817 | 5.3171 | 3.6232 | 36.1s |

Monitoring commands:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 \
  'cd /nvme0n1-disk/code/stock-prediction && tail -n 80 analysis/remote_runs/probe_hourly_remote_live_fix_20260316/pipeline.log'

rsync -az -e "ssh -o StrictHostKeyChecking=no" \
  administrator@93.127.141.100:/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/probe_hourly_remote_live_fix_20260316/ \
  analysis/remote_runs/probe_hourly_remote_live_fix_20260316/remote_run/
```

### Bounds-aware relaunch

The first probe still overran the remote data freshness during validation export because local hourly data was ahead of the 5090 box by roughly two days. The launcher now checks remote overlap bounds before building windows.

Fresh relaunch:

- run id: `probe_hourly_remote_live_bounds_20260316`
- local manifest: `analysis/remote_runs/probe_hourly_remote_live_bounds_20260316/launch_manifest.json`
- remote pid: `3907425`
- effective shared window:
  - train: `2026-02-25T04:00:00+00:00 .. 2026-03-11T03:00:00+00:00`
  - val: `2026-03-11T04:00:00+00:00 .. 2026-03-14T03:00:00+00:00`

Current status at write time:

- remote pipeline has started successfully and re-entered the LoRA batch with the corrected windowing path

## 2026-03-16 Gemini Double-Pass Daily Replay

Goal:

- test a two-pass Gemini path:
  - pass 1: normal Gemini trade plan
  - pass 2: Gemini re-reads the full task plus the prior JSON plan and can revise the whole plan
- score it on the daily-decision hourly-replay simulator before considering deployment

Code changes:

- `llm_hourly_trader/providers.py`
  - added `reprompt_passes` support in `call_llm`
  - added a Gemini review prompt builder for pass 2+
  - added bounded Gemini HTTP timeout support via `GEMINI_HTTP_TIMEOUT_MS` (default `120000`)
- `unified_orchestrator/backtest_hybrid.py`
  - added `--decision-cadence {hourly,daily}`
  - added `--reprompt-passes`
  - added `--output-json`
  - fixed max drawdown reporting to compute from the equity curve instead of reading a nonexistent simulator metric key
- `unified_orchestrator/orchestrator.py`
  - added `--reprompt-passes` to the live orchestrator CLI
  - threaded `reprompt_passes` into both crypto and stock `call_llm(...)`

Tests:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_llm_hourly_trader_provider_cache_only.py \
  tests/test_unified_orchestrator_backtest_hybrid.py \
  tests/test_unified_orchestrator_orchestrator.py
```

Result:

- `10 passed`

### 60-day baseline: current model, single pass

Command:

```bash
source .venv313/bin/activate
python -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --days 60 \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 1 \
  --output-json analysis/gemini_reprompt_60d_20260316/crypto5_daily_single_pass.json
```

Exact replay window selected by the simulator:

- `2025-12-08 15:00:00+00:00 .. 2026-02-06 15:00:00+00:00`

Saved artifact:

- `analysis/gemini_reprompt_60d_20260316/crypto5_daily_single_pass.json`

Baseline metrics:

- return: `-17.08%`
- sortino: `-5.27`
- max drawdown: `20.60%`
- fills: `168`
- logical Gemini calls: `305`

### Double-pass outcome on the same model/key

Command attempted:

```bash
source .venv313/bin/activate
python -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --days 60 \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --output-json analysis/gemini_reprompt_60d_20260316/crypto5_daily_double_pass.json
```

What happened:

- the run progressed through all of `BTCUSD` and deep into `ETHUSD`
- Gemini calls were no longer able to hang forever because the new HTTP timeout/retry path bounded them
- but the run hit a hard quota ceiling on the current production model/key:
  - `429 RESOURCE_EXHAUSTED`
  - `limit: 500`
  - `metric: generativelanguage.googleapis.com/generate_content_free_tier_requests`
  - `model: gemini-3.1-flash-lite`

Implication:

- the 60-day daily 5-symbol replay needs `305` decisions
- a double-pass replay needs `610` Gemini requests
- with the current free-tier `gemini-3.1-flash-lite` quota, this experiment cannot complete in one day on the current key/model

Decision:

- do **not** deploy double-pass Gemini on the current production model/key
- reasons:
  - no completed 60-day PnL result for the double-pass path yet
  - it is quota-infeasible on the current Gemini bucket
  - it is materially slower even with the cache isolating pass 2

Next realistic follow-up:

- either rerun tomorrow after quota reset on the same model
- or test a mixed setup where pass 1 stays on current prod and pass 2 uses a different Gemini model/quota bucket

### Mixed-model actionable reviewer path

Code changes:

- `llm_hourly_trader/providers.py`
  - added `reprompt_policy=actionable`
  - added `reprompt_policy=entry_only`
  - added `review_model` so pass 2 can use a different Gemini bucket
  - added `review_thinking_level` so pass 2 can use a lighter Gemini thinking config than pass 1
  - added call tracing so backtests can distinguish logical passes from uncached provider calls
- `unified_orchestrator/backtest_hybrid.py`
  - forwards `review_model`
  - forwards `review_thinking_level`
  - forwards `review_cache_namespace`
  - reports actual provider calls separately from logical passes
  - fixes the explicit-window banner so it reports the true timestamp span instead of the default `--days`
- `unified_orchestrator/orchestrator.py`
  - live path now accepts `--review-model`
  - live path now accepts `--review-thinking-level`

Focused verification:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_llm_hourly_trader_provider_cache_only.py \
  tests/test_unified_orchestrator_backtest_hybrid.py \
  tests/test_unified_orchestrator_orchestrator.py
```

Result:

- `18 passed`

Low-cost smoke test:

```bash
source .venv313/bin/activate
python -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --days 2 \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --reprompt-policy actionable \
  --review-model gemini-2.5-flash \
  --output-json analysis/gemini_reprompt_60d_20260317/smoke_btc_daily_actionable_review25.json
```

Smoke result:

- artifact: `analysis/gemini_reprompt_60d_20260317/smoke_btc_daily_actionable_review25.json`
- window: `2026-03-14 10:00:00+00:00 .. 2026-03-16 10:00:00+00:00`
- return: `+0.00%`
- fills: `0`
- provider calls: `3`
- takeaway: mixed-model review works, and `actionable` really skips pass 2 on flat holds

Cached 60-day first-pass analysis on the exact baseline window:

```bash
source .venv313/bin/activate
python scripts/analyze_actionable_reprompts.py \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/crypto5_actionable_counts.json
```

Actionable-plan counts from cached single-pass prompts:

- `BTCUSD`: `40 / 61` actionable
- `ETHUSD`: `13 / 61` actionable
- `SOLUSD`: `17 / 61` actionable
- `LTCUSD`: `24 / 61` actionable
- `AVAXUSD`: `20 / 61` actionable
- total: `114 / 305` actionable

Entry-only review counts from the same cached first-pass prompts:

- `BTCUSD`: `18 / 61`
- `ETHUSD`: `8 / 61`
- `SOLUSD`: `8 / 61`
- `LTCUSD`: `8 / 61`
- `AVAXUSD`: `8 / 61`
- total: `50 / 305`

Category mix on the same window:

- `50` `entry_with_exit`
- `64` `exit_only`
- `191` `flat_hold`

Implication:

- the mixed-model 60-day replay should need about `305` logical first-pass decisions but only about `114` second-pass Gemini reviewer calls
- that makes the split-bucket path operationally plausible even though the old same-model double-pass was impossible
- the new `entry_only` policy is the cheapest realistic follow-up: it would review only `50` entry plans and skip `64` exit-management holds

Next speed lever:

- reviewer pass latency is still high because pass 2 currently inherits the same `HIGH` thinking config as pass 1 unless overridden
- the code now supports `--review-thinking-level`, so the next cheap runtime experiment is:
  - keep pass 1 on the current production-like thinking level
  - run pass 2 with `LOW` or no extra thinking
- reviewer experiments now also support `--review-cache-namespace`, so `LOW` and `HIGH` reviewer runs no longer silently share the same pass-2 cache entries
- reviewer experiments now also support `--review-max-confidence`, so pass 2 can be limited to shakier first-pass plans

Confidence-capped entry-review counts from the same cached first-pass prompts:

```bash
source .venv313/bin/activate
python scripts/analyze_actionable_reprompts.py \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --reprompt-policy entry_only \
  --review-max-confidence 0.60 \
  --output-json analysis/gemini_reprompt_60d_20260317/crypto5_entryonly_conf06_counts.json

python scripts/analyze_actionable_reprompts.py \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --reprompt-policy entry_only \
  --review-max-confidence 0.55 \
  --output-json analysis/gemini_reprompt_60d_20260317/crypto5_entryonly_conf055_counts.json
```

Counts:

- `entry_only`: `50 / 305`
- `entry_only + conf<=0.60`: `33 / 305`
- `entry_only + conf<=0.55`: `13 / 305`

Interpretation:

- `0.60` looks like the first sensible confidence cap if we want a meaningfully cheaper reviewer without dropping to almost no second-pass coverage
- `0.55` is probably too aggressive for the first full-basket follow-up unless the uncapped `entry_only` run still comes back too slow

Chronos-side follow-up from codebase review:

- best low-blast-radius hook for OHLC smoothing / band-clamp experiments:
  - hourly cache writer: `binanceneural/forecasts.py`
  - daily cache writer: `strategytrainingneural/forecast_cache.py`
- best implementation shape:
  - one shared postprocessor under `src/models/chronos2_postprocessing.py`
  - writer-edge integration first, consumer-edge feature expansion only if the RL model needs band width explicitly

Current run in progress:

```bash
source .venv313/bin/activate
python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --reprompt-policy actionable \
  --review-model gemini-2.5-flash \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/crypto5_daily_actionable_review25_60d.json
```

Additional reviewer-isolation probes:

1. Latest-window BTC smoke with isolated `entry_only + LOW` reviewer:

```bash
source .venv313/bin/activate
python -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --days 2 \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --reprompt-policy entry_only \
  --review-model gemini-2.5-flash \
  --review-thinking-level LOW \
  --review-cache-namespace entry-low-smoke \
  --output-json analysis/gemini_reprompt_60d_20260317/smoke_btc_daily_entryonly_review25_low.json
```

Result:

- artifact: `analysis/gemini_reprompt_60d_20260317/smoke_btc_daily_entryonly_review25_low.json`
- window: `2026-03-14 11:00:00+00:00 .. 2026-03-16 11:00:00+00:00`
- `3` provider calls
- `3` logical passes
- `0` fills
- takeaway: this slice had no entry reviews, so it only validated the isolated-cache machinery

2. Early-December BTC probe with isolated `entry_only + LOW` reviewer:

```bash
source .venv313/bin/activate
python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --reprompt-policy entry_only \
  --review-model gemini-2.5-flash \
  --review-thinking-level LOW \
  --review-cache-namespace entry-low-dec-btc \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2025-12-10T23:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/smoke_btc_dec_entryonly_review25_low.json
```

Result:

- artifact: `analysis/gemini_reprompt_60d_20260317/smoke_btc_dec_entryonly_review25_low.json`
- `3` provider calls
- `4` logical passes
- `0` fills
- takeaway: this window did trigger at least one isolated entry review without forcing a full pass-1 rerun

BTC-only canary now running:

```bash
source .venv313/bin/activate
python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --reprompt-policy entry_only \
  --review-model gemini-2.5-flash \
  --review-thinking-level LOW \
  --review-cache-namespace btc60-entry-low \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/btc60_entryonly_review25_low.json
```

BTC-only confidence-capped canaries now running:

```bash
source .venv313/bin/activate
python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --reprompt-policy entry_only \
  --review-max-confidence 0.60 \
  --review-model gemini-2.5-flash \
  --review-thinking-level LOW \
  --review-cache-namespace btc60-entry-low-conf06 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/btc60_entryonly_conf06_review25_low.json

python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 2 \
  --reprompt-policy entry_only \
  --review-max-confidence 0.55 \
  --review-model gemini-2.5-flash \
  --review-thinking-level LOW \
  --review-cache-namespace btc60-entry-low-conf055 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/btc60_entryonly_conf055_review25_low.json
```

First completed capped-review result:

- artifact: `analysis/gemini_reprompt_60d_20260317/btc60_entryonly_conf055_review25_low.json`
- config:
  - `reprompt_policy=entry_only`
  - `review_max_confidence=0.55`
  - `review_model=gemini-2.5-flash`
  - `review_thinking_level=LOW`
- metrics:
  - return: `-26.40%`
  - sortino: `-4.27`
  - max drawdown: `35.91%`
  - fills: `39`
- versus BTC single-pass baseline:
  - return improved by `+1.38%`
  - sortino improved by `+0.69`
  - max drawdown improved by `0.80%`
  - fills dropped by `20`
- conclusion:
  - this is a real improvement over single-pass on the 60-day BTC window
  - but it is still materially negative, so it is not a production candidate by itself
  - it does justify keeping the cheaper confidence-capped reviewer family in the search set

Instrumentation note:

- historical `provider calls` on the already-running canaries are best treated as an upper bound
- after this result, provider-call tracing was tightened so future runs count actual provider function invocations rather than outer cache misses under concurrent cache fills

Reference baseline on the same BTC window:

```bash
source .venv313/bin/activate
python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 1 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/btc60_single_pass.json
```

BTC single-pass baseline:

- artifact: `analysis/gemini_reprompt_60d_20260317/btc60_single_pass.json`
- return: `-27.77%`
- sortino: `-4.97`
- max drawdown: `36.71%`
- fills: `59`

Chronos writer-edge cleanup:

- added shared forecast repair in `src/models/chronos2_postprocessing.py`
- wired that repair into:
  - `binanceneural/forecasts.py`
  - `strategytrainingneural/forecast_cache.py`
  - `binanceneural/trade_binance_daily_levels.py`
- behavior:
  - monotonic close quantiles
  - high/low wrapped around repaired close
  - move/volatility derived from repaired values instead of raw broken quantiles

Focused verification after the Gemini and Chronos changes:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_llm_hourly_trader_provider_cache_only.py \
  tests/test_unified_orchestrator_backtest_hybrid.py \
  tests/test_unified_orchestrator_orchestrator.py \
  tests/test_analyze_actionable_reprompts.py \
  tests/test_trade_binance_daily_levels.py \
  tests/test_chronos2_postprocessing.py \
  tests/test_strategytrainingneural_forecasts.py \
  tests/test_chronos_forecast_manager_modes.py
```

Result:

- `32 passed`

### `../btcmarketsbot` ideas worth stealing later

Useful:

- exchange metadata filter plus an explicit executable universe allowlist
- account-conditioned universe pruning: keep symbols containing held assets even if they fall outside the main candidate list
- Chronos-side OHLC post-processing with:
  - `wick_scale=0.75`
  - `body_scale=1.0`
  - band-clamped OHLC consistency
- optional close-median EMA smoothing and explicit forecast uncertainty bands (`low_band` / `high_band`)
- multi-symbol Chronos batching with a bounded context window

Not useful:

- the repo mostly picks the first `N` pairs rather than actually scoring them
- no meaningful FDUSD or fee-free routing logic was found
- the old spread/ticker logic is tied to deprecated Poloniex metadata

### 2026-03-17: simulator alignment, reviewer comparison, and next RL branch

Backtest alignment change:

- `unified_orchestrator/backtest_hybrid.py` now enforces a minimum confidence floor for new long entries in the simulator, matching the live crypto execution behavior more closely.
- This gate only suppresses fresh entries while flat. It does not suppress exits for positions already held.
- The backtest now records `min_plan_confidence` and `suppressed_low_conf_entries` in the output JSON.

Focused verification:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_unified_orchestrator_backtest_hybrid.py \
  tests/test_pufferlib_market_autoresearch_rl.py
```

Result:

- `18 passed`

#### Cached 60d confidence-floor sweep on single-pass Gemini

Window:

- `2025-12-08T15:00:00Z .. 2026-02-06T15:00:00Z`

Commands:

```bash
source .venv313/bin/activate
python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 1 \
  --min-plan-confidence 0.4 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/btc60_single_pass_conf04.json

python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 1 \
  --min-plan-confidence 0.5 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/btc60_single_pass_conf05.json

python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 1 \
  --min-plan-confidence 0.55 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/btc60_single_pass_conf055.json

python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 1 \
  --min-plan-confidence 0.4 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/crypto5_daily_single_pass_conf04.json

python -u -m unified_orchestrator.backtest_hybrid \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD \
  --modes gemini_only \
  --decision-cadence daily \
  --reprompt-passes 1 \
  --min-plan-confidence 0.5 \
  --start-ts 2025-12-08T15:00:00Z \
  --end-ts 2026-02-06T15:00:00Z \
  --output-json analysis/gemini_reprompt_60d_20260317/crypto5_daily_single_pass_conf05.json
```

Results:

- BTC `0.40` floor: unchanged from baseline
  - return `-27.77%`, sortino `-4.97`, max drawdown `36.71%`, fills `59`, suppressed entries `0`
- BTC `0.50` floor: slightly worse
  - return `-28.39%`, sortino `-5.06`, max drawdown `37.24%`, fills `57`, suppressed entries `1`
- BTC `0.55` floor: same as `0.50`
  - return `-28.39%`, sortino `-5.06`, max drawdown `37.24%`, fills `57`, suppressed entries `1`
- 5-symbol `0.40` floor: unchanged from baseline
  - return `-17.08%`, sortino `-5.27`, max drawdown `20.60%`, fills `168`, suppressed entries `0`
- 5-symbol `0.50` floor: slightly worse
  - return `-17.19%`, sortino `-5.32`, max drawdown `20.70%`, fills `166`, suppressed entries `1`

Conclusion:

- The recent reviewer gains are not explained by a simple stricter confidence gate.
- Matching the live `0.40` floor is still the right simulator default, but pushing the floor higher hurt this window.

#### BTC capped-review update

Artifact:

- `analysis/gemini_reprompt_60d_20260317/btc60_entryonly_conf06_review25_low.json`

Result on the same BTC 60d window:

- config:
  - `reprompt_policy=entry_only`
  - `review_max_confidence=0.60`
  - `review_model=gemini-2.5-flash`
  - `review_thinking_level=LOW`
- metrics:
  - return `-26.44%`
  - sortino `-4.20`
  - max drawdown `35.94%`
  - fills `13`
- versus single-pass:
  - return improved by `+1.34%`
  - sortino improved by `+0.76`
  - max drawdown improved by `0.77%`
  - fills dropped by `46`
- versus the earlier `conf<=0.55` reviewer:
  - return is slightly worse by `0.04%`
  - sortino is slightly better by `0.07`
  - max drawdown is slightly worse by `0.03%`
  - fills dropped by `26`

Read:

- `conf<=0.60` is the cleanest BTC reviewer candidate so far because it keeps almost the same PnL improvement while cutting fills very aggressively.
- It is still negative on the 60-day BTC window, so it is not deployable by itself.

#### RL autoresearch wiring fix: `smooth_downside_temperature`

Problem:

- `pufferlib_market.train` and the C environment already support `smooth_downside_temperature`.
- `pufferlib_market.autoresearch_rl` was not forwarding that knob, so every smooth-downside experiment was silently stuck on the trainer default temperature.

Fix:

- added `smooth_downside_temperature` to `TrialConfig`
- passed `--smooth-downside-temperature` through the train command
- logged the knob in leaderboard rows
- added two new robust daily variants:
  - `robust_reg_tp005_sds02_t01`
  - `robust_reg_tp005_sds02_t05`

Started targeted sweep:

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD/PufferLib:$PYTHONPATH python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 180 --max-trials 3 \
  --descriptions robust_reg_tp005_sds02,robust_reg_tp005_sds02_t01,robust_reg_tp005_sds02_t05 \
  --periods-per-year 365 --max-steps-override 90 \
  --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
  --holdout-eval-steps 90 --holdout-n-windows 20 \
  --holdout-fee-rate 0.001 --holdout-fill-buffer-bps 5 \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-fill-buffer-bps 5 \
  --rank-metric replay_hourly_return_pct \
  --leaderboard pufferlib_market/autoresearch_mixed23_fresh_sds_temp_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/mixed23_fresh_sds_temp
```

Current status:

- the sweep is running
- leaderboard now exists:
  - `pufferlib_market/autoresearch_mixed23_fresh_sds_temp_leaderboard.csv`
- trial `0` baseline (`robust_reg_tp005_sds02`, `smooth_downside_temperature=0.02`) has finished:
  - validation return `-12.35%`
  - hourly replay return `+7.97%`
  - hourly replay sortino `1.66`
  - hourly replay max drawdown `44.93%`
  - holdout robust score `-186.57`
- first checkpoint exists:
  - `pufferlib_market/checkpoints/mixed23_fresh_sds_temp/robust_reg_tp005_sds02/best.pt`

#### Remote long-budget mixed23 champions sweep launched (2026-03-20 UTC)

Goal:

- the earlier mixed23 fresh daily batches were mostly `180s`-class trials, which looks too short for the stronger replay/marketsim objective.
- this pass pushes a longer `1800s` per-trial budget over the strongest mixed23 families we already identified: `reg_combo_2`, `ent_anneal`, `robust_reg_tp01`, `robust_reg_tp005_sds02`, `gspo_like_mix15`, `gspo_like_drawdown_mix15`, `per_env_adv_smooth`.

Launch/runtime fixes before starting:

- `scripts/launch_mixed23_retrain.py` now launches the real remote `autoresearch_rl` path instead of the old local `train.py` helper.
- it now syncs only the required launcher/helper files plus the actual mixed23 train/val `.bin` files, instead of trying to rsync the entire `pufferlib_market/` tree.
- `pufferlib_market.autoresearch_rl` now surfaces the real training failure tail when no checkpoint is produced, instead of the opaque `no checkpoint`.

First failed probe:

- run `mixed23_champions_long_20260321_002` immediately failed because the remote repo did not have:
  - `pufferlib_market/data/mixed23_fresh_train.bin`
  - `pufferlib_market/data/mixed23_fresh_val.bin`
- the new sync path fixed that and the clean rerun is below.

Live run:

- local launch command:

```bash
source .venv313/bin/activate
python scripts/launch_mixed23_retrain.py \
  --run-id mixed23_champions_long_20260321_003 \
  --preset champions \
  --time-budget 1800
```

- remote host: `administrator@93.127.141.100`
- remote repo: `/nvme0n1-disk/code/stock-prediction`
- remote env: `.venv313`
- remote PID: `187619`
- remote script: `analysis/remote_runs/mixed23_champions_long_20260321_003/pipeline.sh`
- remote log: `analysis/remote_runs/mixed23_champions_long_20260321_003/pipeline.log`
- local manifest: `analysis/remote_runs/mixed23_champions_long_20260321_003/launch_manifest.json`

Exact remote commands:

```bash
python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/mixed23_fresh_train.bin \
  --val-data pufferlib_market/data/mixed23_fresh_val.bin \
  --time-budget 1800 --max-trials 7 \
  --leaderboard pufferlib_market/mixed23_champions_long_20260321_003_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/mixed23_champions_long_20260321_003 \
  --rank-metric replay_hourly_return_pct \
  --descriptions reg_combo_2,ent_anneal,robust_reg_tp01,robust_reg_tp005_sds02,gspo_like_mix15,gspo_like_drawdown_mix15,per_env_adv_smooth \
  --periods-per-year 365.0 --max-steps-override 90 \
  --holdout-data pufferlib_market/data/mixed23_fresh_val.bin \
  --holdout-eval-steps 90 --holdout-n-windows 20 \
  --holdout-seed 1337 --holdout-fee-rate 0.001 \
  --holdout-fill-buffer-bps 5.0 --holdout-max-leverage 1.0 --holdout-short-borrow-apr 0.0 \
  --replay-eval-data pufferlib_market/data/mixed23_fresh_val.bin \
  --replay-eval-hourly-root trainingdatahourly \
  --replay-eval-start-date 2025-06-01 \
  --replay-eval-end-date 2026-02-05 \
  --replay-eval-fill-buffer-bps 5.0 \
  --replay-eval-hourly-periods-per-year 8760.0

python -u pufferlib_market/fast_marketsim_eval.py \
  --root . \
  --output analysis/remote_runs/mixed23_champions_long_20260321_003/marketsim_30_60_90_120.csv \
  --periods 30,60,90,120 \
  --sort-period 120 \
  --checkpoint-dirs pufferlib_market/checkpoints/mixed23_champions_long_20260321_003 \
  --max-workers 2 \
  --cache-path analysis/remote_runs/mixed23_champions_long_20260321_003/marketsim_cache.json \
  --no-compile --sequential
```

Expected artifacts:

- leaderboard:
  - `pufferlib_market/mixed23_champions_long_20260321_003_leaderboard.csv`
- checkpoints:
  - `pufferlib_market/checkpoints/mixed23_champions_long_20260321_003/`
- post-train 30/60/90/120d marketsim CSV:
  - `analysis/remote_runs/mixed23_champions_long_20260321_003/marketsim_30_60_90_120.csv`

Current status:

- remote log timestamp: `2026-03-20T22:42:57Z`
- first trial currently active: `ent_anneal`
- observed live child process:
  - `187694 /nvme0n1-disk/code/stock-prediction/.venv313/bin/python -u -m pufferlib_market.train ... --checkpoint-dir pufferlib_market/checkpoints/mixed23_champions_long_20260321_003/ent_anneal`

#### Mixed23 latest-data refresh + retrain (2026-03-20 UTC)

Goal:

- stop training on the stale February mixed23 bins
- refresh the mixed23 source data to the latest available March bars
- retrain the best mixed23 families on a recent validation slice that can still support `30/60/90/120d` marketsim scoring

Local data refresh commands:

```bash
source .venv313/bin/activate
python update_daily_data.py \
  --symbols AAPL NFLX NVDA ADBE ADSK COIN GOOG MSFT PYPL SAP TSLA \
            BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD DOGEUSD LINKUSD AAVEUSD UNIUSD DOTUSD SHIBUSD XRPUSD

python download_hourly_data.py \
  --symbols AAPL NFLX NVDA ADBE ADSK COIN GOOG MSFT PYPL SAP TSLA \
            BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD DOGEUSD LINKUSD AAVEUSD UNIUSD DOTUSD SHIBUSD XRPUSD \
  --sleep 0.1
```

Observed refresh result:

- daily `trainingdata/train` moved to a common latest date of `2026-03-20`
- hourly `trainingdatahourly` (using the same candidate order as `pufferlib_market.hourly_replay`) moved to a common latest date of `2026-03-20 19:00:00+00:00`
- daily append counts:
  - stocks: `+29` to `+30` rows each
  - crypto: `+42` rows each
- hourly updates:
  - stale stock names such as `NFLX`, `ADBE`, `PYPL`, `SAP` were extended through March 20
  - stale crypto names such as `LTCUSD`, `AVAXUSD`, `UNIUSD`, `DOTUSD`, `SHIBUSD`, `XRPUSD` were extended through March 20

Latest exported bins:

```bash
source .venv313/bin/activate
PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH python -m pufferlib_market.export_data_daily \
  --symbols AAPL,NFLX,NVDA,ADBE,ADSK,COIN,GOOG,MSFT,PYPL,SAP,TSLA,BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,AAVEUSD,UNIUSD,DOTUSD,SHIBUSD,XRPUSD \
  --data-root trainingdata/train \
  --output pufferlib_market/data/mixed23_latest_train_20260320.bin \
  --end-date 2025-09-21 \
  --min-days 200

PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH python -m pufferlib_market.export_data_daily \
  --symbols AAPL,NFLX,NVDA,ADBE,ADSK,COIN,GOOG,MSFT,PYPL,SAP,TSLA,BTCUSD,ETHUSD,SOLUSD,LTCUSD,AVAXUSD,DOGEUSD,LINKUSD,AAVEUSD,UNIUSD,DOTUSD,SHIBUSD,XRPUSD \
  --data-root trainingdata/train \
  --output pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin \
  --start-date 2025-09-22 \
  --end-date 2026-03-20 \
  --min-days 180
```

Exported ranges:

- train bin:
  - `pufferlib_market/data/mixed23_latest_train_20260320.bin`
  - `2024-01-01 .. 2025-09-21`
  - `630` days
- validation/holdout/replay bin:
  - `pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin`
  - `2025-09-22 .. 2026-03-20`
  - `180` days

Remote training note:

- the stale run `mixed23_champions_long_20260321_003` was stopped because it still used:
  - `pufferlib_market/data/mixed23_fresh_train.bin`
  - `pufferlib_market/data/mixed23_fresh_val.bin`

Current latest-data run:

```bash
source .venv313/bin/activate
python scripts/launch_mixed23_retrain.py \
  --run-id mixed23_latest_champions_20260321_001 \
  --preset champions \
  --time-budget 1800 \
  --train-data pufferlib_market/data/mixed23_latest_train_20260320.bin \
  --val-data pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin \
  --holdout-data pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin \
  --replay-eval-start-date 2025-09-22 \
  --replay-eval-end-date 2026-03-20
```

Remote metadata:

- host: `administrator@93.127.141.100`
- repo: `/nvme0n1-disk/code/stock-prediction`
- env: `.venv313`
- remote PID: `314048`
- remote log: `analysis/remote_runs/mixed23_latest_champions_20260321_001/pipeline.log`
- local manifest: `analysis/remote_runs/mixed23_latest_champions_20260321_001/launch_manifest.json`
- leaderboard target:
  - `pufferlib_market/mixed23_latest_champions_20260321_001_leaderboard.csv`
- checkpoint root:
  - `pufferlib_market/checkpoints/mixed23_latest_champions_20260321_001`
- post-train marketsim output:
  - `analysis/remote_runs/mixed23_latest_champions_20260321_001/marketsim_30_60_90_120.csv`

Current status:

- remote log timestamp: `2026-03-20T22:51:13Z`
- first active trial: `ent_anneal`
- observed live child process:
  - `314144 /nvme0n1-disk/code/stock-prediction/.venv313/bin/python -u -m pufferlib_market.train --data-path pufferlib_market/data/mixed23_latest_train_20260320.bin ... --checkpoint-dir pufferlib_market/checkpoints/mixed23_latest_champions_20260321_001/ent_anneal`

## 2026-03-26 Robust Replay Remote Batch + Bootstrap Fixes

Goal of this pass:

- move the mixed23 remote launcher onto the new robust-start replay ranking path
- make the 5-minute batch actually reproducible on the remote 5090/A40 box instead of failing on environment drift
- keep progress and failure modes written down instead of losing them in shell history

Launcher / infra changes completed locally:

- `pufferlib_market.autoresearch_rl` already had robust-start replay metrics wired; I extended the remote wrappers to expose them end-to-end:
  - `src/remote_training_pipeline.py`
  - `scripts/launch_mixed23_retrain.py`
- new launcher flags now supported:
  - `--replay-eval-run-hourly-policy`
  - `--replay-eval-robust-start-states`
  - `--replay-eval-hourly-periods-per-year`
  - robust replay rank metrics including `replay_combo_score`, `replay_hourly_robust_worst_return_pct`, `replay_hourly_policy_robust_worst_return_pct`
- remote sync was tightened to avoid re-rsyncing bulky `pufferlib_market/checkpoints/` and `pufferlib_market/data/`
- remote pipeline bootstrap now:
  - removes stale `pufferlib_market/build` and existing binding `.so`
  - rebuilds `pufferlib_market.binding` in place with `python pufferlib_market/setup.py build_ext --inplace --force`
  - verifies the rebuilt extension with `python -c 'import pufferlib_market.binding; print("binding OK")'`

Validation:

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH
pytest -q \
  tests/test_pufferlib_market_replay_eval.py \
  tests/test_pufferlib_market_hourly_replay_initial_state.py \
  tests/test_pufferlib_market_autoresearch_rl.py \
  tests/test_gpu_pool_rl.py \
  tests/test_remote_training_pipeline.py \
  tests/test_launch_mixed23_retrain.py
```

- result: `78 passed in 0.95s`

Remote failures found and fixed before the live run:

1. first detached launch failed because only `src/remote_training_pipeline.py` was synced, while remote `autoresearch_rl` also needed updated `src/robust_trading_metrics.py`
2. second launch failed because the old remote `.venv313` had `numpy==2.4.3` while `pufferlib_market.binding` was still compiled against NumPy 1.x
3. attempted editable bootstrap `uv pip install -e .` failed due a broken setuptools package-discovery path on the remote checkout
4. attempted `cd pufferlib_market && python setup.py build_ext --inplace` wrote to the wrong target path for this repo layout
5. final fix was to rebuild from repo root with a clean `build/` and `--force`

Current live proof batch:

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD:$PWD/PufferLib:$PYTHONPATH
python scripts/launch_mixed23_retrain.py \
  --run-id mixed23_robust_probe_20260326_2215 \
  --descriptions reg_combo_2,gspo_like_drawdown_mix15,per_env_adv_smooth,robust_reg_tp005_ent_seed42,robust_reg_tp005_ent_seed7,robust_reg_tp005_ent_seed123 \
  --time-budget 300 \
  --rank-metric replay_hourly_policy_robust_worst_return_pct \
  --replay-eval-run-hourly-policy \
  --replay-eval-robust-start-states flat,long:BTCUSD:0.25,long:ETHUSD:0.25,short:BTCUSD:0.25,short:ETHUSD:0.25 \
  --replay-eval-fill-buffer-bps 5 \
  --replay-eval-hourly-periods-per-year 8760 \
  --post-eval-periods 30,60
```

Remote metadata:

- host: `administrator@93.127.141.100`
- repo: `/nvme0n1-disk/code/stock-prediction`
- env: `.venv313`
- remote PID: `3078275`
- remote log: `analysis/remote_runs/mixed23_robust_probe_20260326_2215/pipeline.log`
- local manifest: `analysis/remote_runs/mixed23_robust_probe_20260326_2215/launch_manifest.json`
- leaderboard target:
  - `pufferlib_market/mixed23_robust_probe_20260326_2215_leaderboard.csv`
- checkpoint root:
  - `pufferlib_market/checkpoints/mixed23_robust_probe_20260326_2215`
- post-train marketsim output:
  - `analysis/remote_runs/mixed23_robust_probe_20260326_2215/marketsim_30_60.csv`

Current live status:

- remote bootstrap completed successfully
- `binding OK` confirmed on the server after a clean rebuild
- `pufferlib_market.autoresearch_rl` has entered trial `[0] reg_combo_2`
- latest observed log line still shows the first 300s training block in progress
- observed live child process:
  - `3079101 /nvme0n1-disk/code/stock-prediction/.venv313/bin/python -u -m pufferlib_market.train --data-path pufferlib_market/data/mixed23_fresh_train.bin ... --checkpoint-dir pufferlib_market/checkpoints/mixed23_robust_probe_20260326_2215/reg_combo_2`

Git sync note:

- I fetched remote refs but did not run a plain `git pull`
- current state remains `main...origin/main [ahead 1, behind 49]` with a dirty worktree
- pulling or rebasing blindly here is unsafe because there are many unrelated local modifications already present

Follow-up repo-hardening fixes after the live batch started:

- `pyproject.toml`
  - added `namespaces = false` under `[tool.setuptools.packages.find]`
  - added NumPy to `[build-system].requires`
- this fixes two separate editable-install bugs:
  - implicit namespace discovery was treating data directories like `forecast_cache/h6` as packages
  - `uv pip install -e .` could fail in build isolation because NumPy headers were not declared as build requirements
- `setup.py`
  - fixed the root build script to compile `pufferlib_market.binding` from repo-relative paths under `pufferlib_market/src/` instead of the broken old `src/` absolute-path assumptions
- `launch_mixed23_retrain.py`
  - replaced the stale top-level copy with a thin wrapper around `scripts.launch_mixed23_retrain.main`
  - this removes the drift that caused me to patch the wrong launcher copy at the start of this pass

Local validation for the packaging/install fixes:

```bash
source .venv313/bin/activate
uv pip install -e . --no-deps
```

- result: editable install now succeeds locally under `.venv313`
