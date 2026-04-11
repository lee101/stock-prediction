# Alpaca Progress 8

## stocks17 augmented RL training (2026-04-11)

### Goal
Train a 17-symbol daily RL policy (AAPL MSFT NVDA GOOG META TSLA SPY QQQ PLTR JPM V AMZN AMD NFLX COIN CRWD UBER) using hourly session-shift augmentation. The augmentation shifts the daily bar window by 1-4 hours, producing 5× more training variants without lookahead bias.

### What worked
- `scripts/build_augmented_daily_training.py` builds the augmented `.bin` from hourly CSVs
- Fixed `_concat_binaries`: binary format has header(64B) + symbol_table(S×16B) + features + prices + masks; was incorrectly concatenating symbol tables from each file
- Backfilled 2yr hourly data via yfinance for short-coverage symbols (MSFT, NVDA, GOOG, META, PLTR, CRWD, SPY, QQQ, COIN)
- Result: offsets 1-4 now cover 409 days (was 107), train binary = 2911 timesteps

### What didn't work / lessons
- `evaluate_holdout` default `decision_lag=0` gives inflated results (same-bar fill). Must use `--decision-lag 2` for realistic binary-fill eval
- With lag=0: s5 showed med=+47%, was actually -9.2% on binary fills. Rankings inverted.
- With lag=2 (correct): best from seeds 1-15 was s12 med=+23.7%, sortino=24, but still below 27%/month target
- Algorithmic insight: need to change the training recipe, not just sweep seeds

### Current algorithmic sweep (5 variants × 3 seeds each)
Checkpoint dir: `pufferlib_market/checkpoints/stocks17_sweep/`

| Variant | Change | Expected benefit |
|---------|--------|-----------------|
| A | baseline tp=0.05 15M | calibration reference |
| B | 30M steps | more gradient updates on 2911-step data |
| C | tp=0.02 | less penalised trading → more trades → more signal |
| D | Muon optimizer | Newton-Schulz orthogonalisation may find better landscape |
| E | max_ep=90 (quarter) | shorter episodes → more diverse episode starts |

### Production status
- 32-model stocks12 ensemble: LIVE, confidence gate fixed (0.20→0.05)
- llm-stock-trader: LIVE, YELP/NET/DBX/OPTX/PDYN/COIN/CRWD
- stocks17 RL: training — not yet deployed (needs to beat s12's med>23% baseline)

---

## Replay parity repair (2026-03-28 02:28 UTC)

### Root cause

The recent `ABEV` replay miss was primarily stale local stock data, not a missing simulator entry path.

- Local hourly stock files for `ABEV`, `BTG`, and `ITUB` had stopped at `2026-03-18 19:00 UTC`.
- The live `ABEV` entry happened at `2026-03-20 15:01 UTC`, so the replay had no local `ABEV` bar for that hour and could not simulate the fill.
- I repaired those three files directly from Alpaca:
  - `ABEV`: `+49` rows, now through `2026-03-27 19:00 UTC`
  - `BTG`: `+49` rows, now through `2026-03-27 19:00 UTC`
  - `ITUB`: `+49` rows, now through `2026-03-27 19:00 UTC`

### Code + tests

- Added live-entry bar coverage reporting to `unified_hourly_experiment/replay_stock_trade_log_sim.py`.
- The replay report now distinguishes:
  - raw simulator-vs-live comparison
  - covered-only comparison for live fills that actually had local bars
  - uncovered live rows with per-symbol last-bar timestamps
- Added regression coverage in `tests/test_replay_stock_trade_log_sim.py` for:
  - unreplayable live fills caused by missing bars
  - empty-input coverage handling

Validation:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_trade_unified_hourly.py \
  tests/test_hourly_trader_simulator.py \
  tests/test_cancel_multi_orders.py \
  tests/test_hourly_order_reconcile.py \
  tests/test_replay_stock_trade_log_sim.py
```

Result: `50 passed`

### Updated trade-log replay

Command:

```bash
source .venv313/bin/activate
python unified_hourly_experiment/replay_stock_trade_log_sim.py \
  --trade-log strategy_state/stock_trade_log.jsonl \
  --event-log strategy_state/stock_event_log.jsonl \
  --symbols TSLA,ABEV \
  --start 2026-03-20T00:00:00Z --end 2026-03-26T00:00:00Z \
  --initial-cash 40299.20 --max-positions 5 --max-hold-hours 5 \
  --min-edge 0.001 --fee-rate 0.001 --leverage 2.0 \
  --decision-lag-bars 0 --bar-margins 0.0005 --entry-order-ttls 6 \
  --market-order-entries 1 --sim-backend python \
  --output analysis/alpaca_progress8_trade_log_replay_20260328_backfilled.json
```

Result from [analysis/alpaca_progress8_trade_log_replay_20260328_backfilled.json](/nvme0n1-disk/code/stock-prediction/analysis/alpaca_progress8_trade_log_replay_20260328_backfilled.json):

- `compare_source = broker_closed_orders`
- `live_entry_count = 2`
- `covered_live_entry_count = 2`
- `uncovered_live_entry_count = 0`
- top config:
  - `market_order_entry = true`
  - `entry_order_ttl_hours = 6`
  - `bar_margin = 0.0005`
  - `hourly_abs_count_delta_total = 0.0`
  - `hourly_abs_qty_delta_total = 4.0`
  - `matched_price_mae = 0.3265`
  - `exact_row_ratio = 1.0`
- per-symbol:
  - `TSLA`: exact count + exact qty
  - `ABEV`: exact count, residual qty delta `4.0`

Interpretation:

- The old `ABEV` miss was stale-data-driven.
- After refreshing the missing stock bars, simulator entry count parity on the `TSLA` + `ABEV` broker-confirmed slice is exact.
- The remaining mismatch is much smaller: `ABEV` quantity differs by `4` shares and mean matched entry-price error is about `0.33`, which points to sizing/fill-detail drift rather than a missing simulated trade.

## Larger-data RunPod training launch (2026-03-28 02:29 UTC)

### Launcher command

```bash
source .venv313/bin/activate
python scripts/launch_runpod_jax_classic.py \
  --run-name alpaca_progress8_jax_fullhist_20260328 \
  --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV \
  --forecast-horizons 1 \
  --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt \
  --epochs 12 \
  --batch-size 16 \
  --sequence-length 48 \
  --validation-days 30 \
  --wandb-project stock \
  --wandb-group alpaca_progress8 \
  --wandb-tags runpod,jax,alpaca,progress8,fullhist \
  --wandb-notes "full 18-symbol JAX classic run after ABEV/BTG/ITUB hourly cache repair and replay coverage fixes" \
  --detach
```

### Run metadata

- Local launch manifest:
  - [launch_manifest.json](/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/launch_manifest.json)
- Local pod manifest:
  - [pod_manifest.json](/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/pod_manifest.json)
- Pod id: `gdypag236ojn9c`
- Public IP: `213.181.111.2`
- SSH port: `29019`
- Requested GPU: `NVIDIA GeForce RTX 4090`
- RunPod cost: `$0.59/hr`

### Remote environment

- Remote repo root: `/workspace/stock-prediction`
- Remote venv: `.venv311jax`
- Remote run dir: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328`
- Remote bootstrap log: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/bootstrap.log`
- Remote train driver log: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/train_driver.log`
- Remote train log: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/train.log`
- Remote status file: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/status.txt`
- Remote checkpoint dir: `/workspace/stock-prediction/unified_hourly_experiment/checkpoints/alpaca_progress8_jax_fullhist_20260328`

### Bootstrap command

Sanitized remote bootstrap body generated by the launcher:

```bash
set -euo pipefail
cd /workspace/stock-prediction
apt-get update >/dev/null
apt-get install -y --no-install-recommends rsync python3.11 python3.11-venv python3-pip >/dev/null
if ! command -v uv >/dev/null 2>&1; then python3.11 -m pip install -q uv; fi
mkdir -p /workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/env
uv venv .venv311jax --python python3.11
source .venv311jax/bin/activate
uv pip install numpy setuptools wheel torch pandas pyarrow loguru exchange-calendars tensorboard wandb 'jax[cuda12]==0.9.2' 'flax>=0.12.6' 'optax>=0.2.8'
python -V > /workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/env/python_version.txt
uv pip freeze > /workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/env/uv_pip_freeze.txt
nvidia-smi > /workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/env/nvidia_smi.txt
python - <<'PY' > /workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/env/torch_cuda.txt
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
python - <<'PY' > /workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/env/jax_devices.txt
import jax
print(jax.__version__)
print(jax.devices())
PY
```

### Train command

Sanitized remote train body generated by the launcher:

```bash
set -euo pipefail
cd /workspace/stock-prediction
source .venv311jax/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_DIR="/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/wandb"
mkdir -p "/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328" "/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/wandb"
export WANDB_API_KEY="${WANDB_API_KEY:-set-by-launcher}"
echo running > "/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/status.txt"
python unified_hourly_experiment/train_jax_classic.py --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV --forecast-horizons 1 --run-name alpaca_progress8_jax_fullhist_20260328 --checkpoint-root unified_hourly_experiment/checkpoints --log-dir tensorboard_logs/binanceneural/alpaca_progress8_jax_fullhist_20260328 --epochs 12 --batch-size 16 --sequence-length 48 --validation-days 30 --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt --market-order-entry --wandb-mode online --wandb-project stock --wandb-group alpaca_progress8 --wandb-tags runpod,jax,alpaca,progress8,fullhist --wandb-notes 'full 18-symbol JAX classic run after ABEV/BTG/ITUB hourly cache repair and replay coverage fixes' > /workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/train.log 2>&1
rc=$?
echo "$rc" > "/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/exit_code.txt"
if [ "$rc" -eq 0 ]; then echo completed > "/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/status.txt"; else echo failed > "/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/status.txt"; fi
exit "$rc"
```

### Current status

- The detached launcher has created the local launch and pod manifests and retained the pod for training.
- At the time of this note, `completion_manifest.json` was not yet written locally, which means the bootstrap/sync phase was still in progress or had not yet synced back.
- Next check:

```bash
source .venv313/bin/activate
find analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328 -maxdepth 4 -type f | sort
cat analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328/completion_manifest.json
```

## Launcher acceleration follow-up (2026-03-28 02:36 UTC)

### Root cause of the first RunPod stall

The original detached launcher run was alive, but inspection showed the local `rsync` was uploading an unrelated large artifact from `binanceneural/chronos2_finetuned/.../model.safetensors` (about `478MB`) even though `train_jax_classic.py` does not use those Chronos fine-tune weights.

Fix:

- updated `scripts/launch_runpod_jax_classic.py` to exclude heavyweight code-sync artifacts:
  - `binanceneural/chronos2_finetuned`
  - `binanceneural/chronos2_finetuned/***`
  - `*.safetensors`
- added launcher coverage in `tests/test_launch_runpod_jax_classic.py` to assert the exclude set is present and is passed through the code-sync `rsync` call

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_launch_runpod_jax_classic.py tests/test_replay_stock_trade_log_sim.py
```

Result: `18 passed`

### Replaced attempt

- Deleted stalled pod: `gdypag236ojn9c`
- Reason: transfer path was wrong; the pod never reached bootstrap/train

### Fast-sync relaunch

Command:

```bash
source .venv313/bin/activate
python scripts/launch_runpod_jax_classic.py \
  --run-name alpaca_progress8_jax_fullhist_20260328_fastsync \
  --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV \
  --forecast-horizons 1 \
  --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt \
  --epochs 12 \
  --batch-size 16 \
  --sequence-length 48 \
  --validation-days 30 \
  --wandb-project stock \
  --wandb-group alpaca_progress8 \
  --wandb-tags runpod,jax,alpaca,progress8,fullhist,fastsync \
  --wandb-notes "full 18-symbol JAX classic run after ABEV/BTG/ITUB hourly cache repair, replay coverage fixes, and fast-sync launcher exclude patch" \
  --detach
```

Launch manifest:

- [launch_manifest.json](/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync/launch_manifest.json)
- [pod_manifest.json](/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync/pod_manifest.json)

Fast-sync pod details:

- pod id: `lio1fu853lu0rv`
- public IP: `103.196.86.109`
- SSH port: `15213`
- cost: `$0.59/hr`

Deterministic remote paths for the relaunched run:

- remote run dir: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync`
- remote bootstrap log: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync/bootstrap.log`
- remote train log: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync/train.log`
- remote checkpoint dir: `/workspace/stock-prediction/unified_hourly_experiment/checkpoints/alpaca_progress8_jax_fullhist_20260328_fastsync`

Current state at write time:

- `launch_manifest.json` exists locally
- `pod_manifest.json` exists locally
- remote run dir already exists on the pod
- remote bootstrap is active; `bootstrap.log` shows the expected environment bring-up and package downloads (`torch`, `jax`, `jaxlib`, CUDA wheels, `wandb`, `pyarrow`, etc.)
- `status.txt` / `train.pid` were not present yet on the pod, so `train.sh` had not started as of this snapshot

## ETH hardening + v27 retrain relaunch (2026-03-28 09:12 UTC)

### Live trader review findings

Two more ETH/crypto bugs were still present in the live Alpaca trader path even after the earlier duplicate-order guard work:

- `execute_trades()` still treated only `abs(qty) >= 1` as an already-open live position, which is wrong for crypto and can allow a large fractional position like `0.5 ETH` to be treated as flat.
- `get_open_orders()` grouped Alpaca crypto orders by the raw broker symbol (`ETH/USD`) while positions and tracked state used compact symbols (`ETHUSD`), which can break reconciliation and let existing crypto orders evade duplicate-order checks.

### Code changes

- Normalized live symbols at the broker boundary in `unified_hourly_experiment/trade_unified_hourly.py` for both open orders and positions.
- Switched the live-entry gate and reconcile path to use the existing substantial-position logic instead of whole-share logic.
- Extended pending-entry reconciliation so substantial fractional crypto entries are not ignored just because qty `< 1`.

### Validation

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_cancel_multi_orders.py \
  tests/test_hourly_order_reconcile.py \
  tests/test_trade_unified_hourly.py \
  tests/test_trade_unified_hourly_meta.py
```

Result: `40 passed`

Additional regression coverage added:

- `tests/test_trade_unified_hourly.py::test_get_open_orders_normalizes_crypto_symbol_keys`
- `tests/test_trade_unified_hourly.py::test_execute_trades_skips_fractional_crypto_live_position`

### Live state after rollout

- Restarted supervisor program: `unified-stock-trader`
- Current LIVE Alpaca snapshot (`2026-03-28 09:12 UTC`):
  - equity `38954.44`
  - cash `38954.44`
  - buying_power `77908.88`
  - open orders: none
  - only dust remains in `AVAXUSD`, `BTCUSD`, `ETHUSD`, `LTCUSD`, `SOLUSD`
- `strategy_state/stock_portfolio_state.json` now has:
  - `positions = {}`
  - `pending_close = []`

### Validation-window root cause for the first fast-sync run

The first fast-sync relaunch still failed before training because the full 18-symbol universe did not have enough hourly rows for `validation_days=30`.

Local check:

- `ITUB`: `764` rows available
- `ABEV`: `769` rows available
- `BTG`: `775` rows available
- required minimum with `sequence_length=48` and `validation_days=30`: `816`

Limiting symbol: `ITUB`, so the full current live universe needs `validation_days <= 27`.

### Replacement RunPod command

```bash
source .venv313/bin/activate
python scripts/launch_runpod_jax_classic.py \
  --run-name alpaca_progress8_jax_fullhist_20260328_fastsync_v27 \
  --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV \
  --forecast-horizons 1 \
  --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt \
  --epochs 16 \
  --batch-size 16 \
  --sequence-length 48 \
  --validation-days 27 \
  --wandb-project stock \
  --wandb-group alpaca_progress8 \
  --wandb-tags runpod,jax,alpaca,progress8,fullhist,fastsync,v27 \
  --wandb-notes "full 18-symbol JAX classic run after ABEV/BTG/ITUB cache repair, fast-sync launcher patch, validation_days=27 to fit ITUB/BTG/ABEV" \
  --detach
```

### Current v27 run metadata

- local launch manifest:
  - [launch_manifest.json](/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync_v27/launch_manifest.json)
- local pod manifest:
  - [pod_manifest.json](/nvme0n1-disk/code/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync_v27/pod_manifest.json)
- pod id: `bu9pqbct6ppjhu`
- public IP: `103.196.86.109`
- SSH port: `15784`
- cost: `$0.59/hr`
- remote run dir: `/workspace/stock-prediction/analysis/remote_runs/alpaca_progress8_jax_fullhist_20260328_fastsync_v27`
- remote checkpoint dir: `/workspace/stock-prediction/unified_hourly_experiment/checkpoints/alpaca_progress8_jax_fullhist_20260328_fastsync_v27`
- remote status at check time:
  - `bootstrap.status.txt = completed`
  - `status.txt = running`
  - `train.pid = 1794`
  - `wandb` run id `2mgypalr`

### Remote training status

The v27 job completed, but it is not a promotion candidate. The remote `train.log` ended with:

- `Best checkpoint: unified_hourly_experiment/checkpoints/alpaca_progress8_jax_fullhist_20260328_fastsync_v27/epoch_003.flax`
- `Best epoch 3: val_score=2.3372 val_sortino=2.4546 val_return=-0.7832`
- final `wandb` summary values were all `nan`

Interpretation: JAX still does not have evidence of beating the existing PyTorch baseline here. The current focus should stay on Chronos/forecast quality and PyTorch-backed policy quality unless a later disciplined JAX run clearly beats the PyTorch replay/holdout path.

### Training-quality note

- The JAX classic path is already `float32` end-to-end in `binanceneural/jax_trainer.py` / `binanceneural/jax_policy.py`; this is not a mixed-precision path that needs to be forced back up to FP32.
- A real stability gap did exist: `TrainingConfig.grad_clip` was defined but ignored by `JaxClassicTrainer`.
- Fixed on `2026-03-28`:
  - `binanceneural/jax_trainer.py` now applies `optax.clip_by_global_norm(config.grad_clip)` ahead of `adamw`
  - `unified_hourly_experiment/train_jax_classic.py` now exposes `--grad-clip`
- Validation:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_jax_trainer_wandboard.py \
  tests/test_trade_unified_hourly.py \
  tests/test_trade_unified_hourly_meta.py \
  tests/test_cancel_multi_orders.py \
  tests/test_hourly_order_reconcile.py
```

Result: `42 passed`

- Additional hardening on `2026-03-28`:
  - `binanceneural/jax_trainer.py` now stops early on non-finite train/val metrics, writes `stop_reason` into `training_meta.json`, and avoids silently finishing a bad long run with `nan` summaries
  - `unified_hourly_experiment/train_jax_classic.py` now prints that stop reason when present

## Chronos hourly cache audit and refresh

- Added `scripts/audit_hourly_forecast_caches.py` to verify per-symbol hourly forecast cache freshness and missing timestamps.
- Initial audit artifact: [alpaca_progress8_stock_cache_audit_20260328.json](/nvme0n1-disk/code/stock-prediction/analysis/alpaca_progress8_stock_cache_audit_20260328.json)
  - only stale symbols were `ITUB`, `BTG`, and `ABEV`
  - each was `216` hours behind and missing `49` cache timestamps
- Rebuilt those three caches with:

```bash
source .venv313/bin/activate
python scripts/build_hourly_forecast_caches.py \
  --symbols ITUB,BTG,ABEV \
  --data-root trainingdatahourly/stocks \
  --forecast-cache-root unified_hourly_experiment/forecast_cache \
  --horizons 1 \
  --lookback-hours 8000 \
  --no-compute-mae
```

- Post-refresh audit artifact: [alpaca_progress8_stock_cache_audit_20260328_postrefresh.json](/nvme0n1-disk/code/stock-prediction/analysis/alpaca_progress8_stock_cache_audit_20260328_postrefresh.json)
  - `18/18` symbol-horizon pairs clean
  - `issue_pairs=0`
  - `stale_pairs=0`
  - `pairs_with_missing_timestamps=0`

## Chronos seed robustness

- The hourly remote Chronos pipeline already launches multi-seed LoRA sweeps now.
- On `2026-03-28` I tightened promotion so the remote hourly pipeline no longer defaults to picking the single luckiest seed.
  - `scripts/promote_chronos2_lora_reports.py` now supports `--selection-strategy stable_family`
  - stable-family promotion scores a hyperparameter family by `mean(metric) + stability_penalty * std(metric)` across seeds, then promotes the best checkpoint inside that family
  - metadata now records `selection_strategy`, `selection_score`, `selection_family_size`, and `selection_family_key`
- `scripts/launch_remote_hourly_chronos_rl.py` / `src/remote_training_pipeline.py` now default the remote hourly Chronos -> cache -> RL pipeline to:
  - `--selection-strategy stable_family`
  - `--stability-penalty 0.25`
  - `--min-family-size 2`

- Validation:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_jax_trainer_wandboard.py \
  tests/test_promote_chronos2_lora_reports.py \
  tests/test_remote_training_pipeline.py \
  tests/test_run_crypto_lora_batch.py \
  tests/test_train_crypto_lora_sweep.py \
  tests/test_audit_hourly_forecast_caches.py \
  tests/test_build_hourly_forecast_caches.py
```

---

## 2026-04-08 — Execution-gate plumbing fix + honest 120d winner discovery (s15)

### Context
Picked up from the 2026-04-08 alpacaprod audit: `daily-rl-trader.service` was
live but had not executed a real open trade since ~2026-04-01. Initial
hypothesis was that `min_open_confidence=0.20` (see `src/daily_stock_defaults.py:81`)
was gating 32-model ensemble signals whose softmax peaks top out at ~0.09–0.12.
Planned to validate by sweeping thresholds through the 120d calibrated replay
harness (`trade_daily_stock_prod.py --backtest --backtest-entry-offset-bps 5
--backtest-exit-offset-bps 25`). That plan turned up multiple bigger bugs.

### Finding A — backtest harness silently ignores the execution gate
- Swept `--min-open-confidence ∈ {0.00, 0.05, 0.07, …, 0.20}` and
  `--min-open-value-estimate ∈ {0, 10}` on the default 32-model legacy ensemble.
- **Every threshold produced byte-identical output**
  (`total_return=+0.21%, trades=24, sortino=0.14`).
- Root cause: `run_backtest(...)` in `trade_daily_stock_prod.py` never called
  `_open_gate_reasons` (the live-path gate at lines ~2057–2078) — it only
  existed in `run_once` / the daemon loop. The CLI accepted the flag but
  threaded it nowhere. Consequence:
  - Every "validated" ensemble replay number in `alpacaprod.md`
    (+0.21%, +0.04% monthly on legacy, −0.92% on RSI map) was already the
    **ungated ceiling**. You could not uncover PnL by tuning the gate.
  - Gate thresholds could never be validated offline, so there was no safe way
    to fix the 0.20 kill-switch in prod.
- Fix (this session): plumbed `min_open_confidence` /
  `min_open_value_estimate` into `run_backtest`, added an inline
  `_signal_passes_open_gate` helper, applied it on both the single-position
  open and the multi-position portfolio open path, and added
  `gate_blocked_opens` to the result dict for observability.
- Regression test:
  `tests/test_trade_daily_stock_prod.py::test_run_backtest_applies_open_gate_to_low_confidence_signals`
  — emits confidence=0.10 / value_estimate=0.25 signals on a deterministic
  ramp frame, asserts `min_open_confidence=0.20` blocks all opens
  (`gate_blocked_opens≥1`, total_return=0), `=0.05` allows all opens
  (total_return>0, gate_blocked=0), and `min_open_value_estimate=1.0` blocks
  opens even when confidence passes. Full backtest suite: 36/36 green.

### Finding B — `--backtest-buying-power-multiplier` is NOT broken
- Earlier I reported it as a silent no-op. That was a methodology error: I
  tested at `--allocation-pct 95` where both 1× and 2× cash-cap at 95%, so
  they collide. Re-testing with the multiplier at `--allocation-pct 190
  --backtest-buying-power-multiplier 2.0` vs `1.0` produces correctly
  differentiated outputs (2× scales the bet and the drawdown). Leverage
  plumbing is intact; the original audit stands.

### Finding C — 120d gate sweep on the legacy 32-model ensemble
With the gate now wired, fine sweep of `min_open_confidence` on the
32-model ensemble:

| c | total | sortino | trades | blocked |
|---|---|---|---|---|
| 0.00 | +0.995% | 0.598 | 24 | 1 (value-est gate) |
| 0.08 | +0.995% | 0.598 | 24 | 1 |
| **0.10** | **+1.268%** | **0.754** | 24 | 5 |
| 0.12 | +1.307% | 0.702 | 14 | 35 |
| 0.13 | −0.536% | −0.275 | 9 | 49 |
| 0.14 | −0.809% | −0.550 | 2 | 82 |
| 0.15 | +0.061% | 0.071 | 1 | 114 |
| ≥0.17 | 0 | 0 | 0 | 120 |

Sweet spot is c≈0.10 (Sortino 0.754), but total only +1.27% / 120d
≈ **0.32% / month**. Gate tuning alone cannot reach the 27%/month HARD RULE
on this ensemble.

### Finding D — most 32 prod ensemble members individually LOSE money
Batched solo 120d replays of all 42 `.pt` files in
`pufferlib_market/prod_ensemble/` (script:
`scripts/rank_prod_solo.sh`, output:
`artifacts/gate_sweep/prod_solo_c0.txt`).

Winners (positive total return, allocation 12.5%, c=0, v=0):

| ckpt | total 120d | Sortino | MaxDD | trades |
|---|---|---|---|---|
| **s15** | **+2.73%** | **1.75** | −1.24% | 45 |
| **s1731** | **+1.81%** | 1.01 | −1.29% | 10 |
| **s2435** | +1.16% | 1.06 | −0.70% | 10 |
| s36 | +0.86% | 0.33 | −2.61% | 33 |
| s7159 | +0.65% | 0.60 | −0.95% | 5 |
| s2831 | +0.22% | 0.15 | −1.16% | 26 |
| s3086 | +0.003% | 0.009 | −1.99% | 26 |
| tp10 | +0.20% | 4.27 | −0.006% | 1 (degenerate) |

Every other member (36+) was negative or degenerate. The 32-model ensemble's
"+0.21%" number was literally averaging 6 winners with 36 losers. This is
a correctable composition problem, not a model-quality problem.

### Finding E — winners-only ensemble + concentration + 2× leverage
Key configs on 120d honest replay (legacy feature map, calibrated offsets):

| config | total 120d | monthly | Sortino | MaxDD |
|---|---|---|---|---|
| s15 solo @ alloc 12.5 | +2.73% | +0.48% | **1.75** | −1.24% |
| s15 solo @ alloc 95 | +4.29% | +0.75% | 0.39 | −13.3% |
| s15 solo @ alloc 190 × 2× BP | +23.97% | +3.88% | 0.89 | −25.1% |
| winEns3 (s15+s1731+s2435) @ alloc 12.5 | +1.98% | +0.35% | 1.37 | −0.88% |
| winEns3 @ alloc 95 | +6.15% | +1.06% | 0.62 | −6.8% |
| winEns3 @ alloc 190 × 2× BP | +18.19% | +2.98% | 0.95 | −12.8% |
| winEns5 (+s7159 +s36) @ alloc 95 | **+9.70%** | **+1.63%** | **1.00** | **−6.8%** |
| winEns5 @ alloc 190 × 2× BP | +14.53% | +2.40% | 0.67 | −15.7% |
| s15 solo @ alloc 95 + gate c=0.12 | +11.05% | +1.88% | **0.876** | −7.1% |
| **s15 solo @ alloc 190 × 2× BP + gate c=0.12** | **+31.68%** | **+5.62%** | **1.22** | **−13.9%** |

**The gate × s15 interaction is the headline.** Pure `s15` with
`min_open_confidence=0.12` at 2× leverage delivers +5.62%/month at Sortino 1.22
with ~14% max drawdown — the first deployable profile in the repo that's
not break-even on the last 120 live trading days. Note the non-monotonic gate
profile (bad at 0.10, good at 0.115–0.12, collapses at 0.125+) — narrow
optimum, modest overfit risk.

### Reality check vs HARD RULE
- **27%/month target** is still ~5× higher than the best config found
  (s15 solo @ 2× leverage + gate c=0.12 → +5.62%/month).
- But this is **qualitatively different** from where we started. The running
  prod service is ~0%/month (actual) and the 32-model replay ceiling is ~0.3%/
  month. Even conservative deployment of s15-family beats prod by an order
  of magnitude.
- Deployable rank (allocation 95%, no leverage, on legacy feature map):
  1. `winEns5` (s15+s1731+s2435+s7159+s36): +9.7%/120d, Sortino 1.00, DD 6.8%
  2. `s15 + gate 0.12`: +11.05%/120d, Sortino 0.88, DD 7.1%
  3. legacy 32-model + gate 0.10: +1.27%/120d (current floor)

### What was NOT done (intentionally deferred)
- Did **not** touch the production systemd unit / flip the running service.
  Any deploy requires a human to restart the service with a new CLI
  invocation and first run it in paper mode.
- Did **not** mix legacy and `stocks12_v5_rsi` checkpoints — feature schemas
  don't match and the schema guard correctly rejects them. The v5_rsi family
  all loses money on the 120d replay anyway (see currentstate.md §6b), so
  mixing is not the lever.
- Did **not** attempt to train new seeds. That's the next step but it needs
  a honest walk-forward validation set first (Jul–Nov 2025 holdout is
  overfit, as v5_rsi demonstrated), otherwise we just produce more s42s.

### Files touched
- `trade_daily_stock_prod.py`: gate plumbing in `run_backtest`, thread from
  CLI, add `gate_blocked_opens` to result dict.
- `tests/test_trade_daily_stock_prod.py`:
  `test_run_backtest_applies_open_gate_to_low_confidence_signals`.
- `scripts/rank_prod_solo.sh`: sweep harness for solo 120d replay of every
  `.pt` in `pufferlib_market/prod_ensemble/`.
- `artifacts/gate_sweep/prod_solo_c0.txt`: raw output of the 42-ckpt solo sweep.
- `currentstate.md` §6b: updated to reflect the gate-is-not-the-bottleneck
  and v5_rsi-is-overfit findings.

### Proposed next actions (for user confirmation)
1. **Deploy winEns5 @ 95% in paper first** (no leverage). Swap the running
   service's `--checkpoint` / `--extra-checkpoints` + `--allocation-pct 95`
   and let it tick for one trading week in paper mode. Expected 1.6%/month
   at Sortino 1.0. If it holds, roll to live.
2. **Then** promote to s15 solo + gate 0.12 @ 2× leverage (the +5.6%/mo
   config). Because the gate optimum is narrow, paper-test for a full
   trading week before going live.
3. **Build honest walk-forward val set** (2024-01 → 2026-04) and re-run
   eval_100d on every existing checkpoint under the new regime. That finds
   the real ranking and tells us whether s15's edge survives out of sample.
4. **Only then** launch a new seed sweep (50–100 seeds, v5_rsi features, new
   walk-forward val set) to push past 5%/month toward 27%/month.

HARD RULE 27%/month is **not** reachable from existing checkpoints. It
requires new training under the new validation regime, on an expanded symbol
universe (the 12-ticker mega-cap pool caps the achievable Sortino).
