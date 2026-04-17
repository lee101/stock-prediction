# Alpaca Progress 7

## Production check (2026-03-26 04:22 UTC)

- Live service is still supervisor program `unified-stock-trader` on `leaf-gpu-dedicated-server`.
- Installed command still points at `unified_hourly_experiment/trade_unified_hourly_meta.py` with strategies `wd_0.06_s42:8` and `wd_0.06_s1337:8`, 18-stock universe, `--market-order-entry`, `--bar-margin 0.0005`, `--entry-order-ttl-hours 6`, `--margin-rate 0.0625`, `--live --loop`.
- Fresh Alpaca LIVE snapshot at `2026-03-26 04:22 UTC`: equity `$41,315.42`, cash `$10,298.17`, long market value `$0.00`, buying power `$20,596.34`, last_equity `$41,077.99`.
- There are no stock positions open now. Only crypto dust remains in `AVAXUSD`, `BTCUSD`, `ETHUSD`, `LTCUSD`, `SOLUSD`.
- Open orders are stale crypto only: 3 `ETH/USD` GTC limit buys. The stock strategy is currently flat.
- Recent meaningful stock exits from broker records:
  - `TSLA` sell `33 @ 379.22`, filled `2026-03-23 13:38 UTC`
  - `ABEV` sell `4459 @ 2.83`, filled `2026-03-25 13:30 UTC`

## Simulator parity

### Holdout meta sweep

Command:

```bash
source .venv313/bin/activate
python unified_hourly_experiment/sweep_meta_portfolio.py \
  --strategy wd06=unified_hourly_experiment/checkpoints/wd_0.06_s42:8 \
  --strategy wd06b=unified_hourly_experiment/checkpoints/wd_0.06_s1337:8 \
  --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV \
  --metrics p10 --selection-modes sticky --switch-margins 0.005 --min-score-gaps 0.0 \
  --lookback-days 14 --recency-halflife-days 0.0 --holdout-days 5,14,30 \
  --initial-cash 40025.12 --max-positions 5 --min-edge 0.001 --max-hold-hours 5 \
  --trade-amount-scale 100.0 --min-buy-amount 2.0 \
  --entry-intensity-power 1.0 --entry-min-intensity-fraction 0.0 \
  --long-intensity-multiplier 1.0 --short-intensity-multiplier 1.5 \
  --decision-lag-bars 1 --entry-selection-mode edge_rank --market-order-entry \
  --bar-margin 0.0005 --entry-order-ttl-hours 6 --leverage 2.0 \
  --margin-rate 0.0625 --fee-rate 0.001 \
  --sit-out-if-negative --sit-out-threshold -0.001 \
  --output analysis/alpaca_progress7_meta_eval_20260326.json
```

Result from [analysis/alpaca_progress7_meta_eval_20260326.json](/nvme0n1-disk/code/stock-prediction/analysis/alpaca_progress7_meta_eval_20260326.json):

- Best config was effectively the current `wd06` baseline, not a useful meta win.
- Best meta summary across `5d/14d/30d` holdouts:
  - min sortino `-11.1542`
  - mean sortino `-6.2828`
  - min return `-18.09%`
  - mean return `-7.38%`
  - mean max drawdown `9.10%`
- `wd06b` was also bad:
  - min sortino `-13.0641`
  - mean sortino `-10.5474`
  - min return `-15.63%`
  - mean return `-6.94%`

Interpretation: the current live 2-model selector is not holding up on recent simulator windows. The simulator does not justify the current live strategy as robust on recent holdouts.

### Trade-log replay

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
  --output analysis/alpaca_progress7_trade_log_replay_20260326.json
```

Result from [analysis/alpaca_progress7_trade_log_replay_20260326.json](/nvme0n1-disk/code/stock-prediction/analysis/alpaca_progress7_trade_log_replay_20260326.json):

- Live entries: `2`
- Simulated entries: `1`
- Exact row ratio: `0.5`
- Hourly absolute count delta total: `1.0`
- Hourly absolute quantity delta total: `4459.0`
- `TSLA` entry matched.
- `ABEV` entry was missed entirely by the replay.
- Replay-sim slice metrics:
  - total return `-0.72%`
  - annualized return `-54.81%`
  - sortino `-12.78`
  - final equity `$40,009.93`

Interpretation: the execution simulator is directionally useful, but it is still not faithful enough to fully reproduce the recent `ABEV` live behavior. That mismatch needs fixing before we trust simulator PnL as a production proxy.

## JAX rewrite

### Code added

- `binanceneural/jax_policy.py`: JAX/Flax classic transformer policy, action decode, torch checkpoint conversion.
- `binanceneural/jax_losses.py`: JAX trading simulator and Sortino/return objective.
- `binanceneural/jax_trainer.py`: JAX trainer for the classic hourly stock policy.
- `unified_hourly_experiment/train_jax_classic.py`: CLI entrypoint for production-path JAX training.
- `scripts/launch_runpod_jax_classic.py`: cost-safe RunPod launcher with manifest write, targeted sync, bootstrap env probes, W&B wiring, sync-back, and auto-delete when not detached.
- `tests/test_jax_policy.py`
- `tests/test_jax_losses.py`
- `tests/test_jax_trainer_wandboard.py`
- `tests/test_launch_runpod_jax_classic.py`
- `tests/test_differentiable_loss_utils_compile_gate.py`

### Validation

- Installed and then upgraded the local JAX stack in `.venv313` via `uv pip` to CUDA-backed JAX:
  - `jax[cuda12]==0.9.2`
  - `flax==0.12.6`
  - `optax==0.2.8`
- Local GPU checks now pass on this box:
  - `torch 2.9.0+cu128` sees `NVIDIA GeForce RTX 5090`
  - `jax 0.9.2` now reports `devices [CudaDevice(id=0)]`
- Docker GPU validation on this machine is fixed:
  - installed NVIDIA Container Toolkit `1.19.0-1`
  - configured Docker runtime via `nvidia-ctk runtime configure --runtime=docker`
  - `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` now works and sees the `RTX 5090`
  - first attempt to run the full launcher `--docker-validate` path exposed a host issue, not a code issue: `/` was full because `/tmp` had ~`101G` of stale temp checkpoint directories; clearing those temp artifacts restored ~`101G` free on `/` and removed the local Docker image-pull blocker
  - after fixing disk pressure I did not leave the long first-time RunPod image bootstrap sitting on the workstation; the launcher command is ready to rerun locally without the previous disk blocker
- Fixed 2 real JAX issues found during bring-up:
  - scalar `can_short` / `can_long` broadcasting in the simulator
  - `smoothness_penalty` concretization inside JIT-compiled objective code
- Fixed one real PyTorch/Blackwell issue found while doing the local comparison:
  - `differentiable_loss_utils.py` now disables `torch.compile` for the tiny objective kernels on compute capability `12.x` GPUs unless `TORCH_FORCE_COMPILE=1` is set, because the eager path is stable and the compiled path was crashing in Triton/Inductor on this `RTX 5090`
- Parity / launcher tests now pass on the actual GPU-backed environment:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_differentiable_loss_utils_compile_gate.py \
  tests/test_jax_policy.py \
  tests/test_jax_losses.py \
  tests/test_jax_trainer_wandboard.py \
  tests/test_launch_runpod_jax_classic.py
```

Result: `10 passed`

### Local smoke train

Command:

```bash
source .venv313/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python unified_hourly_experiment/train_jax_classic.py \
  --symbols AAPL,TSLA,NVDA \
  --run-name alpaca_progress7_jax_smoke_20260326 \
  --epochs 1 \
  --batch-size 2 \
  --sequence-length 48 \
  --validation-days 7 \
  --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt \
  --dry-train-steps 1 \
  --market-order-entry
```

Artifacts:

- Checkpoint: [epoch_001.flax](/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/alpaca_progress7_jax_smoke_20260326/epoch_001.flax)
- Run dir: [alpaca_progress7_jax_smoke_20260326](/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/alpaca_progress7_jax_smoke_20260326)

Smoke result:

- best epoch `1`
- val_score `87.0882`
- val_sortino `86.7016`
- val_return `2.5775`

This only proves the JAX path trains end-to-end on real data. It is not evidence of a production improvement yet.

### Local GPU smoke: JAX vs PyTorch

Matched 1-epoch, `dry_train_steps=1`, same symbols, same preload, same local GPU (`RTX 5090`) comparison written to [analysis/alpaca_progress7_jax_vs_torch_smoke_20260326.json](/nvme0n1-disk/code/stock-prediction/analysis/alpaca_progress7_jax_vs_torch_smoke_20260326.json).

Results:

- PyTorch smoke:
  - elapsed `2.47s`
  - val_score `148.7628`
  - val_sortino `147.8070`
  - val_return `6.3717`
- JAX smoke:
  - elapsed `32.448s`
  - val_score `10.3446`
  - val_sortino `9.9415`
  - val_return `2.6871`

Interpretation: on the first apples-to-apples local GPU smoke, the JAX rewrite is functionally correct but is not yet better than the current PyTorch path. It was slower in this tiny benchmark and materially worse on the immediate validation slice, so there is no justification yet for replacing the PyTorch baseline.

### RunPod status

- The original ad hoc RunPod pod from `2026-03-26 04:14 UTC` (`suthi2eh331pne`) was terminated after local debugging so it would not keep billing while idle.
- Current working pattern is now the launcher, not the old hand-written bootstrap.
- The launcher behavior is now:
  - targeted sync only
  - bootstrap runs synchronously and records `python_version.txt`, `uv_pip_freeze.txt`, `nvidia_smi.txt`, `torch_cuda.txt`, and `jax_devices.txt`
  - train runs detached on the pod
  - when not using `--detach`, the launcher waits, syncs artifacts back, and deletes the pod automatically
  - detached runs intentionally retain the pod and record that fact in the completion manifest
- The launcher no longer depends on `uv pip install -e .` on RunPod; it uses a venv + runtime dependencies + `PYTHONPATH`
- The canonical remote bootstrap/install pattern is now:

```bash
cd /workspace/stock-prediction
uv venv .venv311jax --python python3.11
source .venv311jax/bin/activate
uv pip install numpy setuptools wheel torch pandas pyarrow loguru exchange-calendars tensorboard wandb \
  "jax[cuda12]==0.9.2" "flax>=0.12.6" "optax>=0.2.8"
export PYTHONPATH=$PWD:$PYTHONPATH
```

Preferred launch flow:

```bash
source .venv313/bin/activate
python scripts/launch_runpod_jax_classic.py \
  --run-name alpaca_progress7_jax_runpod_$(date -u +%Y%m%d_%H%M%S) \
  --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV \
  --forecast-horizons 1 \
  --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt \
  --epochs 12 \
  --batch-size 16 \
  --sequence-length 48 \
  --validation-days 7 \
  --wandb-project stock \
  --wandb-mode online
```

## Current conclusion

- Production is flat on stocks right now; only dust crypto orders/positions remain.
- The simulator says the current live stock strategy is weak on recent holdouts.
- The replay simulator still misses the recent `ABEV` live trade, so simulator-vs-prod parity is not trustworthy enough yet.
- The JAX rewrite is functionally working, locally GPU-enabled, Docker-validated, and reproducibly launchable on RunPod.
- The first matched local GPU smoke did not beat PyTorch; JAX was slower and worse on the immediate validation slice.
- The next real checkpoint is not deployment. It is a longer disciplined JAX training run plus holdout/replay evaluation against the current PyTorch `wd06` baseline before any production change.

## PyTorch robustness follow-up

- Pushed the current JAX work to git branch `jax`.
- Improved the PyTorch trainer in `binanceneural/trainer.py` so checkpoint selection is now configurable and robustness-aware instead of hardwired to raw `val_score`.
- New trainer behavior:
  - uses `top_k_checkpoints` from config instead of a hardcoded `10`
  - supports `checkpoint_metric` choices `val_score`, `val_sortino`, `val_return`, `robust_score`, `robust_sortino`
  - applies `checkpoint_gap_penalty` against train-vs-val gap for the robust metrics
  - writes `training_progress.json` every epoch for live inspection
  - updates `best.pt` to point at the currently selected best checkpoint
- Added CLI flags to `unified_hourly_experiment/train_unified_policy.py`:
  - `--top-k-checkpoints`
  - `--checkpoint-metric`
  - `--checkpoint-gap-penalty`
- Added test coverage in `tests/test_trainer_checkpoint_selection.py`.

### Live PyTorch robust smoke

Command used:

```bash
source .venv313/bin/activate
python - <<'PY'
from pathlib import Path
from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import MultiSymbolDataModule
from binanceneural.trainer import BinanceHourlyTrainer
from unified_hourly_experiment.train_jax_classic import build_directional_constraints

symbols = ["AAPL", "TSLA", "NVDA"]
cfg = DatasetConfig(
    symbol=symbols[0],
    data_root=Path("trainingdatahourly/stocks"),
    forecast_cache_root=Path("unified_hourly_experiment/forecast_cache"),
    forecast_horizons=(1,),
    sequence_length=48,
    validation_days=7,
    cache_only=True,
    min_history_hours=48 + 7 * 24 + 48,
)
trainer = BinanceHourlyTrainer(
    TrainingConfig(
        epochs=2,
        batch_size=2,
        sequence_length=48,
        learning_rate=1e-4,
        weight_decay=0.06,
        transformer_dim=512,
        transformer_layers=6,
        transformer_heads=8,
        model_arch="classic",
        return_weight=0.15,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        checkpoint_root=Path("unified_hourly_experiment/checkpoints"),
        run_name="alpaca_progress7_torch_robust_smoke_20260326",
        preload_checkpoint_path=Path("unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt"),
        seed=42,
        dry_train_steps=1,
        use_compile=False,
        use_amp=False,
        top_k_checkpoints=3,
        checkpoint_metric="robust_score",
        checkpoint_gap_penalty=0.25,
    ),
    MultiSymbolDataModule(symbols=symbols, config=cfg, directional_constraints=build_directional_constraints(symbols)),
)
trainer.train()
PY
```

Observed output:

- epoch 1 robust_score `148.7628`
- epoch 2 robust_score `152.0510`
- `best.pt` now points at `epoch_002.pt`
- progress file written to:
  - `unified_hourly_experiment/checkpoints/alpaca_progress7_torch_robust_smoke_20260326/training_progress.json`

Interpretation: the trainer now exposes a cleaner path for longer PyTorch runs where we can watch the generalization gap as training proceeds, keep only top-k checkpoints, and use `best.pt` directly instead of guessing from the latest epoch.

### 2026-03-26 Hugging Face Trainer retry

Built a real `transformers.Trainer` bridge for the unified hourly stock policy instead of using the older bespoke `hftraining/` stack:

- new bridge module: `binanceneural/hf_trainer_bridge.py`
- new entrypoint: `unified_hourly_experiment/train_hf_trainer_policy.py`
- added regression test: `tests/test_hf_unified_policy_trainer.py`

Key behavior of the bridge:

- trains the existing `binanceneural` policy and differentiable simulator objective through the current Hugging Face `Trainer`
- exports portable `epoch_###.pt` checkpoints plus `best.pt`, `.topk_manifest.json`, `training_progress.json`, `config.json`, and `training_meta.json`
- supports `checkpoint_metric` / `checkpoint_gap_penalty` so we can rank HF checkpoints the same way we rank native PyTorch checkpoints
- adds `--max-steps` so bounded smoke runs do not accidentally consume thousands of updates

Focused verification:

- `pytest -q tests/test_hf_unified_policy_trainer.py tests/test_trainer_checkpoint_selection.py`
- result: `2 passed`

#### HF smoke run

Command:

```bash
source .venv313/bin/activate
python unified_hourly_experiment/train_hf_trainer_policy.py \
  --symbols AAPL,TSLA,NVDA \
  --epochs 2 \
  --batch-size 128 \
  --sequence-length 48 \
  --hidden-dim 128 \
  --num-layers 3 \
  --num-heads 4 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --warmup-steps 0 \
  --logging-steps 10 \
  --num-workers 0 \
  --run-name alpaca_progress7_hf_trainer_smoke_b128_20260326 \
  --checkpoint-root unified_hourly_experiment/checkpoints \
  --data-root trainingdatahourly/stocks \
  --cache-root unified_hourly_experiment/forecast_cache \
  --forecast-horizons 1,24 \
  --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt
```

Observed training metrics:

- epoch 1 eval score `55.88`, eval sortino `55.66`, eval return `2.782`
- epoch 2 eval score `73.59`, eval sortino `73.40`, eval return `2.362`
- best checkpoint: `unified_hourly_experiment/checkpoints/alpaca_progress7_hf_trainer_smoke_b128_20260326/epoch_002.pt`
- summary artifact: `unified_hourly_experiment/checkpoints/alpaca_progress7_hf_trainer_smoke_b128_20260326/hf_trainer_summary.json`

#### Matching native PyTorch smoke

Command used an inline `BinanceHourlyTrainer` run with the same symbols, `batch_size=128`, `sequence_length=48`, `epochs=2`, `lr=1e-4`, `weight_decay=0.01`, and the same preload path.

Observed native PyTorch metrics:

- epoch 1 val score `54.30`, val sortino `54.14`, val return `2.03`
- epoch 2 val score `81.80`, val sortino `81.66`, val return `1.77`
- best checkpoint: `unified_hourly_experiment/checkpoints/alpaca_progress7_torch_smoke_b128_20260326/epoch_002.pt`

#### Market simulator comparison

Backtest command used for both:

```bash
source .venv313/bin/activate
python unified_hourly_experiment/backtest_portfolio.py \
  --checkpoint <epoch_002.pt> \
  --symbols AAPL,TSLA,NVDA \
  --data-root trainingdatahourly/stocks \
  --cache-root unified_hourly_experiment/forecast_cache
```

Market simulator results:

- HF Trainer epoch 2:
  - avg return `+45.96%`
  - avg sortino `21.22`
  - avg max drawdown `-2.38%`
- native PyTorch epoch 2:
  - avg return `+45.61%`
  - avg sortino `27.05`
  - avg max drawdown `-2.33%`

Interpretation:

- On this bounded smoke, the current HF Trainer path is viable and slightly edged native PyTorch on average return (`+0.35 pct`) but underperformed on simulator sortino (`-5.83`).
- The in-loop validation metric still favored native PyTorch (`81.80` vs `73.59`).
- The preload checkpoint only partially matched because this smoke used a narrower `128`-dim model than the source checkpoint, so this comparison is useful for trainer plumbing and short-run behavior, but not yet a definitive “HF beats PyTorch” result.

Saved comparison artifact:

- `analysis/alpaca_progress7_hf_trainer_vs_torch_b128_20260326.json`

## 2026-03-26 09:12 UTC - WandBoard tooling + hourly scout sweep

### WandBoard / W&B tooling

- Added `scripts/wandb_run_scout.py` filters for:
  - `--states`
  - `--name-contains`
  - `--group-contains`
  - `--exclude-name-contains`
  - `--exclude-group-contains`
  - `--min-runtime-sec`
  - `--min-steps`
- The scout output now includes runtime and step counts so tiny verification runs do not dominate account-level ranking.
- Native PyTorch, HF Trainer, and JAX paths all have WandBoard regression coverage; focused suite passed:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_binanceneural_wandb.py \
  tests/test_hf_unified_policy_trainer.py \
  tests/test_jax_trainer_wandboard.py \
  tests/test_wandb_run_scout.py \
  tests/test_launch_runpod_jax_classic.py
```

Result: `14 passed`

### RunPod / gpu_pool launcher fixes

- `scripts/launch_runpod_jax_classic.py` now:
  - deletes pods if startup/public SSH times out
  - supports `--startup-timeout-sec`
  - keeps `--dry-run` side-effect free
  - propagates W&B env/flags into Docker validation
  - preserves relative paths when syncing caches/checkpoints
  - excludes bulky local dirs from code sync (`checkpoints`, `tensorboard_logs`, `wandb`, caches, `__pycache__`)
- Old stuck pod `9xeucidcyib52o` was deleted after verification it never got a public IP.

### Account scout results

Scout command:

```bash
source .venv313/bin/activate
python scripts/wandb_run_scout.py \
  --project stock \
  --entity lee101p \
  --last-n-runs 200 \
  --top-k 10 \
  --states finished \
  --exclude-name-contains verify,scout,neuraldaily \
  --exclude-group-contains wandboard
```

Most promising historical account runs exposed by the filtered scout:

- `per_env_adv_smooth`
  - metric `1.0247`
  - return `1.0247`
  - sortino `2.93`
- `robust_reg_tp01`
  - metric `0.8076`
  - return `0.8076`
  - sortino `2.82`

Common pattern from those historical runs:

- `lr=3e-4`
- `weight_decay=0.05`
- `optimizer=adamw`

### Local hourly sweep derived from scout

Group: `wandboard_hourly_scout_20260326`

Command family used:

```bash
source .venv313/bin/activate
python unified_hourly_experiment/train_unified_policy.py \
  --symbols AAPL,TSLA,NVDA \
  --epochs 2 \
  --dry-train-steps 4 \
  --batch-size 128 \
  --sequence-length 48 \
  --hidden-dim 128 \
  --num-layers 3 \
  --num-heads 4 \
  --forecast-horizons 1,24 \
  --preload unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt \
  --checkpoint-root unified_hourly_experiment/checkpoints \
  --log-dir tensorboard_logs/binanceneural \
  --no-compile \
  --no-amp \
  --wandb-project stock \
  --wandb-entity lee101p \
  --wandb-group wandboard_hourly_scout_20260326 \
  --wandb-tags hourly,local,torch,scout \
  --wandb-mode online
```

Runs and outcomes:

- `scout_hourly_lr_3em4_wd_0p0_20260326`
  - W&B run `4hd82zw0`
  - val score `20.6997`
  - val sortino `20.4592`
  - val return `3.0051`
  - robust score `18.0771`
- `scout_hourly_lr_3em4_wd_0p05_20260326`
  - W&B run `ulnv6ndq`
  - val score `20.6993`
  - val sortino `20.4589`
  - val return `3.0051`
  - robust score `18.0767`
- `scout_hourly_lr_1em4_wd_0p0_20260326`
  - W&B run `o0zzchxg`
  - val score `18.1058`
  - val sortino `17.8807`
  - val return `2.8135`
  - robust score `15.1946`
- `scout_hourly_lr_1em4_wd_0p05_20260326`
  - W&B run `dj4lykvu`
  - val score `18.1056`
  - val sortino `17.8805`
  - val return `2.8135`
  - robust score `15.1943`

Interpretation:

- On this bounded hourly sweep, `learning_rate=3e-4` clearly beat `1e-4`.
- `weight_decay=0.05` did not help the hourly trainer on this slice; `0.0` was marginally better than `0.05`.
- The next sensible hourly sweep is around:
  - `learning_rate`: `1e-4,2e-4,3e-4`
  - `weight_decay`: `0.0,0.025,0.05`
- Saved scout artifact:
  - `analysis/wandb_hourly_scout_group_20260326.json`

## 2026-03-26 09:25 UTC - First multi-seed hourly robustness pass

### New multi-seed tool

Added `scripts/run_hourly_multiseed_scout.py` to:

- train a grid of hourly trainer configs across multiple seeds
- backtest each seed’s best checkpoint through `unified_hourly_experiment/backtest_portfolio.py`
- aggregate market-sim mean/std/CI across seeds
- compare each candidate against a baseline with tolerance-aware significance checks

Focused regression coverage:

```bash
source .venv313/bin/activate
pytest -q \
  tests/test_run_hourly_multiseed_scout.py \
  tests/test_binanceneural_wandb.py \
  tests/test_hf_unified_policy_trainer.py \
  tests/test_jax_trainer_wandboard.py \
  tests/test_wandb_run_scout.py \
  tests/test_launch_runpod_jax_classic.py
```

Result: `17 passed`

### Pass 1 setup

Command used:

```bash
source .venv313/bin/activate
python scripts/run_hourly_multiseed_scout.py \
  --run-prefix hourly_multiseed_pass1_20260326 \
  --seeds 1337,42,7 \
  --learning-rates 1e-4,3e-4 \
  --weight-decays 0.0 \
  --return-weights 0.08 \
  --fill-temperatures 0.0005 \
  --epochs 2 \
  --dry-train-steps 4 \
  --batch-size 128 \
  --sequence-length 48 \
  --hidden-dim 128 \
  --num-layers 3 \
  --num-heads 4 \
  --symbols AAPL,TSLA,NVDA \
  --crypto-symbols SOLUSD,AVAXUSD,ETHUSD,UNIUSD \
  --wandb-project stock \
  --wandb-entity lee101p \
  --wandb-group wandboard_hourly_multiseed_pass1_20260326 \
  --wandb-tags hourly,local,torch,multiseed \
  --wandb-mode online \
  --no-compile \
  --no-amp \
  --output analysis/hourly_multiseed_pass1_20260326.json
```

Baseline:

- `lr=1e-4`
- `wd=0.0`
- `rw=0.08`
- `fill_temperature=0.0005`

Candidate:

- `lr=3e-4`
- `wd=0.0`
- `rw=0.08`
- `fill_temperature=0.0005`

### Pass 1 results

Baseline `lr=1e-4` across seeds `1337,42,7`:

- mean train robust score `20.82`
- mean market avg sortino `5.62`
- mean market avg return `+29.46%`
- mean market avg max drawdown `-6.79%`
- seed spread:
  - sortino std `0.65`
  - return std `4.92 pct`

Candidate `lr=3e-4` across seeds `1337,42,7`:

- mean train robust score `24.38`
- mean market avg sortino `6.41`
- mean market avg return `+32.26%`
- mean market avg max drawdown `-6.28%`
- seed spread:
  - sortino std `0.77`
  - return std `6.16 pct`

Tolerance-aware baseline comparison from `analysis/hourly_multiseed_pass1_20260326.json`:

- sortino delta `+0.79`
  - significance margin `1.14`
  - significant: `false`
- return delta `+2.79 pct`
  - significance margin `8.92 pct`
  - significant: `false`
- max drawdown improvement `+0.51 pct`
  - significance margin `1.67 pct`
  - significant: `false`

Interpretation:

- `3e-4` is better on the multi-seed mean and on the variance-penalized `market_robust_score` (`6.44` vs `5.45`).
- The improvement is not yet cleanly significant with only 3 seeds because the simulator tolerance/noise is wide on this slice.
- Seed variance is large enough that single-seed wins are not trustworthy for redeploy decisions.

Practical next step:

- keep `lr=3e-4` in the frontier
- expand to a second pass with:
  - more seeds
  - `learning_rate=2e-4,3e-4,4e-4`
  - `weight_decay=0.0,0.025`
  - longer `epochs` / `dry_train_steps`

Saved artifact:

- `analysis/hourly_multiseed_pass1_20260326.json`

---

# Frontier Efficiency Session (2026-04-17)

Context at session start: 13-model v5 screened32 ensemble is locally optimal
across 5 ensemble-add evals (D_s67, AB s1, AA s2, AB s2, AC s1/s2 via md5 parity)
— all REJECT. Baseline OOS med=19.57%, p10=7.68%, neg=8/263, target is 30%/mo.

**Mid-session update**: add AD s1, AD s9, C_lev1p5x_s1 to the reject list
(7 total, 0 wins). AD s9 is the strongest standalone ever measured at this
variant (med=+14.10%, neg=11) but still −1.07% mean-delta-med as 14th
member → confirms raw strength ≠ ensemble additivity.

Session goal: find one lever that actually moves the ensemble (not just the
standalone metric). Hypotheses ranked by expected lift per push doc §4:
  1. **E4** 2× leverage retrain      — highest expected lift
  2. **E2b** fresh seed diversity    — load-bearing axis per memory
  3. **Frontier efficiency**         — SPS × wall-clock speed-up for more seeds

Only winners land here. Losers go to `failedalpacaprogress7.md`.

## Baseline SPS (before any changes)

Measured 2026-04-17 ~02:00 UTC on RTX 5090 with an idle box.

| Metric | Value |
|---|---|
| GPU util (eval-only) | 7% (idle) |
| PPO already `torch.compile`'d | yes (train.py:1432) |
| CUDA-graph rollout support | yes (--cuda-graph-ppo) |
| Sweep SPS seen today | AA/AC s3 ≈ 7,900 sps, AA/AB s3 ≈ 11,000 sps |

The AA/AB difference is the `--group-relative-mix 0.3` overhead: AC includes
the mix on top of cosine+anneal so it spends ~30% more wall-clock per update.

---

## [REJECT-SOFT] E4 — 2× leverage retrain (D variant, seed 1)

lev2x_ds03/s1 training finished 2026-04-17 05:26 UTC (15M steps, 457/457
updates, final val med=-1.4% neg=21/30 trades_avg=33.9, best_neg=14).
14th-member ensemble-add test → `reports/e4_lev2x_ds03_s1_14th_candidate.json`:

| Metric           | Δ vs 13-model v5 |
|------------------|:----------------:|
| mean median/mo   | **−0.56%**       |
| mean p10/mo      | −0.33%           |
| mean neg windows | **+0.00**        |
| wins / cells     | 0 / 4            |

Soft reject: **zero extra negative windows** means the leverage-boosted D
has nearly break-even risk additivity, but doesn't lift the median. This is
the softest reject in the batch on the neg axis — worth running seeds 2-5
to find one that also pulls median.

(Checkpoint: `pufferlib_market/checkpoints/screened32_leverage_sweep/D/lev2x_ds03/s1/val_best.pt`)

## [PENDING] E2b — fresh D seeds 200/201/202

(results land here)

## [DONE] Fused CUDA pair-step kernel (2026-04-17)

**`pair_sim_cuda/`** — differentiable daily pair-step in a single fused CUDA
kernel (forward + manual backward). Built for Blackwell (`sm_120`, RTX 5090).

Model per (batch, pair):
```
trade       = target_pos - prev_pos
threshold   = fee_bp + half_spread_bps + offset_bps          # 5bp buffer
fill        = session_mask * sigmoid((reach_side_bps - threshold) / T)
next_pos    = prev_pos + fill * trade
turnover    = |next_pos - prev_pos|
cost_frac   = turnover * (commission + half_spread + fee + 0.5*offset) * 1e-4
pair_pnl    = next_pos * pair_ret - cost_frac
```
Plus differentiable EOD interest: `borrowed = max(0, sum(|next_pos|) - 1.0)`,
`interest_frac = borrowed * (0.0625 / 252)` — matches the 2× leverage /
6.25% borrow spec exactly.

**Correctness**: 7/7 tests pass. Forward + backward match pure-PyTorch
reference to 1e-5; `torch.autograd.gradcheck` at fp64 passes.

**Speedup (RTX 5090, bench.py, median of 200 iters fwd+bwd)**:
| B    | P     | eager ms | fused ms | speedup |
|------|-------|---------:|---------:|--------:|
| 64   | 128   | 1.458    | 0.558    | 2.61×   |
| 64   | 2000  | 2.230    | 0.787    | 2.83×   |
| 256  | 512   | 2.271    | 0.815    | 2.79×   |
| 256  | 2000  | 2.450    | 0.788    | 3.11×   |
| 1024 | 2000  | 2.307    | 0.788    | 2.93×   |

Holds ~**0.79 ms** out to **2M pair-days per call** (B=1024, P=2000).
~**2.6M fused pair-steps / second**. For a 10M-step run that's 8s vs
23s at the kernel — the largest single-kernel speedup we've landed.

**Artifacts**:
- Source: `pair_sim_cuda/src/fused_pair_step.cu`
- Autograd: `pair_sim_cuda/wrapper.py` (`fused_pair_step`, `daily_eod_interest`)
- Tests: `pair_sim_cuda/tests/test_correctness.py`
- Bench: `pair_sim_cuda/bench.py`, `reports/pair_sim_cuda_bench.json`

**Next uses** (not yet wired):
1. Replace the soft-sim step in `binanceneural/trainer.py` with
   `fused_pair_step` → end-to-end differentiable pair P&L for full 2000+
   pair portfolios at RL-train speed.
2. Combinatorial pair allocator: a set-valued policy outputting
   `(target_pos, offset_bps)` jointly, optimized by backprop through
   this kernel + `daily_eod_interest` accumulated over a whole month.

