# Stock Universe Expansion Progress 2026-03-18

## Current Live Status

- Live Supervisor config remains unchanged and still runs `PAPER=0`.
- No stock candidate has cleared the 120-day promotion gate yet.
- Current rejects remain:
  - `SOFI`: forecast quality acceptable, Sortino regressed.
  - `INTC`: forecast quality acceptable, Sortino regressed.
  - `PFE`: return and drawdown improved, Sortino regressed.
  - `F`: return and drawdown improved, Sortino regressed.
  - `MU`: failed forecast cache gate.
  - `TTD`: failed forecast cache gate.

## Fixes Applied In This Iteration

- Fixed `src/chronos2_objective.py` to build correlation matrices with an outer join instead of a global inner join.
  - This prevents mixed-history symbols from collapsing the entire cohort map.
- Fixed hourly stock cohort generation in `src/alpaca_stock_expansion.py` to floor timestamps to hourly buckets and deduplicate per bucket.
  - This recovers peer relationships across stock CSVs that use different minute offsets.
- Applied the same hourly bucket normalization inside `hyperparam_chronos_hourly.py` so hourly tuning cohort scoring uses the same corrected alignment.
- Extended `scripts/run_alpaca_stock_expansion.py` to write training recommendations per candidate:
  - `recommended_next_step`
  - `recommended_reason`
  - `recommended_peer_symbols`
  - `hourly_tuning_command`
  - `lora_command`
- Extended `scripts/retrain_chronos2_hourly_loras.py` with `--save-name` so long runs can use deterministic artifact paths.

## Corrected Training Queue

- `F`
  - next step: `multivariate_lora`
  - peers: `TRIP, EXPE, DBX, META`
- `PFE`
  - next step: `multivariate_lora`
  - peers: `F, TRIP, EXPE, NYT`
- `SOFI`
  - next step: `multivariate_lora`
  - peers: `NET, PLTR, META, NVDA`
- `INTC`
  - next step: `multivariate_lora`
  - peers: `MU, NVDA, NET, META`
- `MU`
  - next step: `multivariate_lora`
  - peers: `NVDA, PLTR, NET, META`
  - note: still blocked by cache quality until retrain improves forecasts.
- `TTD`
  - next step: `hourly_tune`
  - peers: `META, SOFI, NET, EXPE`
  - note: no stored hourly Chronos2 config exists yet.

## Completed Multivariate LoRA Trial

- symbol: `F`
- runner: local `RTX 5090`
- reason local: remote host `administrator@93.127.141.100` is currently inaccessible from this session (`Permission denied (publickey,password)`).
- training session: `tmux` session `f_lora_stockexp_multivar_20260318`
- eval session: `tmux` session `f_lora_stockexp_multivar_eval_20260318`
- training log path: `analysis/local_training_logs/f_lora_stockexp_multivar_20260318.log`
- eval log path: `analysis/local_training_logs/f_lora_stockexp_multivar_eval_20260318.log`
- output artifact path: `chronos2_finetuned/F_lora_stockexp_multivar_20260318/finetuned-ckpt`
- training outcome:
  - `val_mae%=1.4267`
  - `test_mae%=2.1510`
- rebuilt forecast cache outcome:
  - `h1 MAE%=3.9817`
  - `h24 MAE%=4.7542`
- market simulator outcome vs baseline:
  - baseline: `return=-0.041069`, `sortino=-6.7113`, `max_drawdown=0.041186`
  - `F` LoRA: `return=-0.039196`, `sortino=-6.9440`, `max_drawdown=0.039854`
- decision: rejected
  - reason: return and drawdown improved slightly, but `Sortino` regressed by `-0.2328`, so the candidate did not meet promotion thresholds.

## Active Training Run

- symbol: `PFE`
- runner: local `RTX 5090`
- started at: `2026-03-18 20:43 UTC`
- session: `tmux` session `pfe_lora_stockexp_multivar_20260318`
- env:
  - `cd /nvme0n1-disk/code/stock-prediction`
  - `source .venv313/bin/activate`
- command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol PFE \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 2048 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name PFE_lora_stockexp_multivar_20260318 \
  --covariate-symbols TRIP,META,NYT \
  --covariate-cols close \
  --no-update-hparams
```

- log path: `analysis/local_training_logs/pfe_lora_stockexp_multivar_20260318.log`
- output artifact path: `chronos2_finetuned/PFE_lora_stockexp_multivar_20260318/finetuned-ckpt`
- first log lines confirm covariates loaded:
  - `TRIP`
  - `META`
  - `NYT`

## Armed Follow-On Evaluation

- session: `tmux` session `pfe_lora_stockexp_multivar_eval_20260318`
- start gate: waits for `chronos2_finetuned/PFE_lora_stockexp_multivar_20260318/finetuned-ckpt`
- log path: `analysis/local_training_logs/pfe_lora_stockexp_multivar_eval_20260318.log`
- result directory: `analysis/alpaca_stock_expansion_pfe_lora_20260318`
- evaluation steps:
  1. Rebuild only `PFE` hourly forecast caches using the new checkpoint.
  2. Re-run one-symbol expansion evaluation for `PFE` against the current live baseline.
- cache rebuild command:

```bash
python scripts/build_hourly_forecast_caches.py \
  --symbols PFE \
  --data-root trainingdatahourly/stocks \
  --forecast-cache-root unified_hourly_experiment/forecast_cache \
  --horizons 1,24 \
  --model-id chronos2_finetuned/PFE_lora_stockexp_multivar_20260318/finetuned-ckpt \
  --context-hours 2048 \
  --batch-size 32 \
  --lookback-hours 5000 \
  --force-rebuild \
  --output-json analysis/alpaca_stock_expansion_pfe_lora_20260318/forecast_cache_mae.json
```

- follow-on market-sim command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols PFE \
  --base-stock-universe live20260318 \
  --base-long-only-symbols NVDA,PLTR,GOOG,AAPL,MSFT,META,TSLA,NET,DBX,SOFI,INTC,MU,TTD,PATH,NBIS,TME \
  --baseline-source-dir analysis/alpaca_stock_expansion_pfe_20260318/baseline \
  --skip-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_pfe_lora_20260318
```

## Short Proof Run

- completed smoke run:
  - artifact: `chronos2_finetuned/F_lora_stockexp_multivar_smoke_20260318/finetuned-ckpt`
  - outcome: command path and covariate loading validated successfully before the full run.
