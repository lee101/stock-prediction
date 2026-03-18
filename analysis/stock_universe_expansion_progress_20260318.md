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

## Active Training Run

- runner: local `RTX 5090`
- reason local: remote host `administrator@93.127.141.100` is currently inaccessible from this session (`Permission denied (publickey,password)`).
- started at: `2026-03-18 20:22 UTC`
- session: `tmux` session `f_lora_stockexp_multivar_20260318`
- env:
  - `cd /nvme0n1-disk/code/stock-prediction`
  - `source .venv313/bin/activate`
- command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol F \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 2048 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name F_lora_stockexp_multivar_20260318 \
  --covariate-symbols TRIP,EXPE,DBX,META \
  --covariate-cols close \
  --no-update-hparams
```

- log path: `analysis/local_training_logs/f_lora_stockexp_multivar_20260318.log`
- output artifact path: `chronos2_finetuned/F_lora_stockexp_multivar_20260318/finetuned-ckpt`
- first log lines confirm covariates loaded:
  - `TRIP`
  - `EXPE`
  - `DBX`
  - `META`

## Short Proof Run

- completed smoke run:
  - artifact: `chronos2_finetuned/F_lora_stockexp_multivar_smoke_20260318/finetuned-ckpt`
  - outcome: command path and covariate loading validated successfully before the full run.
