# Stock Universe Expansion Progress 2026-03-18

## Current Live Status

- Live Supervisor now runs `PAPER=0` with `ITUB`, `BTG`, and `ABEV` added to the stock universe.
- `ITUB` was the first new stock candidate to clear the 120-day promotion gate and be deployed live.
- `BTG` is the second promoted live addition from the current one-by-one expansion batch.
- `ABEV` is the third promoted live addition from the current one-by-one expansion batch.
- Current rejects remain:
  - `SOFI`: forecast quality acceptable, Sortino regressed.
  - `INTC`: forecast quality acceptable, Sortino regressed.
  - `PFE`: return and drawdown improved, Sortino regressed.
  - `F`: return and drawdown improved, Sortino regressed.
  - `MU`: failed forecast cache gate.
  - `TTD`: failed forecast cache gate.
  - `PLUG`: failed forecast cache gate.
  - `NOK`: failed forecast cache gate.
  - `RIG`: failed forecast cache gate.
  - `MARA`: failed forecast cache gate.

## Completed History Ingest Batch

- runner: local `RTX 5090` workstation
- env:
  - `cd /nvme0n1-disk/code/stock-prediction`
  - `source .venv313/bin/activate`
- command:

```bash
python -m alpacaconstrainedexp.refresh_hourly_data \
  --symbols AAL,BTG,RIG,ABEV,ITUB,NOK,PLUG \
  --data-root trainingdatahourly \
  --backfill-hours 5000 \
  --overlap-hours 2
```

- output hourly CSVs now exist under `trainingdatahourly/stocks/` for:
  - `AAL` (`1051` rows)
  - `BTG` (`1006` rows)
  - `RIG` (`1005` rows)
  - `ABEV` (`1000` rows)
  - `ITUB` (`995` rows)
  - `NOK` (`1142` rows)
  - `PLUG` (`1256` rows)
- note:
  - the refresher still reports most stocks as `stale` after the session because the validator uses a flat `6h` threshold against the last hourly equity bar. The files themselves were written successfully and are sufficient for expansion backtests.

## Completed First-Pass Evaluation

- symbol: `AAL`
- runner: local `RTX 5090`
- started at: `2026-03-19 01:51 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols AAL \
  --base-stock-universe live20260318 \
  --baseline-source-dir analysis/alpaca_stock_expansion_pfe_20260318/baseline \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_aal_20260319
```

- current status:
  - `AAL` passed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE so far from `analysis/alpaca_stock_expansion_aal_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=8.3361`
    - `h24 MAE%=9.4679`
  - final 120-day market simulator result from `analysis/alpaca_stock_expansion_aal_20260319/AAL/metrics.json`:
    - `return=-0.043932`
    - `sortino=-6.9282`
    - `max_drawdown=0.043932`
    - `num_fills=2144`
  - baseline comparison:
    - baseline: `return=-0.041069`, `sortino=-6.7113`, `max_drawdown=0.041186`
  - decision: rejected
    - reason: `AAL` made return, Sortino, and max drawdown all worse than the current live baseline, so it is not worth a retrain cycle yet.

## Completed First-Pass Evaluation

- symbol: `ITUB`
- runner: local `RTX 5090`
- started at: `2026-03-19 01:56 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols ITUB \
  --base-stock-universe live20260318 \
  --baseline-source-dir analysis/alpaca_stock_expansion_pfe_20260318/baseline \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_itub_20260319
```

- current status:
  - `ITUB` passed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_itub_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=4.9362`
    - `h24 MAE%=5.8424`
  - final 120-day market simulator result from `analysis/alpaca_stock_expansion_itub_20260319/ITUB/metrics.json`:
    - `return=-0.039515`
    - `sortino=-6.4767`
    - `max_drawdown=0.039762`
    - `num_fills=2116`
  - baseline comparison:
    - baseline: `return=-0.041069`, `sortino=-6.7113`, `max_drawdown=0.041186`
  - decision: promoted and deployed
    - reason: `ITUB` improved return by `+0.001554`, Sortino by `+0.2346`, and max drawdown by `-0.001424` versus the current live baseline.
  - live deploy:
    - repo config updated: `deployments/unified-stock-trader/supervisor.conf`
    - active Supervisor config synced: `/etc/supervisor/conf.d/unified-stock-trader.conf`
    - `sudo supervisorctl reread` -> `unified-stock-trader: changed`
    - `sudo supervisorctl update` restarted the program successfully
    - live status after deploy: `unified-stock-trader RUNNING`

## Completed First-Pass Evaluation

- symbol: `PLUG`
- runner: local `RTX 5090`
- started at: `2026-03-19 03:17 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols PLUG \
  --baseline-source-dir analysis/alpaca_stock_expansion_itub_20260319/ITUB \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_plug_20260319
```

- current status:
  - `PLUG` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_plug_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=16.0449`
    - `h24 MAE%=17.7900`
  - promotion summary from `analysis/alpaca_stock_expansion_plug_20260319/promotion_summary.json`:
    - `promote=false`
    - reason: `No candidate met promotion thresholds. Rejected candidates: PLUG`
  - decision: rejected
    - reason: `PLUG` failed both cache gates and terminated before the simulator stage, so the next useful step is tuning or retraining rather than a production candidate comparison.

## Completed First-Pass Evaluation

- symbol: `NOK`
- runner: local `RTX 5090`
- started at: `2026-03-19 03:37 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols NOK \
  --baseline-source-dir analysis/alpaca_stock_expansion_itub_20260319/ITUB \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_nok_20260319
```

- current status:
  - `NOK` cleared the short-horizon forecast cache gate but failed on `h24`.
  - cache MAE from `analysis/alpaca_stock_expansion_nok_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=9.4432`
    - `h24 MAE%=11.1331`
  - promotion summary from `analysis/alpaca_stock_expansion_nok_20260319/promotion_summary.json`:
    - `promote=false`
    - reason: `No candidate met promotion thresholds. Rejected candidates: NOK`
  - decision: rejected
    - reason: `NOK` was close enough to keep on the retrain list, but the `24h` cache error still exceeded the `10%` gate, so it did not earn a simulator run on the base checkpoint.

## Completed First-Pass Evaluation

- symbol: `BTG`
- runner: local `RTX 5090`
- started at: `2026-03-19 03:39 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols BTG \
  --baseline-source-dir analysis/alpaca_stock_expansion_itub_20260319/ITUB \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_btg_20260319
```

- current status:
  - `BTG` passed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_btg_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=8.0979`
    - `h24 MAE%=9.3240`
  - final 120-day market simulator result from `analysis/alpaca_stock_expansion_btg_20260319/BTG/metrics.json`:
    - `return=-0.036784`
    - `sortino=-6.2673`
    - `max_drawdown=0.036860`
    - `num_fills=2118`
  - baseline comparison:
    - baseline: `return=-0.039515`, `sortino=-6.4767`, `max_drawdown=0.039762`
  - decision: promoted and deployed
    - reason: `BTG` improved return by `+0.002731`, Sortino by `+0.2094`, and max drawdown by `-0.002902` versus the current live baseline.
  - live deploy:
    - repo config updated: `deployments/unified-stock-trader/supervisor.conf`
    - active Supervisor config synced: `/etc/supervisor/conf.d/unified-stock-trader.conf`
    - explicit live env now includes `PAPER=0`
    - live status after deploy: `unified-stock-trader RUNNING`

## Completed First-Pass Evaluation

- symbol: `RIG`
- runner: local `RTX 5090`
- started at: `2026-03-19 03:49 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols RIG \
  --baseline-source-dir analysis/alpaca_stock_expansion_btg_20260319/BTG \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_rig_20260319
```

- current status:
  - `RIG` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_rig_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=13.0350`
    - `h24 MAE%=15.3028`
  - decision: rejected
    - reason: `RIG` missed both cache gates on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `BTG` baseline.

## Completed First-Pass Evaluation

- symbol: `ABEV`
- runner: local `RTX 5090`
- started at: `2026-03-19 03:51 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols ABEV \
  --baseline-source-dir analysis/alpaca_stock_expansion_btg_20260319/BTG \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_abev_20260319
```

- current status:
  - `ABEV` passed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_abev_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=6.7103`
    - `h24 MAE%=7.8806`
  - final 120-day market simulator result from `analysis/alpaca_stock_expansion_abev_20260319/ABEV/metrics.json`:
    - `return=-0.033554`
    - `sortino=-5.8506`
    - `max_drawdown=0.033797`
    - `num_fills=2118`
  - baseline comparison:
    - baseline: `return=-0.036784`, `sortino=-6.2673`, `max_drawdown=0.036860`
  - decision: promoted and deployed
    - reason: `ABEV` improved return by `+0.003229`, Sortino by `+0.4167`, and max drawdown by `-0.003063` versus the current live baseline.
  - live deploy:
    - repo config updated: `deployments/unified-stock-trader/supervisor.conf`
    - active Supervisor config synced: `/etc/supervisor/conf.d/unified-stock-trader.conf`
    - explicit live env remains `PAPER=0`
    - live status after deploy: `unified-stock-trader RUNNING`

## Completed History Ingest Batch

- runner: local `RTX 5090` workstation
- env:
  - `cd /nvme0n1-disk/code/stock-prediction`
  - `source .venv313/bin/activate`
- command:

```bash
python -m alpacaconstrainedexp.refresh_hourly_data \
  --symbols ONDS,BMNR,NBIS,TME,MARA,OWL,HIMS,PATH,RCAT,RKLB \
  --data-root trainingdatahourly \
  --backfill-hours 5000 \
  --overlap-hours 2
```

- output hourly CSVs now exist under `trainingdatahourly/stocks/` for:
  - `ONDS` (`1085` rows)
  - `BMNR` (`1140` rows)
  - `NBIS` (`1092` rows)
  - `TME` (`989` rows)
  - `MARA` (`1181` rows)
  - `OWL` (`992` rows)
  - `HIMS` (`1089` rows)
  - `PATH` (`1019` rows)
  - `RCAT` (`1012` rows)
  - `RKLB` (`1012` rows)
- note:
  - the refresher summary line was misleading and ended with `no hourly data available`, but the per-symbol CSVs were written correctly and are usable for one-by-one expansion trials.

## Completed First-Pass Evaluation

- symbol: `MARA`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:01 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols MARA \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_mara_20260319
```

- current status:
  - `MARA` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_mara_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=19.1680`
    - `h24 MAE%=21.6344`
  - decision: rejected
    - reason: `MARA` missed both cache gates badly on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Fixes Applied In This Iteration

- Updated `scripts/run_alpaca_stock_expansion.py` so the default base stock universe is `live20260319`.
  - Future first-pass screens now compare against the actual live basket that already includes `ITUB`, while `live20260318` remains available as the pre-promotion alias for reproducibility.
- Updated `scripts/run_alpaca_stock_expansion.py` again after the `BTG` deploy so the default base stock universe is `live20260319_post_btg`.
  - `live20260319` is preserved as the `ITUB`-only baseline, while new trials now default to the current live basket with both `ITUB` and `BTG`.
- Updated `scripts/run_alpaca_stock_expansion.py` again after the `ABEV` deploy so the default base stock universe is `live20260319_post_abev`.
  - `live20260319_post_btg` remains available for reproducibility, while new trials now default to the current live basket with `ITUB`, `BTG`, and `ABEV`.
- Fixed `src/chronos2_objective.py` to build correlation matrices with an outer join instead of a global inner join.
  - This prevents mixed-history symbols from collapsing the entire cohort map.
- Fixed hourly stock cohort generation in `src/alpaca_stock_expansion.py` to floor timestamps to hourly buckets and deduplicate per bucket.
  - This recovers peer relationships across stock CSVs that use different minute offsets.
- Applied the same hourly bucket normalization inside `hyperparam_chronos_hourly.py` so hourly tuning cohort scoring uses the same corrected alignment.
  - Follow-up fix: `_build_cohort_map()` now uses `.dt.floor("h")` on timestamp series; the previous `.floor("h")` call crashed the real `TTD` tuner path before it could score any configs.
- Fixed `src/models/chronos2_wrapper.py` so true multivariate/joint Chronos inference preserves the source cadence instead of defaulting inferred frequencies to daily steps.
  - This was especially important for hourly stock experiments, where multivariate forecasts could otherwise be timestamped at the wrong horizon.
- Fixed `hyperparam_chronos_hourly.py` so `use_multivariate=True` now routes through the real OHLC multivariate wrapper instead of looping over single-target `predict_df` calls.
  - Regression coverage now includes `tests/test_chronos2_wrapper.py`, `tests/test_hyperparam_chronos_hourly_objective.py`, `tests/test_chronos_forecast_manager_modes.py`, and `tests/test_build_hourly_forecast_caches.py`.
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

## Completed Multivariate LoRA Trial

- symbol: `PFE`
- runner: local `RTX 5090`
- training session: `tmux` session `pfe_lora_stockexp_multivar_20260318`
- eval session: `tmux` session `pfe_lora_stockexp_multivar_eval_20260318`
- training log path: `analysis/local_training_logs/pfe_lora_stockexp_multivar_20260318.log`
- eval log path: `analysis/local_training_logs/pfe_lora_stockexp_multivar_eval_20260318.log`
- output artifact path: `chronos2_finetuned/PFE_lora_stockexp_multivar_20260318/finetuned-ckpt`
- training outcome:
  - `val_mae%=1.5617`
  - `test_mae%=3.1063`
- rebuilt forecast cache outcome:
  - `h1 MAE%=1.9165`
  - `h24 MAE%=2.1824`
- market simulator outcome vs baseline:
  - baseline: `return=-0.041069`, `sortino=-6.7113`, `max_drawdown=0.041186`
  - `PFE` LoRA: `return=-0.039127`, `sortino=-7.1234`, `max_drawdown=0.039280`
- decision: rejected
  - reason: return and drawdown improved again, but `Sortino` regressed by `-0.4121`, so the candidate did not meet promotion thresholds.

## Completed Hourly Tune

- symbol: `TTD`
- runner: local `RTX 5090`
- tune session: `tmux` session `ttd_hourly_tune_20260318_retry`
- eval session: `tmux` session `ttd_hourly_tune_eval_20260318`
- tune log path: `analysis/local_training_logs/ttd_hourly_tune_20260318.log`
- eval log path: `analysis/local_training_logs/ttd_hourly_tune_eval_20260318.log`
- tuned config path: `hyperparams/chronos2/hourly/TTD.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols TTD \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- first attempt failed immediately on a real pandas timestamp-floor bug in `_build_cohort_map()`.
- retry outcome:
  - saved `hyperparams/chronos2/hourly/TTD.json`
  - best objective config: `context_length=1024`, `skip_rates=[1]`, `aggregation=single`, `multivariate=False`
  - tuner reported `pct_return_mae=2.0127%`
- follow-on cache outcome:
  - `h1 MAE%=18.8375`
  - `h24 MAE%=23.7331`
- decision: rejected
  - reason: tuning did not improve the forecast cache gate; `TTD` remained an immediate reject before market-sim promotion could matter.

## Completed Multivariate LoRA Trial

- symbol: `TTD`
- runner: local `RTX 5090`
- started at: `2026-03-18 21:00 UTC`
- session: `tmux` session `ttd_lora_stockexp_multivar_20260318`
- env:
  - `cd /nvme0n1-disk/code/stock-prediction`
  - `source .venv313/bin/activate`
- command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol TTD \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name TTD_lora_stockexp_multivar_20260318 \
  --covariate-symbols META,NET,EXPE,NVDA \
  --covariate-cols close \
  --no-update-hparams
```

- log path: `analysis/local_training_logs/ttd_lora_stockexp_multivar_20260318.log`
- output artifact path: `chronos2_finetuned/TTD_lora_stockexp_multivar_20260318/finetuned-ckpt`
- first checkpoint signal:
  - at `500/1500`, `eval_loss=1.467`

- completed training outcome:
  - `val_mae%=0.2556`
  - `test_mae%=1.5668`
- rebuilt forecast cache outcome:
  - `h1 MAE%=10.8580`
  - `h24 MAE%=12.5066`
- decision: rejected
  - reason: multivariate LoRA was a large improvement over the original `TTD` cache, but it still missed the hard gate (`10%`) on both horizons, so it never reached a real market-sim comparison.

## Completed Multivariate LoRA Trial

- symbol: `INTC`
- runner: local `RTX 5090`
- started at: `2026-03-18 21:06 UTC`
- session: `tmux` session `intc_lora_stockexp_multivar_20260318`
- env:
  - `cd /nvme0n1-disk/code/stock-prediction`
  - `source .venv313/bin/activate`
- command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol INTC \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 2048 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name INTC_lora_stockexp_multivar_20260318 \
  --covariate-symbols NVDA,NET,META,GOOG \
  --covariate-cols close \
  --no-update-hparams
```

- log path: `analysis/local_training_logs/intc_lora_stockexp_multivar_20260318.log`
- output artifact path: `chronos2_finetuned/INTC_lora_stockexp_multivar_20260318/finetuned-ckpt`
- training outcome:
  - `val_mae%=8.8221`
  - `test_mae%=11.4416`
- rebuilt forecast cache outcome:
  - `h1 MAE%=0.9225`
  - `h24 MAE%=8.5875`
- market simulator outcome vs baseline:
  - baseline: `return=-0.041069`, `sortino=-6.7113`, `max_drawdown=0.041186`
  - `INTC` LoRA: `return=-0.039686`, `sortino=-7.0823`, `max_drawdown=0.039780`
- decision: rejected
  - reason: return and drawdown improved, but `Sortino` regressed by `-0.3711`, so the candidate still did not meet promotion thresholds.

## Completed Follow-On Evaluation

- session: `tmux` session `intc_lora_stockexp_multivar_eval_20260318`
- start gate: waits for `chronos2_finetuned/INTC_lora_stockexp_multivar_20260318/finetuned-ckpt`
- log path: `analysis/local_training_logs/intc_lora_stockexp_multivar_eval_20260318.log`
- result directory: `analysis/alpaca_stock_expansion_intc_lora_20260318`
- evaluation steps:
  1. Rebuild only `INTC` hourly forecast caches using the new checkpoint.
  2. Re-run one-symbol expansion evaluation for `INTC` against the current live baseline.
- follow-on market-sim command:

```bash
python scripts/build_hourly_forecast_caches.py \
  --symbols INTC \
  --data-root trainingdatahourly/stocks \
  --forecast-cache-root unified_hourly_experiment/forecast_cache \
  --horizons 1,24 \
  --model-id chronos2_finetuned/INTC_lora_stockexp_multivar_20260318/finetuned-ckpt \
  --context-hours 2048 \
  --batch-size 32 \
  --lookback-hours 5000 \
  --force-rebuild \
  --output-json analysis/alpaca_stock_expansion_intc_lora_20260318/forecast_cache_mae.json

python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols INTC \
  --base-stock-universe live20260318 \
  --base-long-only-symbols NVDA,PLTR,GOOG,AAPL,MSFT,META,TSLA,NET,DBX,SOFI,INTC,MU,TTD,PATH,NBIS,TME \
  --baseline-source-dir analysis/alpaca_stock_expansion_pfe_20260318/baseline \
  --skip-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_intc_lora_20260318
```

- written artifacts:
  - `analysis/alpaca_stock_expansion_intc_lora_20260318/forecast_cache_mae.json`
  - `analysis/alpaca_stock_expansion_intc_lora_20260318/expansion_results.json`
  - `analysis/alpaca_stock_expansion_intc_lora_20260318/promotion_summary.json`

## Completed Multivariate LoRA Trial

- symbol: `SOFI`
- runner: local `RTX 5090`
- reason local: remote host `administrator@93.127.141.100` is still inaccessible from this session (`Permission denied (publickey,password)`).
- started at: `2026-03-18 21:23 UTC`
- training session: `tmux` session `sofi_lora_stockexp_multivar_20260318`
- eval session: `tmux` session `sofi_lora_stockexp_multivar_eval_20260318`
- env:
  - `cd /nvme0n1-disk/code/stock-prediction`
  - `source .venv313/bin/activate`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol SOFI \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name SOFI_lora_stockexp_multivar_20260318 \
  --covariate-symbols NET,PLTR,META,NVDA \
  --covariate-cols close \
  --no-update-hparams
```

- evaluation command chain:

```bash
python scripts/build_hourly_forecast_caches.py \
  --symbols SOFI \
  --data-root trainingdatahourly/stocks \
  --forecast-cache-root unified_hourly_experiment/forecast_cache \
  --horizons 1,24 \
  --model-id chronos2_finetuned/SOFI_lora_stockexp_multivar_20260318/finetuned-ckpt \
  --context-hours 1024 \
  --batch-size 32 \
  --lookback-hours 5000 \
  --force-rebuild \
  --output-json analysis/alpaca_stock_expansion_sofi_lora_20260318/forecast_cache_mae.json

python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols SOFI \
  --base-stock-universe live20260318 \
  --base-long-only-symbols NVDA,PLTR,GOOG,AAPL,MSFT,META,TSLA,NET,DBX,SOFI,INTC,MU,TTD,PATH,NBIS,TME \
  --baseline-source-dir analysis/alpaca_stock_expansion_pfe_20260318/baseline \
  --skip-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_sofi_lora_20260318
```

- training log path: `analysis/local_training_logs/sofi_lora_stockexp_multivar_20260318.log`
- eval log path: `analysis/local_training_logs/sofi_lora_stockexp_multivar_eval_20260318.log`
- output artifact path: `chronos2_finetuned/SOFI_lora_stockexp_multivar_20260318/finetuned-ckpt`
- training outcome:
  - `val_mae%=3.5084`
  - `test_mae%=16.1681`
- rebuilt forecast cache outcome:
  - `h1 MAE%=5.7553`
  - `h24 MAE%=6.6636`
- market simulator outcome vs baseline:
  - baseline: `return=-0.041069`, `sortino=-6.7113`, `max_drawdown=0.041186`
  - `SOFI` LoRA: `return=-0.040488`, `sortino=-7.2631`, `max_drawdown=0.040634`
- decision: rejected
  - reason: cache gate still passed, but `Sortino` regressed by `-0.5518`, so the candidate did not meet promotion thresholds.

## Tuned-Config Evaluation Command

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols TTD \
  --base-stock-universe live20260318 \
  --base-long-only-symbols NVDA,PLTR,GOOG,AAPL,MSFT,META,TSLA,NET,DBX,SOFI,INTC,MU,TTD,PATH,NBIS,TME \
  --baseline-source-dir analysis/alpaca_stock_expansion_pfe_20260318/baseline \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_ttd_tuned_20260318
```

## Short Proof Run

- completed smoke run:
  - artifact: `chronos2_finetuned/F_lora_stockexp_multivar_smoke_20260318/finetuned-ckpt`
  - outcome: command path and covariate loading validated successfully before the full run.
