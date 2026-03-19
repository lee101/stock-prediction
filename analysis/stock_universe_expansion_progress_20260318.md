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
  - `RKLB`: failed forecast cache gate.
  - `PATH`: failed forecast cache gate.
  - `TME`: failed forecast cache gate.
  - `HIMS`: failed forecast cache gate.
  - `ONDS`: failed forecast cache gate.
  - `OWL`: failed forecast cache gate.
  - `NBIS`: failed forecast cache gate.
  - `BMNR`: failed forecast cache gate.
  - `RCAT`: failed forecast cache gate.

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

## Completed First-Pass Evaluation

- symbol: `RKLB`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:03 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols RKLB \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_rklb_20260319
```

- current status:
  - `RKLB` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_rklb_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=13.3793`
    - `h24 MAE%=14.9696`
  - decision: rejected
    - reason: `RKLB` missed both cache gates on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `PATH`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:10 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols PATH \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_path_20260319
```

- current status:
  - `PATH` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_path_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=12.9369`
    - `h24 MAE%=15.0405`
  - decision: rejected
    - reason: `PATH` missed both cache gates on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `TME`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:09 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols TME \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_tme_20260319
```

- current status:
  - `TME` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_tme_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=11.8672`
    - `h24 MAE%=14.2333`
  - decision: rejected
    - reason: `TME` missed both cache gates on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `HIMS`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:11 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols HIMS \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_hims_20260319
```

- current status:
  - `HIMS` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_hims_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=21.0355`
    - `h24 MAE%=23.9933`
  - decision: rejected
    - reason: `HIMS` missed both cache gates badly on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `ONDS`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:14 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols ONDS \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_onds_20260319
```

- current status:
  - `ONDS` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_onds_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=14.6653`
    - `h24 MAE%=16.6711`
  - decision: rejected
    - reason: `ONDS` missed both cache gates on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `OWL`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:16 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols OWL \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_owl_20260319
```

- current status:
  - `OWL` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_owl_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=10.3220`
    - `h24 MAE%=12.0786`
  - decision: rejected
    - reason: `OWL` came close on `h1` but still missed both cache gates on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `NBIS`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:18 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols NBIS \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_nbis_20260319
```

- current status:
  - `NBIS` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_nbis_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=12.0542`
    - `h24 MAE%=14.2682`
  - decision: rejected
    - reason: `NBIS` missed both cache gates on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `BMNR`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:20 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols BMNR \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_bmnr_20260319
```

- current status:
  - `BMNR` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_bmnr_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=20.2033`
    - `h24 MAE%=22.9856`
  - decision: rejected
    - reason: `BMNR` missed both cache gates badly on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed First-Pass Evaluation

- symbol: `RCAT`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:22 UTC`
- command:

```bash
python scripts/run_alpaca_stock_expansion.py \
  --manifest-path docs/stock_universe_candidates_20260318.json \
  --candidate-symbols RCAT \
  --baseline-source-dir analysis/alpaca_stock_expansion_abev_20260319/ABEV \
  --candidate-only-cache-build \
  --candidate-max-h1-mae-percent 10 \
  --candidate-max-h24-mae-percent 10 \
  --output-dir analysis/alpaca_stock_expansion_rcat_20260319
```

- current status:
  - `RCAT` failed the forecast cache gate on the base Chronos2 checkpoint.
  - cache MAE from `analysis/alpaca_stock_expansion_rcat_20260319/forecast_cache_mae.json`:
    - `h1 MAE%=19.4586`
    - `h24 MAE%=22.5221`
  - decision: rejected
    - reason: `RCAT` missed both cache gates badly on the base checkpoint, so it stays in the tune-or-retrain bucket and does not earn a simulator run against the live `ABEV` baseline.

## Completed Hourly Tune

- symbol: `OWL`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:25 UTC`
- tuned config path: `hyperparams/chronos2/hourly/OWL.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols OWL \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- tuning outcome:
  - saved `hyperparams/chronos2/hourly/OWL.json`
  - best objective config: `context_length=1024`, `skip_rates=[1]`, `aggregation=single`, `multivariate=True`
  - tuner reported `pct_return_mae=30.6795%`
- tuned follow-on cache outcome:
  - `h1 MAE%=10.3220`
  - `h24 MAE%=12.0786`
- decision: rejected
  - reason: the tuned hourly config was loaded (`ctx=1024`, `batch=32`) but did not improve the cache gate, so `OWL` stays below promotion quality and moves to the multivariate LoRA step.

## Started Multivariate LoRA Trial

- symbol: `OWL`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:28 UTC`
- log path: `analysis/local_training_logs/owl_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/OWL_lora_stockexp_multivar_20260319/finetuned-ckpt`
- command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol OWL \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name OWL_lora_stockexp_multivar_20260319 \
  --covariate-symbols META,PLTR,EXPE,TRIP \
  --covariate-cols close \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/OWL_lora_stockexp_multivar_smoke_20260319/finetuned-ckpt`
    - smoke outcome: covariate loading, model load, fine-tune, and checkpoint write all succeeded.
  - full 1500-step run is active with peers `META, PLTR, EXPE, TRIP`.

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

## Completed Multivariate LoRA Trial

- symbol: `OWL`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:28 UTC`
- log path: `analysis/local_training_logs/owl_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/OWL_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=18.9221`
  - `test_mae%=24.1881`
- rebuilt forecast cache outcome:
  - `h1 MAE%=13.4962`
  - `h24 MAE%=15.7451`
- promotion summary from `analysis/alpaca_stock_expansion_owl_lora_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: OWL`
- decision: rejected
  - reason: the multivariate LoRA path made `OWL` materially worse than both the base and tuned checkpoints on `h1` and `h24`, so it remains below the cache gate and does not earn a simulator run against the live `ABEV` baseline.

## Completed Hourly Tune

- symbol: `NOK`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:36 UTC`
- tuned config path: `hyperparams/chronos2/hourly/NOK.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols NOK \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- tuning outcome:
  - saved `hyperparams/chronos2/hourly/NOK.json`
  - best objective config: `context_length=1024`, `skip_rates=[1]`, `aggregation=single`, `multivariate=True`
  - tuner reported `pct_return_mae=8.4974%`
- tuned follow-on cache outcome:
  - `h1 MAE%=9.4432`
  - `h24 MAE%=11.1331`
- promotion summary from `analysis/alpaca_stock_expansion_nok_tuned_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: NOK`
- decision: move to retrain
  - reason: the tuned hourly config loaded correctly (`ctx=1024`, `batch=32`) but did not change the realized cache MAE at all, so `NOK` remains blocked on the `h24` gate and now moves to the single-symbol LoRA step.

## Started Single-Symbol LoRA Trial

- symbol: `NOK`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:38 UTC`
- training log path: `analysis/local_training_logs/nok_lora_stockexp_single_20260319.log`
- output artifact path: `chronos2_finetuned/NOK_lora_stockexp_single_20260319/finetuned-ckpt`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol NOK \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name NOK_lora_stockexp_single_20260319 \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/NOK_lora_stockexp_single_smoke_20260319/finetuned-ckpt`
    - smoke outcome: `val_mae%=8.1683`, `test_mae%=10.7665`
  - full 1500-step run is active on the single-symbol LoRA baseline because the evaluator recommended `single_symbol_lora` rather than a multivariate cohort for `NOK`.

## Completed Single-Symbol LoRA Trial

- symbol: `NOK`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:38 UTC`
- training log path: `analysis/local_training_logs/nok_lora_stockexp_single_20260319.log`
- output artifact path: `chronos2_finetuned/NOK_lora_stockexp_single_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=4.9328`
  - `test_mae%=7.7063`
- rebuilt forecast cache outcome:
  - `h1 MAE%=13.4480`
  - `h24 MAE%=15.4710`
- promotion summary from `analysis/alpaca_stock_expansion_nok_lora_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: NOK`
- decision: rejected
  - reason: the single-symbol LoRA checkpoint looked better during training but degraded badly in the real cache pipeline, with repeated heuristic fallbacks and both horizons landing well above the `10%` gate.

## Completed Hourly Tune

- symbol: `TME`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:45 UTC`
- tuned config path: `hyperparams/chronos2/hourly/TME.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols TME \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- tuning outcome:
  - saved `hyperparams/chronos2/hourly/TME.json`
  - best objective config: `context_length=1024`, `skip_rates=[1]`, `aggregation=single`, `multivariate=True`
  - tuner reported `pct_return_mae=14.6673%`
- tuned follow-on cache outcome:
  - `h1 MAE%=11.8672`
  - `h24 MAE%=14.2333`
- promotion summary from `analysis/alpaca_stock_expansion_tme_tuned_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: TME`
- decision: move to multivariate LoRA
  - reason: the tuned hourly config loaded correctly but did not change the realized cache gate, so `TME` now moves to the evaluator-recommended multivariate LoRA path with `ITUB, META, PLTR, NVDA`.

## Started Multivariate LoRA Trial

- symbol: `TME`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:46 UTC`
- training log path: `analysis/local_training_logs/tme_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/TME_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol TME \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name TME_lora_stockexp_multivar_20260319 \
  --covariate-symbols ITUB,META,PLTR,NVDA \
  --covariate-cols close \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/TME_lora_stockexp_multivar_smoke_20260319/finetuned-ckpt`
    - smoke outcome: `val_mae%=13.5911`, `test_mae%=11.7607`
  - full 1500-step run is active with peers `ITUB, META, PLTR, NVDA`.

## Completed Multivariate LoRA Trial

- symbol: `TME`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:46 UTC`
- training log path: `analysis/local_training_logs/tme_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/TME_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=12.3284`
  - `test_mae%=10.5245`
- rebuilt forecast cache outcome:
  - `h1 MAE%=17.3820`
  - `h24 MAE%=20.7000`
- promotion summary from `analysis/alpaca_stock_expansion_tme_lora_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: TME`
- decision: rejected
  - reason: the multivariate LoRA checkpoint degraded the real cache pipeline badly, with repeated heuristic fallbacks and both horizons landing far above the `10%` gate.

## Completed Hourly Tune

- symbol: `NBIS`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:53 UTC`
- tuned config path: `hyperparams/chronos2/hourly/NBIS.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols NBIS \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- tuning outcome:
  - saved `hyperparams/chronos2/hourly/NBIS.json`
  - best objective config: `context_length=2048`, `skip_rates=[1]`, `aggregation=single`, `multivariate=True`
  - tuner reported `pct_return_mae=15.2276%`
  - tuner found no usable correlated cohort peers (`cohort_used=0/0`)
- tuned follow-on cache outcome:
  - `h1 MAE%=12.0542`
  - `h24 MAE%=14.2682`
- promotion summary from `analysis/alpaca_stock_expansion_nbis_tuned_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: NBIS`
- decision: move to multivariate LoRA
  - reason: the tuned hourly config loaded correctly but produced exactly the same realized cache gate as first pass, so `NBIS` now moves to the evaluator-recommended multivariate LoRA path with `PLTR, NVDA, NET, GOOG`.

## Started Multivariate LoRA Trial

- symbol: `NBIS`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:55 UTC`
- training log path: `analysis/local_training_logs/nbis_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/NBIS_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol NBIS \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 2048 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name NBIS_lora_stockexp_multivar_20260319 \
  --covariate-symbols PLTR,NVDA,NET,GOOG \
  --covariate-cols close \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/NBIS_lora_stockexp_multivar_smoke_20260319/finetuned-ckpt`
    - smoke outcome: `val_mae%=4.2773`, `test_mae%=18.4523`
  - full 1500-step run is active with peers `PLTR, NVDA, NET, GOOG`.

## Completed Multivariate LoRA Trial

- symbol: `NBIS`
- runner: local `RTX 5090`
- started at: `2026-03-19 04:55 UTC`
- training log path: `analysis/local_training_logs/nbis_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/NBIS_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=2.0952`
  - `test_mae%=15.2995`
- rebuilt forecast cache outcome:
  - `h1 MAE%=9.2047`
  - `h24 MAE%=10.1838`
- strict-gate promotion summary from `analysis/alpaca_stock_expansion_nbis_lora_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: NBIS`
- soft-gate research sim from `analysis/alpaca_stock_expansion_nbis_lora_softgate_20260319`:
  - relaxed gate used only for research: `candidate_max_h24_mae_percent=10.25`
  - simulated return improved to `-0.0317950` from baseline `-0.0335545`
  - simulated max drawdown improved to `0.0320248` from baseline `0.0337971`
  - simulated Sortino worsened to `-5.9790` from baseline `-5.8506`
- decision: rejected
  - reason: the LoRA finally pushed `h1` below gate, but `h24` still missed the strict threshold and even the relaxed research sim failed the actual promotion rule because Sortino regressed against the live `ABEV` baseline.

## Completed Hourly Tune

- symbol: `RKLB`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:05 UTC`
- tuned config path: `hyperparams/chronos2/hourly/RKLB.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols RKLB \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- tuning outcome:
  - saved `hyperparams/chronos2/hourly/RKLB.json`
  - best objective config: `context_length=1024`, `skip_rates=[1]`, `aggregation=single`, `multivariate=False`
  - tuner reported `pct_return_mae=1.5071%`
- tuned follow-on cache outcome:
  - `h1 MAE%=13.3793`
  - `h24 MAE%=14.9696`
- promotion summary from `analysis/alpaca_stock_expansion_rklb_tuned_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: RKLB`
- decision: move to multivariate LoRA
  - reason: the tuned hourly config did not improve the realized cache gate, so `RKLB` moved to the evaluator-recommended multivariate LoRA path with `PLTR, NVDA, NET, META`.

## Started Multivariate LoRA Trial

- symbol: `RKLB`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:09 UTC`
- training log path: `analysis/local_training_logs/rklb_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/RKLB_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol RKLB \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name RKLB_lora_stockexp_multivar_20260319 \
  --covariate-symbols PLTR,NVDA,NET,META \
  --covariate-cols close \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/RKLB_lora_stockexp_multivar_smoke_20260319/finetuned-ckpt`
    - smoke outcome: `val_mae%=4.5845`, `test_mae%=3.5420`
  - full 1500-step run launched with peers `PLTR, NVDA, NET, META`.

## Completed Multivariate LoRA Trial

- symbol: `RKLB`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:09 UTC`
- training log path: `analysis/local_training_logs/rklb_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/RKLB_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=1.7887`
  - `test_mae%=2.1666`

## Completed Follow-On Evaluation

- symbol: `RKLB`
- evaluation directory: `analysis/alpaca_stock_expansion_rklb_lora_20260319`
- rebuilt forecast cache outcome:
  - `h1 MAE%=14.0942`
  - `h24 MAE%=15.9921`
- strict-gate promotion summary from `analysis/alpaca_stock_expansion_rklb_lora_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: RKLB`
- decision: rejected
  - reason: despite strong trainer metrics, the real cache pipeline emitted repeated heuristic fallbacks and both horizons landed well above the `10%` gate, so `RKLB` never earned a simulator comparison against the live `ABEV` baseline.

## Completed Soft-Gate Research Sim

- symbol: `NOK`
- evaluation directory: `analysis/alpaca_stock_expansion_nok_softgate_20260319`
- relaxed gate used only for research:
  - `candidate_max_h24_mae_percent=11.25`
- simulated outcome:
  - return `-0.0340083` vs baseline `-0.0335545`
  - Sortino `-5.7834` vs baseline `-5.8506`
  - max drawdown `0.0345393` vs baseline `0.0337971`
- promotion summary from `analysis/alpaca_stock_expansion_nok_softgate_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: NOK`
- decision: rejected
  - reason: the relaxed-gate research sim improved Sortino slightly, but return and max drawdown both regressed against the live `ABEV` baseline, so `NOK` is not worth more retrain work right now.

## Completed Hourly Tune

- symbol: `RIG`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:22 UTC`
- tuned config path: `hyperparams/chronos2/hourly/RIG.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols RIG \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- tuning outcome:
  - saved `hyperparams/chronos2/hourly/RIG.json`
  - best objective config: `context_length=1024`, `skip_rates=[1]`, `aggregation=single`, `multivariate=True`
  - tuner reported `pct_return_mae=6.0401%`
  - tuner found no usable correlated cohort peers (`cohort_used=0/0`)
- tuned follow-on cache outcome:
  - `h1 MAE%=13.0350`
  - `h24 MAE%=15.3028`
- promotion summary from `analysis/alpaca_stock_expansion_rig_tuned_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: RIG`
- decision: move to single-symbol LoRA
  - reason: the tuned hourly config did not improve the realized cache gate, and the evaluator recommends the single-symbol LoRA baseline because there is no usable correlated cohort for `RIG`.

## Started Single-Symbol LoRA Trial

- symbol: `RIG`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:24 UTC`
- training log path: `analysis/local_training_logs/rig_lora_stockexp_single_20260319.log`
- output artifact path: `chronos2_finetuned/RIG_lora_stockexp_single_20260319/finetuned-ckpt`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol RIG \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name RIG_lora_stockexp_single_20260319 \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/RIG_lora_stockexp_single_smoke_20260319/finetuned-ckpt`
    - smoke outcome: `val_mae%=4.3410`, `test_mae%=4.9974`
  - full 1500-step run is active on the single-symbol baseline.

## Completed Single-Symbol LoRA Trial

- symbol: `RIG`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:24 UTC`
- training log path: `analysis/local_training_logs/rig_lora_stockexp_single_20260319.log`
- output artifact path: `chronos2_finetuned/RIG_lora_stockexp_single_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=2.3538`
  - `test_mae%=3.0388`

## Completed Follow-On Evaluation

- symbol: `RIG`
- evaluation directory: `analysis/alpaca_stock_expansion_rig_lora_20260319`
- rebuilt forecast cache outcome:
  - `h1 MAE%=17.0920`
  - `h24 MAE%=19.5189`
- strict-gate promotion summary from `analysis/alpaca_stock_expansion_rig_lora_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: RIG`
- decision: rejected
  - reason: the single-symbol LoRA looked clean in trainer metrics, but the real cache pipeline emitted repeated heuristic fallbacks and both horizons worsened badly versus the already-rejected tuned/base checkpoint, so `RIG` remains below promotion quality.

## Completed Hourly Tune

- symbol: `PATH`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:31 UTC`
- tuned config path: `hyperparams/chronos2/hourly/PATH.json`
- tuning command:

```bash
python hyperparam_chronos_hourly.py \
  --symbols PATH \
  --quick \
  --holdout-hours 168 \
  --prediction-length 24 \
  --objective composite \
  --cohort-size 4 \
  --cohort-min-abs-corr 0.25 \
  --save-hyperparams
```

- tuning outcome:
  - saved `hyperparams/chronos2/hourly/PATH.json`
  - best objective config: `context_length=1024`, `skip_rates=[1]`, `aggregation=single`, `multivariate=True`
  - tuner reported `pct_return_mae=5.6334%`
  - tuner found no usable correlated cohort peers (`cohort_used=0/0`)
- tuned follow-on cache outcome:
  - `h1 MAE%=12.9369`
  - `h24 MAE%=15.0405`
- promotion summary from `analysis/alpaca_stock_expansion_path_tuned_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: PATH`
- decision: move to multivariate LoRA
  - reason: the tuned hourly config did not change the realized cache gate, so `PATH` moves to the evaluator-recommended multivariate LoRA path with `NET, MSFT, PLTR, DBX`.

## Started Multivariate LoRA Trial

- symbol: `PATH`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:32 UTC`
- training log path: `analysis/local_training_logs/path_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/PATH_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol PATH \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 1024 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name PATH_lora_stockexp_multivar_20260319 \
  --covariate-symbols NET,MSFT,PLTR,DBX \
  --covariate-cols close \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/PATH_lora_stockexp_multivar_smoke_20260319/finetuned-ckpt`
    - smoke outcome: `val_mae%=2.2262`, `test_mae%=2.4837`
  - full 1500-step run is active with peers `NET, MSFT, PLTR, DBX`.

## Completed Multivariate LoRA Trial

- symbol: `PATH`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:32 UTC`
- training log path: `analysis/local_training_logs/path_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/PATH_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=1.6146`
  - `test_mae%=1.5895`

## Completed Follow-On Evaluation

- symbol: `PATH`
- evaluation directory: `analysis/alpaca_stock_expansion_path_lora_20260319`
- rebuilt forecast cache outcome:
  - `h1 MAE%=13.0119`
  - `h24 MAE%=15.0021`
- strict-gate promotion summary from `analysis/alpaca_stock_expansion_path_lora_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: PATH`
- decision: rejected
  - reason: even after the multivariate LoRA with `NET, MSFT, PLTR, DBX`, the real cache pipeline still emitted repeated heuristic fallbacks and stayed well above both MAE gates, so `PATH` remains a cache-quality reject.

## Runner Fix

- component: `scripts/run_alpaca_stock_expansion.py`
- issue:
  - `--skip-cache-build` in a fresh output directory could silently bypass candidate MAE gates because no local `forecast_cache_mae.json` existed to load.
- fix:
  - added `--forecast-cache-mae-source` so rebased evaluations can explicitly reuse an existing `forecast_cache_mae.json`
  - `--skip-cache-build` now raises an actionable error when MAE gates are enabled but no summary is available
- validation:

```bash
pytest tests/test_run_alpaca_stock_expansion.py -q
```

  - outcome: `15 passed`

## Completed Rebased Evaluation

- symbol: `F`
- evaluation directory: `analysis/alpaca_stock_expansion_f_rebase_20260319`
- baseline: current live `ABEV` basket
- simulator outcome:
  - `return=-0.032538`
  - `sortino=-6.0765`
  - `max_drawdown=0.033253`
- baseline comparison:
  - baseline: `return=-0.033554`, `sortino=-5.8506`, `max_drawdown=0.033797`
- decision: move to multivariate LoRA
  - reason: `F` again improved return and max drawdown but still regressed Sortino by about `-0.2259`, so the remaining serious path was the evaluator-recommended multivariate LoRA with `TRIP, EXPE, DBX, META`.

## Started Multivariate LoRA Trial

- symbol: `F`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:47 UTC`
- training log path: `analysis/local_training_logs/f_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/F_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol F \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 2048 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1500 \
  --save-name F_lora_stockexp_multivar_20260319 \
  --covariate-symbols TRIP,EXPE,DBX,META \
  --covariate-cols close \
  --no-update-hparams
```

- current status:
  - startup validated cleanly in a 1-step smoke run:
    - artifact: `chronos2_finetuned/F_lora_stockexp_multivar_smoke_20260319/finetuned-ckpt`
    - smoke outcome: `val_mae%=2.1770`, `test_mae%=2.9874`

## Completed Multivariate LoRA Trial

- symbol: `F`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:47 UTC`
- training log path: `analysis/local_training_logs/f_lora_stockexp_multivar_20260319.log`
- output artifact path: `chronos2_finetuned/F_lora_stockexp_multivar_20260319/finetuned-ckpt`
- training outcome:
  - `val_mae%=1.5534`
  - `test_mae%=2.3437`

## Completed Follow-On Evaluation

- symbol: `F`
- evaluation directory: `analysis/alpaca_stock_expansion_f_multivar_20260319`
- rebuilt forecast cache outcome:
  - `h1 MAE%=4.0562`
  - `h24 MAE%=4.6900`
- final 120-day market simulator result from `analysis/alpaca_stock_expansion_f_multivar_20260319/F/metrics.json`:
  - `return=-0.032534`
  - `sortino=-6.0759`
  - `max_drawdown=0.033249`
  - `num_fills=2118`
- baseline comparison:
  - baseline: `return=-0.033554`, `sortino=-5.8506`, `max_drawdown=0.033797`
- strict-gate promotion summary from `analysis/alpaca_stock_expansion_f_multivar_20260319/promotion_summary.json`:
  - `promote=false`
  - reason: `No candidate met promotion thresholds. Rejected candidates: F`
- decision: rejected
  - reason: the multivariate LoRA preserved the good cache quality but did not change the live-strategy interaction. `F` still improved return and drawdown while regressing Sortino by about `-0.2253`, so it is not promotable against the current `ABEV` live basket.

## Frontier Audit

- finding:
  - the manifest was stale for several high-priority names. Their multivariate branches had already been run on `2026-03-18`, but the candidate statuses still said they needed retraining.
- audited artifacts:
  - `analysis/local_training_logs/intc_lora_stockexp_multivar_20260318.log`
  - `analysis/local_training_logs/intc_lora_stockexp_multivar_eval_20260318.log`
  - `analysis/local_training_logs/pfe_lora_stockexp_multivar_20260318.log`
  - `analysis/local_training_logs/pfe_lora_stockexp_multivar_eval_20260318.log`
  - `analysis/local_training_logs/sofi_lora_stockexp_multivar_20260318.log`
  - `analysis/local_training_logs/sofi_lora_stockexp_multivar_eval_20260318.log`
  - `analysis/local_training_logs/mu_lora_stockexp_multivar_20260318.log`
  - `analysis/local_training_logs/mu_lora_stockexp_multivar_eval_20260318.log`
  - `analysis/local_training_logs/ttd_lora_stockexp_multivar_20260318.log`
  - `analysis/local_training_logs/ttd_lora_stockexp_multivar_eval_20260318.log`
- closeout summary:
  - `INTC` multivariate already finished with `val_mae%=8.8221`, `test_mae%=11.4416` and still failed promotion on Sortino.
  - `PFE` multivariate already finished with `val_mae%=1.5617`, `test_mae%=3.1063` and still failed promotion on Sortino.
  - `SOFI` multivariate already failed promotion on Sortino with `sortino=-7.2631`.
  - `MU` multivariate terminated at the cache gate even after retrain.
  - `TTD` multivariate terminated at the cache gate even after retrain.
- decision:
  - the serious high-priority Chronos2 multivariate frontier for this stock batch is effectively exhausted.

## Aborted Smoke Retry

- symbol: `INTC`
- runner: local `RTX 5090`
- started at: `2026-03-19 05:59 UTC`
- artifact path: `chronos2_finetuned/INTC_lora_stockexp_multivar_smoke_20260319/finetuned-ckpt`
- smoke command:

```bash
python -u scripts/retrain_chronos2_hourly_loras.py \
  --symbol INTC \
  --data-root trainingdatahourly/stocks \
  --output-root chronos2_finetuned \
  --context-length 2048 \
  --batch-size 32 \
  --learning-rate 5e-05 \
  --num-steps 1 \
  --save-name INTC_lora_stockexp_multivar_smoke_20260319 \
  --covariate-symbols NVDA,NET,META,GOOG \
  --covariate-cols close \
  --no-update-hparams
```

- smoke outcome:
  - `val_mae%=12.4300`
  - `test_mae%=15.1430`
- decision: do not rerun full training
  - reason: the retry smoke was materially worse than the already-recorded March 18 multivariate run, so there is no justification for burning another duplicate 1500-step job on `INTC`.
