# Binance Results (updated 2026-02-02)

## Summary
- Baseline best (full-history classic): `chronos_sol_20260202_041139` total_return=0.3433, sortino=136.8686, annualized_return=49893.1508
- Nano + Muon (90d history): `chronos_sol_nano_muon_20260202_230000` total_return=0.1934, sortino=50.0261, annualized_return=651.0996
- Nano + AdamW (90d history): `chronos_sol_nano_adamw_20260202_230500` total_return=0.2217, sortino=47.5819, annualized_return=1536.6275
- Live Binance run attempt failed from this environment due to restricted region (Binance API eligibility block).

## Experiments

### Baseline (classic policy, full history)
- Run: `chronos_sol_20260202_041139`
- Checkpoint: `binancechronossolexperiment/checkpoints/chronos_sol_20260202_041139/policy_checkpoint.pt`
- Metrics (10-day test): total_return=0.3433135070, sortino=136.8685975259, annualized_return=49893.15079393116, final_equity=13433.135070134389
- Last 2 days: total_return=0.08457621078, sortino=218.4418841462, annualized_return=2722471.909152695, final_equity=10859.423396243676

### Nano + Muon (90d history, 6 epochs)
- Run: `chronos_sol_nano_muon_20260202_230000`
- Command:
  `python -m binancechronossolexperiment.run_experiment --symbol SOLUSDT --data-root trainingdatahourlybinance --forecast-cache-root binancechronossolexperiment/forecast_cache --model-id chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt --context-hours 3072 --chronos-batch-size 32 --horizons 1 --sequence-length 72 --val-days 20 --test-days 10 --max-history-days 90 --epochs 6 --batch-size 128 --learning-rate 3e-4 --weight-decay 1e-4 --optimizer muon_mix --model-arch nano --num-kv-heads 2 --mlp-ratio 4.0 --logits-softcap 12.0 --muon-lr 0.02 --muon-momentum 0.95 --muon-ns-steps 5 --cache-only --no-compile --run-name chronos_sol_nano_muon_20260202_230000`
- Metrics (10-day test): total_return=0.1933922754, sortino=50.0260710990, annualized_return=651.0995876932, final_equity=11935.220843872594
- Last 2 days: total_return=0.045477, sortino=70.023416, annualized_return=3347.674857, final_equity=10470.791117

### Nano + AdamW (90d history, 6 epochs)
- Run: `chronos_sol_nano_adamw_20260202_230500`
- Command:
  `python -m binancechronossolexperiment.run_experiment --symbol SOLUSDT --data-root trainingdatahourlybinance --forecast-cache-root binancechronossolexperiment/forecast_cache --model-id chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt --context-hours 3072 --chronos-batch-size 32 --horizons 1 --sequence-length 72 --val-days 20 --test-days 10 --max-history-days 90 --epochs 6 --batch-size 128 --learning-rate 3e-4 --weight-decay 1e-4 --optimizer adamw --model-arch nano --num-kv-heads 2 --mlp-ratio 4.0 --logits-softcap 12.0 --cache-only --no-compile --run-name chronos_sol_nano_adamw_20260202_230500`
- Metrics (10-day test): total_return=0.2216511106, sortino=47.5818631247, annualized_return=1536.6274827896, final_equity=12217.968001
- Last 2 days: total_return=0.044726, sortino=46.942293, annualized_return=2936.213771, final_equity=10453.053959

## Live Trading Attempt (Binance)
- Command used (single-cycle):
  `python -m binanceneural.trade_binance_hourly --symbols SOLUSDT --checkpoint binancechronossolexperiment/checkpoints/chronos_sol_20260202_041139/policy_checkpoint.pt --horizon 1 --sequence-length 72 --allocation-usdt 1 --data-root trainingdatahourlybinance --forecast-cache-root binancechronossolexperiment/forecast_cache --forecast-horizons 1 --cache-only --once`
- Result: Binance API returned restricted-region eligibility error; client failed to initialize.
- Next step: run from an eligible region or configure `BINANCE_TLD`/`BINANCE_BASE_ENDPOINT` to a permitted domain (user preference was no Binance US).
