# Binance Latest (Runbook)

Updated: 2026-02-02

## Best SOLUSD model (marketsimulator)

- Checkpoint: `binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt`
- Best PnL config: `intensity-scale=20.0`, `horizon=24`, `sequence-length=96`, `min-gap-pct=0.0003`

## Clone + setup (inference host)

```bash
git clone <REPO_URL> stock-prediction
cd stock-prediction

python3.13 -m venv .venv313
source .venv313/bin/activate

uv pip install -e .
uv pip install -e toto/
```

Sync required assets from R2 (mirror local paths):

```bash
aws s3 sync s3://models/stock/trainingdata/ trainingdata/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/models/binance/binanceneural/checkpoints/ binanceneural/checkpoints/ --endpoint-url "$R2_ENDPOINT"
```

Note: if you want to avoid rebuilding Chronos forecasts on the inference host, sync `binanceneural/forecast_cache/` as well (if present).

## Inference-only (no live orders)

```bash
source .venv313/bin/activate
PYTHONPATH=/path/to/stock-prediction \\
python -m binanceexp1.trade_binance_hourly \\
  --symbols SOLUSD \\
  --checkpoint /path/to/stock-prediction/binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt \\
  --horizon 24 \\
  --sequence-length 96 \\
  --intensity-scale 20.0 \\
  --min-gap-pct 0.0003 \\
  --once \\
  --dry-run
```

Optional: add `--cache-only` if you have a populated `binanceneural/forecast_cache/`.

## Production run (on a Binance.com-allowed host)

1) Ensure env vars are set:
- `BINANCE_API_KEY`
- `BINANCE_SECRET`
- Do **not** set `BINANCE_TLD=us` if using Binance.com.

2) Activate venv and run the live trader:
```bash
source .venv313/bin/activate
PYTHONPATH=/path/to/stock-prediction \
python -m binanceexp1.trade_binance_hourly \
  --symbols SOLUSD \
  --checkpoint /path/to/stock-prediction/binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt \
  --horizon 24 \
  --sequence-length 96 \
  --intensity-scale 20.0 \
  --min-gap-pct 0.0003 \
  --poll-seconds 30 \
  --expiry-minutes 90 \
  --log-metrics \
  --metrics-log-path /path/to/stock-prediction/strategy_state/binanceexp1-solusd-metrics.csv
```

3) Optional API smoke test (place + cancel tiny limit order):
```bash
source .venv313/bin/activate
PYTHONPATH=/path/to/stock-prediction \
python scripts/binance_api_smoketest.py --symbol SOLUSDT --side sell --notional 1 --price-multiplier 1.15
```

## Supervisor example (other machine)

```ini
[program:binanceexp1-solusd]
command=/path/to/stock-prediction/.venv313/bin/python -m binanceexp1.trade_binance_hourly --symbols SOLUSD --checkpoint /path/to/stock-prediction/binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt --horizon 24 --sequence-length 96 --intensity-scale 20.0 --min-gap-pct 0.0003 --poll-seconds 30 --expiry-minutes 90 --log-metrics --metrics-log-path /path/to/stock-prediction/strategy_state/binanceexp1-solusd-metrics.csv
directory=/path/to/stock-prediction
user=administrator
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/binanceexp1-solusd-error.log
stdout_logfile=/var/log/supervisor/binanceexp1-solusd.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
environment=HOME="/home/administrator",PYTHONPATH="/path/to/stock-prediction"
```

## S3/R2 sync commands

Note: `R2_ENDPOINT` must be set in the environment.

- Training data:
```bash
aws s3 sync trainingdata/ s3://models/stock/trainingdata/ --endpoint-url "$R2_ENDPOINT"
```

- Market simulator logs:
```bash
aws s3 sync marketsimulator/run_logs/ s3://models/stock/marketsimulator/logs/manual-20251025/ --endpoint-url "$R2_ENDPOINT"
```

- Models (mirror local path under `/models/stock/models/binance/`):
```bash
aws s3 sync binanceneural/checkpoints/ s3://models/stock/models/binance/binanceneural/checkpoints/ --endpoint-url "$R2_ENDPOINT"
```

## Notes
- Binance.com is geo-blocked from US IPs. Use a non-US host or Binance.US keys if running from the US.
- When the best model/config changes, update this file and the production command.
