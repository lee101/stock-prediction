# Alpaca Hourly Trading Runbook (Live)

Updated: 2026-02-05

This runbook covers running the hourly Alpaca trading loop with the best current checkpoints
from `newnanoalpacahourlyexp/`, plus an exit-only guard for NFLX.

## Prereqs

1. **GPU required**
   - CUDA-capable GPU with a working PyTorch CUDA build.

2. **Alpaca credentials**
   - This repo uses `env_real.py` (gitignored) for Alpaca keys.
   - Ensure `ALP_KEY_ID_PROD` and `ALP_SECRET_KEY_PROD` are set there.
   - Live trading is controlled via `PAPER=0` (env var).

3. **Python env**
   ```bash
   python3.13 -m venv .venv313-3
   source .venv313-3/bin/activate

   uv pip install -e .
   uv pip install -e toto/
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

4. **GPU sanity check**
   ```bash
   python - <<'PY'
   import torch
   print("CUDA available:", torch.cuda.is_available())
   print("CUDA version:", torch.version.cuda)
   PY
   ```

## Sync assets from R2 (recommended)

```bash
export R2_ENDPOINT="<your_r2_endpoint>"

# Hourly data
aws s3 sync s3://models/stock/trainingdatahourly/crypto/ trainingdatahourly/crypto/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/trainingdatahourly/stocks/ trainingdatahourly/stocks/ --endpoint-url "$R2_ENDPOINT"

# Alpaca checkpoints (mirror local path)
aws s3 sync s3://models/stock/models/alpaca/binanceneural/checkpoints/ binanceneural/checkpoints/ --endpoint-url "$R2_ENDPOINT"

# Chronos forecast cache (optional but recommended for speed)
aws s3 sync s3://models/stock/models/alpaca/binanceneural/forecast_cache/ binanceneural/forecast_cache/ --endpoint-url "$R2_ENDPOINT"
```

Minimal sync for SOL/LINK/UNI (optional):

```bash
aws s3 sync s3://models/stock/models/alpaca/binanceneural/checkpoints/ binanceneural/checkpoints/ \
  --exclude "*" \
  --include "alpaca_solusd_20260204_013044/*" \
  --include "alpaca_linkusd_20260204_111049/*" \
  --include "alpaca_uniusd_20260204_231926/*" \
  --endpoint-url "$R2_ENDPOINT"
```

## API smoke test

```bash
source .venv313-3/bin/activate
python - <<'PY'
import alpaca_wrapper
print(alpaca_wrapper.get_account_status())
print(alpaca_wrapper.get_clock())
PY
```

## Manual run (live)

```bash
export PAPER=0
source .venv313-3/bin/activate

python -m newnanoalpacahourlyexp.trade_alpaca_hourly \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --checkpoints "SOLUSD=binanceneural/checkpoints/alpaca_solusd_20260204_013044/epoch_006.pt,LINKUSD=binanceneural/checkpoints/alpaca_linkusd_20260204_111049/epoch_006.pt,UNIUSD=binanceneural/checkpoints/alpaca_uniusd_20260204_231926/epoch_004.pt" \
  --default-checkpoint binanceneural/checkpoints/alpaca_solusd_20260204_013044/epoch_006.pt \
  --sequence-length 96 \
  --horizon 1 \
  --forecast-horizons 1,24 \
  --forecast-horizons-map "UNIUSD=1" \
  --forecast-cache-root binanceneural/forecast_cache \
  --crypto-data-root trainingdatahourly/crypto \
  --stock-data-root trainingdatahourly/stocks \
  --allocation-pct 0.05 \
  --intensity-scale 1.0
```

Dry run (no orders):

```bash
python -m newnanoalpacahourlyexp.trade_alpaca_hourly \
  --symbols SOLUSD,LINKUSD,UNIUSD \
  --checkpoints "SOLUSD=binanceneural/checkpoints/alpaca_solusd_20260204_013044/epoch_006.pt,LINKUSD=binanceneural/checkpoints/alpaca_linkusd_20260204_111049/epoch_006.pt,UNIUSD=binanceneural/checkpoints/alpaca_uniusd_20260204_231926/epoch_004.pt" \
  --default-checkpoint binanceneural/checkpoints/alpaca_solusd_20260204_013044/epoch_006.pt \
  --sequence-length 96 \
  --horizon 1 \
  --forecast-horizons 1,24 \
  --forecast-horizons-map "UNIUSD=1" \
  --forecast-cache-root binanceneural/forecast_cache \
  --crypto-data-root trainingdatahourly/crypto \
  --stock-data-root trainingdatahourly/stocks \
  --allocation-pct 0.05 \
  --intensity-scale 1.0 \
  --dry-run \
  --once
```

## NFLX exit-only guard (live)

```bash
export PAPER=0
source .venv313-3/bin/activate

python -m newnanoalpacahourlyexp.exit_only_alpaca \
  --symbols NFLX \
  --poll-seconds 300
```

## Systemd (live)

```bash
sudo bash scripts/setup_systemd_alpaca_hourly.sh
sudo bash scripts/setup_systemd_alpaca_exit_nflx.sh
```

Logs:

```bash
sudo journalctl -u alpaca-hourly-trader -f
sudo journalctl -u alpaca-exit-nflx -f
```

## Notes

- `PAPER=0` is required for live trading; omit it or set `PAPER=1` for paper.
- The hourly loop refreshes local hourly CSVs and computes Chronos forecasts on GPU as needed.
- If you want cache-only forecasts, add `--cache-only` (ensure `binanceneural/forecast_cache/` is populated).
