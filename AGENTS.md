# Agents Guide

## Machines
- **local**: development machine
- **leaf-gpu**: see `~/.secretbashrc` for connection details

## Secrets
Secrets are in `~/.secretbashrc` on each machine (gitignored). Source it before running trading bots.

## Model Checkpoints (Large Files)
Model weights (*.pt, *.pth, *.safetensors) are gitignored. Sync via R2/S3:

```bash
# Upload checkpoint to R2
rclone copy pufferlib_market/prod_ensemble/ r2:model/stock/prod_ensemble/ --progress

# Download checkpoint from R2
rclone copy r2:model/stock/prod_ensemble/ pufferlib_market/prod_ensemble/ --progress

# Sync specific model
rclone copy pufferlib_market/checkpoints/some_run/best.pt r2:model/stock/checkpoints/some_run/

# Full sync between machines
rclone sync r2:model/stock/ ./model_sync/ --progress
```

Configure rclone for R2: `rclone config` with Cloudflare R2 endpoint.

## Git Workflow
- Single branch (main), no feature branches
- After force push, on remote machine: `git fetch origin && git reset --hard origin/main`
- Large binary files go to R2, not git

## Environment Setup
```bash
source ~/.secretbashrc
source .venv/bin/activate  # or .venv312, .venv313 etc
uv pip install -e .
```

## Live Trading Rules
- Exactly one scheduled live writer process may run against a given Alpaca account.
- Automatic live exits must not realize a loss unless they are an explicit force-exit path.
- `ALLOW_ALPACA_LIVE_TRADING=1` is required before any live Alpaca writer may place orders.
