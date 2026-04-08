# Agents Guide

## Machines
- `local`: development machine
- `leaf-gpu`: see `~/.secretbashrc` for connection details

## Secrets
- Secrets live in `~/.secretbashrc` on each machine and are gitignored.
- Source that file before running trading bots.

## Model Checkpoints
- Model weights (`*.pt`, `*.pth`, `*.safetensors`) are gitignored.
- Sync them via R2/S3 instead of git.

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

- Configure `rclone` for Cloudflare R2 with `rclone config`.

## Workflow
- Work on `main`; do not create feature branches for routine changes here.
- Prefer `uv pip`; do not use bare `pip`.
- Prefer activating an existing virtualenv and then using normal `python` / `pytest` over `uv run`.
- This is a monorepo for trading experiments; it is normal to work across multiple subprojects.
- We keep multiple environments around, including `.venv`, `.venv312`, and `.venv313`. Keep them working when practical.
- Do not cut work short because installation or test setup is inconvenient. Finish the task end to end.
- Long-running training or benchmark jobs are expected; do not add arbitrary short timeouts that break real workloads.
- Write and run tests aggressively while developing.

## Git Workflow
- Large binary artifacts go to R2, not git.
- After a force-push, remote machines should realign with `git fetch origin && git reset --hard origin/main`.

## Environment Setup
```bash
source ~/.secretbashrc
source .venv313/bin/activate  # or .venv / .venv312 when appropriate
uv pip install -e .
```

## Engineering Standards
- Keep changes correct, minimal, and robust on edge cases.
- Prefer simple functions and straightforward control flow over unnecessary inheritance or abstraction.
- Fix root causes instead of masking symptoms with blanket workarounds.
- When normalizing inputs, preserve API contracts and compatibility with built-ins and major library types.
- Put reusable utilities in `src/` when that improves clarity, and test them there.
- Keep errors, warnings, and documentation technically accurate and aligned with actual behavior.
- Maintain backward and forward compatibility where possible with feature detection and conservative fallbacks.
- Refactors and bug fixes must not silently discard user data, extension points, hooks, or plugin registrations.

## Expectations
- Production trading code needs careful thought and verification.
- New experiment directories are expected and safe when they are reproducible.
- If you find unexpected changes, resolve them thoroughly instead of stopping at surface symptoms.
- Work autonomously when the right next step is clear.

## Project Context
- We optimize for strong PnL, smooth equity curves, high Sortino, and low max drawdown under realistic simulation.
- We own the full stack here, including forks, C/CUDA kernels, and simulator changes, when they materially help.
- Nearby projects such as `nanochat`, `autoresearch`, and `modded-nanogpt` can be useful references.
- The `chronos2` work is important; LoRA training and hyperparameter tuning are first-class project activities.

## Production
- See `alpacaprod.md` for what is running, marketsim scores, deploy commands, and monitoring.
- Always update `alpacaprod.md` when deploying or changing production systems.
- Validate with binary-fill marketsim at `lag>=2` before deploying any neural model.
- Soft sigmoid fills have lookahead bias; do not trust training Sortino alone.

## Live Trading Rules
- Exactly one scheduled live writer process may run against a given Alpaca account.
- Automatic live exits must not realize a loss unless they are an explicit force-exit path.
- `ALLOW_ALPACA_LIVE_TRADING=1` is required before any live Alpaca writer may place orders.

## Marketsim Realism
- `fee=10bps`, `margin=6.25%`, `fill_buffer=5bps`, `max_hold=6h`, `decision_lag>=2`
- `validation_use_binary_fills=True` (default in `config.py`)
- `fill_temperature=0.01` to limit gradient leakage
- Test at slippage `0/5/10/20 bps` before deploying
- The pufferlib C simulator with binary fills is ground truth; binanceneural soft sim is for training gradients only
