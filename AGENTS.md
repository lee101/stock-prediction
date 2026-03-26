SECURITY: This is an open source repo. NEVER commit API keys, secrets, or credentials. All secrets go in env_real.py (gitignored) or environment variables. Supervisor .conf files with keys must stay gitignored.

use uv pip NEVER just pip

try not use uv run though just activate the python env then use normal python/pytest

this is a monorepo for trading experiments 

we have a few python envs .venv .venv312 etc we try to get them all working as ideally we would be on latest as we can able to use latest tech but sometimes we cant for some experiments

dont use timeouts as we want to train long

fully finish tasks eg if it means install uv pip packages, write the tests and run them then run the related benchmarks for real with long timeouts - dont give up

code is requiring a lot of thought here as its a production trading bot

try do as much work as you can so dont just give up on installing packages - add them to pyproject.toml uv sync and install -e toto/ too just do things and get stuff tested then simulated properly all the way done

write tests/test a lot while developing - use tools 100s of tool calls is great

Ensure every code modification strictly preserves correctness, minimality of change, and robustly handles edge/corner cases related to the problem statement. ok use simple code structures like functions not complex inheritence.

Avoid blanket or “quick fix” solutions that might hide errors or unintentionally discard critical information; always strive to diagnose and address root-causes, not merely symptoms or side-effects.

Where input normalization is necessary - for types, iterables, containers, or input shapes - do so only in a way that preserves API contracts, allows for extensibility, and maintains invariance across all supported data types, including Python built-ins and major library types. can put any re usable utils in src/ and test them

All error/warning messages, exceptions, and documentation updates must be technically accurate, actionable, match the conventions of the host codebase, and be kept fully in sync with new or changed behavior.

Backwards and forwards compatibility: Changes must account for code used in diverse environments (e.g., different Python versions, framework/ORM versions, or platforms), and leverage feature detection where possible to avoid breaking downstream or legacy code.

Refactorings and bugfixes must never silently discard, mask, or change user data, hooks, plugin registrations, or extension points; if a migration or transformation is required, ensure it is invertible/idempotent where possible

use latest tactics in terms of machine learning can see nanochat/ for some good practice

instead of reconfirming with me just do it - you are probably right and yea i can always roll back thats fine lets just do it.

Creating new experiment directories is expected and safe; keep experiments reproducible so we can rerun them and match marketsimulator PnL closely to production.

if you find unexpected changes you should be thorough with resolving them yourself and git commit and push work as you go is expected, work end to end autonomously

remote training on the 5090 server is expected when local runs are too slow or GPU-bound.
- host: `administrator@93.127.141.100`
- repo root: `/nvme0n1-disk/code/stock-prediction`
- ssh style: `ssh -o StrictHostKeyChecking=no administrator@93.127.141.100`
- after ssh: `cd /nvme0n1-disk/code/stock-prediction && source .venv313/bin/activate` (or `.venv312` when required) and export `PYTHONPATH=/nvme0n1-disk/code/stock-prediction:/nvme0n1-disk/code/stock-prediction/PufferLib:$PYTHONPATH`
- prefer long-running `nohup`/`tmux` remote jobs, then fetch artifacts back with `rsync`/`scp`; do not rely on fragile interactive shells
- use the concrete command templates in `docs/REMOTE_TRAINING_RUNBOOK.md`
- for algorithm families, the default remote starting points are:
  - daily unified RL / pufferlib_market sweeps from `pufferlib_market.autoresearch_rl` or other rl environments
  - hourly Chronos2 -> forecast-cache -> RL probes from `scripts/launch_remote_hourly_chronos_rl.py`
  - tagged Chronos2 LoRA batches from `scripts/run_crypto_lora_batch.py`
  - ETH risk PPO via `fastalgorithms/eth_risk_ppo/run_train_remote.sh`
  - Chronos2 tuning loras over trainingdata/ or binancetrainingdata/ daily or trainingdatahourly/
  - existing binance chronos/neural sweep helpers like `scripts/run_sweeps_remote.sh` when working in those experiment trees
- whenever you launch remote training, record the exact remote command, env, log path, and output artifact path in the relevant progress markdown so the run is reproducible

RunPod / gpu_pool operating rules:
- prefer a local proof batch first, then a Docker validation pass, then the real remote launch; do not spin up a paid pod until the local dry run is clean
- for JAX stock training use `scripts/launch_runpod_jax_classic.py`; run `--dry-run` first, then `--docker-validate`, then the real launch
- the safe default is non-detached execution so the launcher waits, syncs back artifacts, and terminates the pod automatically when training finishes
- only use `--detach` when you intentionally want the pod to stay up for a long run; detached runs must be paired with an explicit later sync-back and explicit pod deletion
- do not rely on `uv pip install -e .` on RunPod for this repo; use a venv plus direct runtime deps and `PYTHONPATH=$PWD:$PYTHONPATH`
- preferred RunPod bootstrap pattern:
  - `uv venv .venv311jax --python python3.11`
  - `source .venv311jax/bin/activate`
  - `uv pip install numpy setuptools wheel torch pandas pyarrow loguru exchange-calendars tensorboard wandb "jax[cuda12]==0.9.2" "flax>=0.12.6" "optax>=0.2.8"`
  - `export PYTHONPATH=$PWD:$PYTHONPATH`
- every RunPod launch should capture `nvidia-smi`, `torch.cuda.is_available()`, and `jax.devices()` into the run directory before training starts
- if a pod is idle or bootstrap failed, delete it immediately; do not leave paid pods running while debugging locally
- before relying on local Docker GPU validation, ensure NVIDIA Container Toolkit is configured and `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works on the host

## Deployment
- `scripts/deploy_crypto_model.sh <checkpoint> [symbols...]` -- updates binance-hybrid-spot launch, restarts supervisor
- Remote 5090 autoresearch: `python -m pufferlib_market.autoresearch_rl --train-data <.bin> --val-data <.bin> --time-budget 7200`

## Production Documentation
- `prod.md` is the canonical source of truth for what is currently running in production, how it is launched, and the latest timestamped results/PnL.
- Every production-facing update must refresh `prod.md` with the exact service/supervisor name, launch command, checkpoint(s), mode (live/paper), and a timestamped performance snapshot.
- Before replacing an older production snapshot in `prod.md`, archive that prior state into `old_prod/` using a dated filename like `YYYY-MM-DD[-HHMM]-<slug>.md`.
- `AlpacaProgress*.md`, `binanceprogress*.md`, and other progress notes are working logs; they do not replace `prod.md` as the current-production ledger.
- `PROD.md` is legacy/reference material; keep `prod.md` as the actively maintained current-production file.
