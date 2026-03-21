# Contributing

## Dev setup

```bash
git clone <repo>
cd stock-prediction
source .venv313/bin/activate
```

The primary Python environment is `.venv313` (Python 3.13). Other environments
(`.venv`, `.venv312`) exist for specific experiments but `.venv313` is the
default for all new work.

If you have a `cutedsl` sibling directory, symlink it into the repo root:

```bash
ln -s ../cutedsl cutedsl
```

## Installing packages

Always use `uv pip`, never plain `pip`:

```bash
uv pip install <package>
# or sync everything from pyproject.toml:
uv sync
uv pip install -e .
```

## Running tests

```bash
python -m pytest tests/ -x -q
```

Skip slow or integration tests during development:

```bash
python -m pytest tests/ -x -q -k "not slow" --ignore=tests/integration
```

## Branch policy

All work lands directly on `main`. No feature branches. If you need to
experiment, do it in a numbered experiment directory (e.g.
`experiments/my_idea/`) which is safe to create and commit.

## Code style

- Functions over classes for new code; avoid deep inheritance hierarchies.
- Line length 100 (enforced by Ruff).
- Run `ruff check` and `ruff format` before committing.

## SSH / RunPod security note

The remote training scripts (e.g. `src/remote_training_pipeline.py`,
`scripts/dispatch_rl_training.py`) connect to ephemeral RunPod instances with:

```
StrictHostKeyChecking=no
UserKnownHostsFile=/dev/null
```

This is **intentional**: RunPod pods are spun up fresh for every run and have
no persistent host keys. Enabling strict checking would cause every connection
to fail with a host-key-changed error. The trade-off is acceptable because:

1. Communication goes over RunPod's private network or their SSH proxy.
2. Pod IP addresses are short-lived and not reused by untrusted parties.
3. No production secrets are transmitted over these connections; only training
   data and model checkpoints stored in R2.

Do not remove these flags; they are load-bearing for the training pipeline.
