# uv Workspace Playbook

This repository uses [`uv`](https://docs.astral.sh/uv/latest/) for dependency resolution across multiple packages. Use the commands below to profile slow operations and keep installs fast on Linux workstations.

## Diagnose Slow Syncs

```bash
# High-detail trace for the next sync/lock
RUST_LOG=uv=debug uv -v sync

# When rerunning scripts without dependency changes
source .venv/bin/activate
python -c "print('hello uv')"
```

Key things to watch in the debug logs:

- **Resolver stalls** – large solve graphs or many optional extras. Consider pinning `requires-python` and keeping dependency groups small.
- **Wheel builds** – look for repeated `Building wheel for ...` lines. Prefer binary wheels and add `[tool.uv.sources]` routing (see below).
- **Install/link time** – if uv falls back to copy mode, cache and virtualenv are likely on different filesystems.

## Fast Workflows

- Keep `.venv` and the uv cache on the same filesystem so uv can hardlink instead of copying:
  ```bash
  uv cache dir
  export UV_CACHE_DIR="$HOME/.cache/uv"  # adjust if cache lives elsewhere
  ```
- Run `uv lock` only after dependency changes. For day-to-day scripting, reactivate the existing `.venv` (`source .venv/bin/activate`) and call `python`/`pytest` directly to avoid extra sync checks.
- Install just the packages you’re touching:
  ```bash
  uv sync --package hftraining --no-group dev
  source .venv/bin/activate
  python -m hftraining.train_hf
  ```
- In CI/CD, keep caches lean:
  ```bash
  uv cache prune --ci
  ```

## Workspace Layout

The root `pyproject.toml` lists workspace members so each experiment lives in its own package. Partial installs stay quick because each package declares only the dependencies it truly needs.

```
differentiable_market/
gymrl/
hfshared/
hfinference/
hftraining/
marketsimulator/
pufferlibinference/
pufferlibtraining/
toto/
traininglib/
```

Run targeted installs with `uv sync --package <name>` or install multiple components at once:

```bash
uv sync --package hftraining --package marketsimulator
```

## Torch Wheels

GPU experiments are routed directly to the CUDA 12.8 wheels. You can override the backend on the command line if you need CPU-only wheels:

```bash
uv sync --package hftraining --pip-arg "--config-settings=torch-backend=cpu"
```

## When Things Are Still Slow

- **Resolver**: tighten version ranges, set `[tool.uv].environments = ["sys_platform == 'linux'"]`, and split dev/test tooling into optional groups.
- **Downloads**: mirror PyPI locally or ensure your network isn’t bottlenecking. Route special ecosystems (e.g., PyTorch) to the correct index so uv doesn’t probe multiple registries.
- **Builds**: prefer binary wheels. When a package must build from source, add `extra-build-dependencies` in `pyproject.toml` instead of disabling isolation.
- **Linking**: confirm uv is using hardlinks (`uv cache stats`). If not, move cache/venv onto the same filesystem or set `link-mode` explicitly.

Following this checklist keeps iterative installs in the seconds range while still letting full-lock operations capture the entire monorepo.
