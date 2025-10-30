# uv workspace layouts

This repo now ships with two uv workspace configurations so you can keep the
Python 3.12-compatible stack lightweight while still being able to install the
RL-focused projects that depend on `pufferlib>=3` (and therefore `numpy<2`).

## Core workspace (default)

* **Config:** `pyproject.toml`
* **Members:** core trading + simulator packages only
* **Usage:** works on Python 3.12–3.14; `uv sync` no longer pulls the
  pufferlib-based projects so `numpy>=2` remains satisfied.

```bash
uv venv --python 3.12 .venv312
UV_PROJECT_ENVIRONMENT=.venv312 uv sync --python 3.12
```

## RL workspace (full stack)

* **Config:** `uv.workspace-rl.toml`
* **Members:** everything in the default workspace **plus**
  `rlinc_market`, `pufferlibtraining*`, and `pufferlibinference`.
* **Usage:** requires Python ≥3.14 and `numpy<2`. Invoke uv with the alternate
  config file and (optionally) a dedicated venv:

```bash
uv venv --python 3.14 .venv314
UV_PROJECT_ENVIRONMENT=.venv314 uv sync \
  --python 3.14 \
  --config-file uv.workspace-rl.toml \
  --index-strategy unsafe-best-match
```

Because the RL workspace resorts to `unsafe-best-match`, run it only when you
explicitly need those projects, and prefer the default workspace for day-to-day
development.
