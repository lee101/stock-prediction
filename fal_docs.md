# FAL Training Playbook

This guide explains how to launch the unified FAL training pipeline in-process
via `run_and_train_fal.py`, keep the fal worker aware of every local training
package, and share the heavyweight dependencies (torch, numpy, pandas, …)
across the whole import tree.

## 1. Environment Prep

- Install Python requirements with `uv` and reuse the shared `.venv`:
  - `uv pip install -e .`
  - `uv pip install -e faltrain/ -e tototrainingfal/` (add other editable
    installs as you create new trainers).
- Activate the environment before running any scripts:
  - `source .venv/bin/activate`
- Keep long-running jobs unconstrained; do not add artificial timeouts for
  trainers or benchmarks.

## 2. Running `run_and_train_fal.py`

- The script wraps `fal run faltrain/app.py::StockTrainerApp` and triggers the
  synchronous `/api/train` endpoint once the worker is ready; it never forks an
  extra trainer process.
- Default usage launches sweeps for the HF trainer:
  ```
  source .venv/bin/activate
  python run_and_train_fal.py
  ```
- Override payload or cli knobs when needed:
  - `--fal-app`: alternate `faltrain` entry point.
  - `--payload-file` / `--payload-json`: explicit training payload.
  - `--fal-arg`: set fal CLI flags (repeatable).
  - `--keep-alive`: leave the worker running after the request finishes.
- The script prints the synchronous endpoint URL and the POST payload before
  firing the request; watch the streamed logs for trainer progress.

## 3. Keep `local_python_modules` Complete

- `StockTrainerApp.local_python_modules` lists every in-repo package that must
  be vendored into the fal worker. When you add or reorganize trainers, append
  their top-level package directories (e.g. `nanochat`, `newtrainer`) here.
- In-process wrappers live under `fal_hftraining/` and `fal_pufferlibtraining/`;
  use them instead of shelling out to `python <script>.py`. They expose
  `run_training` helpers that accept the simplified config dictionaries used by
  `run_and_train_fal.py`.
- The worker mirrors these directories into the fal runtime, so any imports
  within the trainer stay local and do not spawn new processes.
- Remember to run `uv pip install -e <package>/` so the package is also
  importable when running locally outside fal.

## 4. Dependency Injection Pipeline

- `StockTrainerApp.setup` eagerly imports torch, numpy, and pandas, then calls
  `faltrain.dependencies.bulk_register_fal_dependencies` to publish them.
- Trainers should request these modules via `faltrain.dependencies` instead of
  importing directly:
  ```python
  from faltrain.dependencies import get_fal_dependency

  torch = get_fal_dependency("torch")
  numpy = get_fal_dependency("numpy")
  pandas = get_fal_dependency("pandas")
  ```
- `get_fal_dependency` returns the injected module if available, otherwise it
  imports the package (and registers it) so the entire import chain stays
  consistent with the fal runtime.
- For multiple modules at once, use
  `torch, numpy = get_fal_dependencies("torch", "numpy")`.

## 5. Expectations for New Trainers

- Copy the trainer package into this repo, add it to
  `local_python_modules`, and register any additional heavy dependencies by
  calling `register_fal_dependency` during setup.
- Write regression tests under `tests/` and run them in the shared environment:
  `source .venv/bin/activate && pytest tests/<pattern>`.
- Benchmarks or sweeps should be executed through the fal worker so that GPU
  resource configuration and artifact sync logic remain consistent.

## 6. Market Simulator Workflows

- Use `run_and_train_fal_marketsimulator.py` to exercise `trade_stock_e2e` via
  the simulator in-process:
  ```
  source .venv/bin/activate
  python run_and_train_fal_marketsimulator.py --symbols AAPL BTCUSD --steps 10 --step-size 6 --initial-cash 10000
  ```
- The CLI forwards the request to `falmarket/app.py::MarketSimulatorApp`,
  which reuses injected torch/numpy/pandas modules via
  `fal_marketsimulator.runner.simulate_trading`.  The helper runs inside
  `torch.inference_mode()` and supports chunked symbol analysis via
  `MARKETSIM_SIM_ANALYSIS_CHUNK` (default processes the full symbol list per
  step).
- The simulator always runs the full analytics stack (Kronos + Toto) and keeps
  the compiled FP32 caches under `compiled_models/` in sync with
  `s3://$R2_BUCKET/compiled_models/`; hyperparameters under `hyperparams/` are
  mirrored with `s3://$R2_BUCKET/stock/hyperparams/` as part of app setup.
- When adding new simulator tooling, list the package in
  `MarketSimulatorApp.local_python_modules` and access heavy dependencies
  through `faltrain.dependencies`.

## 7. Troubleshooting Checklist

- Missing module during fal runs → confirm it is listed in
  `StockTrainerApp.local_python_modules` and installed via `uv pip install -e`.
- Import errors for torch/numpy/pandas inside trainers → replace direct imports
  with `get_fal_dependency`.
- Divergent dependency versions → ensure `StockTrainerApp.requirements`
  contains the canonical versions and avoid pinning conflicting versions inside
  individual trainers.
