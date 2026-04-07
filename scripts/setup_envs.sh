#!/usr/bin/env bash
# Unified env bootstrap: single .venv (py3.13) for fp4 + market_sim_py +
# pufferlib triton + gpu_trading_env. Idempotent — safe to rerun.
#
# Phase-4 P4-7: previously we had .venv (py3.13) and .venv312 split because
# triton lacked cp313 wheels and fp4 CUTLASS wouldn't compile on py312. Both
# are resolved on current torch/triton; one env runs everything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
    echo "[setup_envs] creating .venv (py3.13)"
    uv venv .venv --python 3.13
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python --version

# Ensure CUDA + writable TMPDIR for nvcc/ptxas subprocesses (see gemm.py).
for cand in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12; do
    if [[ -x "$cand/bin/nvcc" ]]; then
        export CUDA_HOME="$cand"
        export CUDA_PATH="$cand"
        export PATH="$cand/bin:$PATH"
        break
    fi
done
BUILD_TMP="$REPO_ROOT/fp4/build_tmp"
mkdir -p "$BUILD_TMP"
export TMPDIR="$BUILD_TMP"
export TRITON_CACHE_DIR="$REPO_ROOT/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

echo "[setup_envs] installing fp4 (editable)"
uv pip install -e fp4/ >/dev/null

echo "[setup_envs] installing pufferlib_cpp_market_sim (editable, --no-build-isolation)"
uv pip install -e pufferlib_cpp_market_sim/ --no-build-isolation >/dev/null || {
    echo "[setup_envs] market_sim_py install failed; trying setup.py build_ext" >&2
    (cd pufferlib_cpp_market_sim && python setup.py build_ext --inplace)
}

echo "[setup_envs] building pufferlib_market C ext"
if [[ -f pufferlib_market/setup.py ]]; then
    (cd pufferlib_market && python setup.py build_ext --inplace) || echo "[setup_envs] pufferlib_market build failed (non-fatal)" >&2
fi

echo "[setup_envs] smoke-testing imports"
python -m pytest fp4/tests/test_env_imports.py -x -q

echo "[setup_envs] done. source .venv/bin/activate to use."
