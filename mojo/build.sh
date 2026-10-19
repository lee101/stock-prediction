#!/usr/bin/env bash
# build mojo/build/tsfm (bf16) and mojo/build/tsfm16 (fp16)
set -euo pipefail
cd "$(dirname "$0")/.."
source mojo/.venv/bin/activate
export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda-12.9/bin/ptxas}
mkdir -p mojo/build
sed 's/^comptime DT = DType.bfloat16/comptime DT = DType.float16/' mojo/src/tsfm.mojo > mojo/src/tsfm16.mojo
mojo build -O3 -I mojo/src mojo/src/tsfm.mojo -o mojo/build/tsfm "$@"
mojo build -O3 -I mojo/src mojo/src/tsfm16.mojo -o mojo/build/tsfm16 "$@"
