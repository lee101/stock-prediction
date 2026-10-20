#!/usr/bin/env bash
# build mojo/build/train{32,16,bf} for L=${LPATCH:-16}: fp32 (validation), fp16, bf16 compute
set -euo pipefail
cd "$(dirname "$0")/.."
source mojo/.venv/bin/activate
export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda-12.9/bin/ptxas}
mkdir -p mojo/build
LPATCH=${LPATCH:-16}
for v in ${VARIANTS:-32 16 bf}; do
  case $v in 32) T=float32;; 16) T=float16;; bf) T=bfloat16;; esac
  sed "s/^comptime DT = DType.bfloat16/comptime DT = DType.$T/" mojo/src/train.mojo > mojo/src/train_$v.mojo
  mojo build -O3 -D L=$LPATCH -I mojo/src mojo/src/train_$v.mojo -o mojo/build/train${v}_L$LPATCH "$@"
done
