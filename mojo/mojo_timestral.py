"""Run the Mojo Timestral-209M forward from Python (file-based): forecast(Z [n, ctx] float32 log returns, sa) -> [n, 21] quantiles."""
import os, subprocess, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def forecast(Z, sa=False, half="fp16", chunk=512):
    Z = np.ascontiguousarray(Z, dtype=np.float32)
    n, ctx = Z.shape
    assert ctx % 32 == 0 and ctx // 32 in (8, 16, 24, 32), f"unsupported context {ctx}; build with LPATCH=ctx/32"
    exe = f"{ROOT}/build/tsfm{'16' if half == 'fp16' else ''}_L{ctx // 32}"
    env = dict(os.environ, MODULAR_NVPTX_COMPILER_PATH=os.environ.get("MODULAR_NVPTX_COMPILER_PATH", "/usr/local/cuda-12.9/bin/ptxas"))
    with tempfile.TemporaryDirectory() as td:
        fi, fo = f"{td}/in.f32", f"{td}/out.f32"
        Z.tofile(fi)
        subprocess.run([exe, "run", "1" if sa else "0", fi, fo, str(n), str(chunk)], check=True, env=env, cwd=os.path.dirname(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.fromfile(fo, dtype=np.float32).reshape(n, 21)
