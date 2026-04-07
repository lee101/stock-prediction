import sys
from pathlib import Path

# Ensure the package under test (python/gpu_trading_env) takes precedence over
# any sibling directory named "gpu_trading_env" the rootdir walk may pick up.
PKG_PARENT = (Path(__file__).resolve().parent.parent / "python").resolve()
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# Drop any pre-imported namespace-package shim.
mod = sys.modules.get("gpu_trading_env")
if mod is not None and not hasattr(mod, "_load_ext"):
    del sys.modules["gpu_trading_env"]
