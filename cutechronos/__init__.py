"""cutechronos: optimised Chronos-2 inference pipeline with custom kernels.

Public API
----------
CuteChronos2Pipeline
    Drop-in replacement for the upstream Chronos2Pipeline.  Supports two
    backends: the custom ``CuteChronos2Model`` (default) and the upstream
    Chronos2Model (``use_cute=False``).

CuteChronos2Model
    Pure-PyTorch Chronos-2 model with optional Triton/CUDA kernel swapins.

CuteChronos2Config
    Configuration dataclass for CuteChronos2Model.
"""

from cutechronos.pipeline import CuteChronos2Pipeline
from cutechronos.model import CuteChronos2Model, CuteChronos2Config

__all__ = [
    "CuteChronos2Pipeline",
    "CuteChronos2Model",
    "CuteChronos2Config",
]
