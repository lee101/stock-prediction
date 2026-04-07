"""fp4 — minimal NVFP4 reference training library."""
from .dtypes import NVFP4_VALUES, encode_nvfp4, decode_nvfp4, is_blackwell
from .quant import quantize_nvfp4_2d, dequantize_nvfp4_2d, quantize_nvfp4_block
from .hadamard import hadamard_matrix, RandomHadamard
from .linear import NVFP4Linear, nvfp4_linear
from .optim import AdamWMaster
from .autocast import keep_precision, is_kept_precision
from .layers import NVFP4MLP

__all__ = [
    "NVFP4_VALUES",
    "encode_nvfp4",
    "decode_nvfp4",
    "is_blackwell",
    "quantize_nvfp4_2d",
    "dequantize_nvfp4_2d",
    "quantize_nvfp4_block",
    "hadamard_matrix",
    "RandomHadamard",
    "NVFP4Linear",
    "nvfp4_linear",
    "AdamWMaster",
    "keep_precision",
    "is_kept_precision",
    "NVFP4MLP",
]
