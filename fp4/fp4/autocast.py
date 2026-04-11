"""Selective higher-precision context manager.

Layers whose name matches any registered keyword stay in BF16/FP32 instead of
being converted to NVFP4. Mirrors the NeMo NVFP4 recipe (embed/lm_head/ln/etc).
"""
from __future__ import annotations

from contextlib import contextmanager

_KEEP_KEYWORDS: list[str] = [
    "embed", "embedding",
    "lm_head", "policy_head", "value_head",
    "norm", "layernorm", "rmsnorm", "ln_",
    "first_block", "last_block",
]


def is_kept_precision(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _KEEP_KEYWORDS)


@contextmanager
def keep_precision(*extra_keywords: str):
    global _KEEP_KEYWORDS
    added = [k.lower() for k in extra_keywords]
    _KEEP_KEYWORDS.extend(added)
    try:
        yield
    finally:
        for k in added:
            try:
                _KEEP_KEYWORDS.remove(k)
            except ValueError:
                pass
