"""
Torch.compile configuration helpers for Chronos2.

Usage:
    import chronos_compile_config
    chronos_compile_config.apply()
"""

from __future__ import annotations

import os
import warnings


def apply(verbose: bool = True) -> int:
    """Apply Chronos2-specific compilation heuristics."""

    tweaks = []

    if os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") != "1":
        os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
        tweaks.append("Scalar output capture")

    if "CHRONOS_COMPILE" not in os.environ:
        os.environ["CHRONOS_COMPILE"] = "1"
        tweaks.append("Compilation enabled")

    os.environ.setdefault("CHRONOS_COMPILE_MODE", "reduce-overhead")
    os.environ.setdefault("CHRONOS_COMPILE_BACKEND", "inductor")

    cache_dir = os.path.join(os.getcwd(), "compiled_models", "chronos2_torch_inductor")
    os.makedirs(cache_dir, exist_ok=True)
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR") != cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        tweaks.append(f"Cache dir: {cache_dir}")

    try:
        import torch._inductor.config as inductor_config  # type: ignore

        inductor_config.max_autotune = True
        inductor_config.triton.cudagraphs = True
        tweaks.append("Inductor autotune + cudagraphs")
    except Exception as exc:  # pragma: no cover - optional
        warnings.warn(f"Unable to configure torch._inductor optimizations: {exc}")

    try:
        import torch._dynamo.config as dynamo_config  # type: ignore

        dynamo_config.recompile_limit = 64
        dynamo_config.automatic_dynamic_shapes = True
        tweaks.append("Dynamo dynamic shapes")
    except Exception:  # pragma: no cover
        pass

    if verbose and tweaks:
        print("Chronos2 Compile Optimizations:")
        for tweak in tweaks:
            print(f"  ✓ {tweak}")

    return len(tweaks)


if __name__ == "__main__":
    apply(verbose=True)
