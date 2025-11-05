"""
Optimized torch.compile configuration for Toto.

Import this at the start of your backtest to apply all optimizations.

Usage:
    import toto_compile_config
    toto_compile_config.apply()
"""

import os
import warnings


def apply(verbose=True):
    """Apply all Toto compilation optimizations."""

    optimizations = []

    # 1. Enable scalar output capture (reduces graph breaks)
    if os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") != "1":
        os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
        optimizations.append("Scalar output capture")

    # 2. Set compile mode if not already set
    if "TOTO_COMPILE_MODE" not in os.environ:
        os.environ["TOTO_COMPILE_MODE"] = "reduce-overhead"
        optimizations.append("Compile mode: reduce-overhead")

    # 3. Enable compilation if not explicitly disabled
    if "TOTO_DISABLE_COMPILE" not in os.environ and "TOTO_COMPILE" not in os.environ:
        os.environ["TOTO_COMPILE"] = "1"
        optimizations.append("Compilation enabled")

    # 4. Set inductor backend
    if "TOTO_COMPILE_BACKEND" not in os.environ:
        os.environ["TOTO_COMPILE_BACKEND"] = "inductor"
        optimizations.append("Backend: inductor")

    # 5. Configure torch inductor optimizations
    try:
        import torch._inductor.config as inductor_config

        # Enable max autotune for matmul (safe optimization)
        inductor_config.max_autotune = True

        # Use triton for better GPU performance
        inductor_config.triton.cudagraphs = True

        # Enable coordinate descent tuning
        inductor_config.coordinate_descent_tuning = True

        optimizations.append("Inductor optimizations")

    except ImportError:
        pass

    # 6. Set compilation cache directory
    cache_dir = os.path.join(os.getcwd(), "compiled_models", "torch_inductor")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    optimizations.append(f"Cache dir: {cache_dir}")

    # 7. Configure dynamo settings
    try:
        import torch._dynamo.config as dynamo_config

        # Increase recompilation limit to handle dynamic shapes better
        dynamo_config.recompile_limit = 32  # Increase from default 8

        # Enable automatic dynamic shapes
        dynamo_config.automatic_dynamic_shapes = True

        # Suppress less critical warnings
        dynamo_config.suppress_errors = False  # Keep errors visible

        optimizations.append("Dynamo configuration")

    except ImportError:
        pass

    if verbose and optimizations:
        print("Toto Compilation Optimizations Applied:")
        for opt in optimizations:
            print(f"  ✓ {opt}")

    return len(optimizations)


def get_recommended_settings():
    """Get recommended settings based on use case."""
    return {
        "maximum_performance": {
            "TOTO_COMPILE": "1",
            "TOTO_COMPILE_MODE": "max-autotune",
            "TOTO_COMPILE_BACKEND": "inductor",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
            "notes": "Best performance, may have recompilations, use for production",
        },
        "balanced": {
            "TOTO_COMPILE": "1",
            "TOTO_COMPILE_MODE": "reduce-overhead",
            "TOTO_COMPILE_BACKEND": "inductor",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
            "notes": "Good performance with stability, recommended default",
        },
        "debugging": {
            "TOTO_COMPILE": "1",
            "TOTO_COMPILE_MODE": "default",
            "TOTO_COMPILE_BACKEND": "inductor",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
            "TORCH_LOGS": "recompiles,graph_breaks",
            "notes": "Fast compilation, verbose logging, use for development",
        },
        "accuracy_first": {
            "TOTO_COMPILE": "1",
            "TOTO_COMPILE_MODE": "default",
            "TOTO_COMPILE_BACKEND": "eager",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "0",
            "notes": "Maximum accuracy, slower, use for validation",
        },
    }


if __name__ == "__main__":
    print("Toto Compile Configuration")
    print("=" * 60)
    print()
    apply(verbose=True)
    print()
    print("Recommended Settings:")
    print("=" * 60)
    for name, settings in get_recommended_settings().items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        for key, value in settings.items():
            if key == "notes":
                print(f"  Note: {value}")
            else:
                print(f"  export {key}=\"{value}\"")
