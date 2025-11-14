"""Chronos2 torch.compile configuration helper."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SAFEST_MODE = "reduce-overhead"
SAFEST_BACKEND = "inductor"
SAFEST_DTYPE = "float32"
SAFEST_ATTN_IMPL = "eager"


@dataclass
class ChronosCompileConfig:
    """Configuration for Chronos2 torch.compile settings."""

    enabled: bool = False
    mode: Optional[str] = SAFEST_MODE
    backend: str = SAFEST_BACKEND
    dtype: str = SAFEST_DTYPE
    attn_implementation: str = SAFEST_ATTN_IMPL
    disable_flash_sdp: bool = True
    disable_mem_efficient_sdp: bool = True
    enable_math_sdp: bool = True
    cache_dir: Optional[str] = None

    def apply(self, verbose: bool = True) -> None:
        """Apply this configuration to environment variables."""
        os.environ["TORCH_COMPILED"] = "1" if self.enabled else "0"
        os.environ["CHRONOS_COMPILE"] = "1" if self.enabled else "0"

        if self.enabled:
            if self.mode:
                os.environ["CHRONOS_COMPILE_MODE"] = self.mode
            if self.backend:
                os.environ["CHRONOS_COMPILE_BACKEND"] = self.backend

        os.environ["CHRONOS_DTYPE"] = self.dtype

        if self.enabled and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = self.cache_dir

        if verbose:
            status = "enabled" if self.enabled else "disabled"
            if self.enabled:
                logger.info(
                    f"Chronos compile config: {status} "
                    f"(mode={self.mode}, backend={self.backend}, dtype={self.dtype})"
                )
            else:
                logger.info(f"Chronos compile config: {status}")

    def configure_torch_backends(self) -> None:
        """Configure PyTorch SDPA backends for stability."""
        try:
            import torch

            if self.disable_flash_sdp:
                torch.backends.cuda.enable_flash_sdp(False)
            if self.disable_mem_efficient_sdp:
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if self.enable_math_sdp:
                torch.backends.cuda.enable_math_sdp(True)

            logger.debug(
                "Configured PyTorch SDPA backends: "
                f"flash={not self.disable_flash_sdp}, "
                f"mem_efficient={not self.disable_mem_efficient_sdp}, "
                f"math={self.enable_math_sdp}"
            )
        except Exception as e:
            logger.debug(f"Could not configure SDPA backends: {e}")

    @classmethod
    def from_env(cls) -> ChronosCompileConfig:
        """Load configuration from environment variables."""
        enabled_str = os.getenv("TORCH_COMPILED", os.getenv("CHRONOS_COMPILE", "0"))
        enabled = enabled_str.lower() in {"1", "true", "yes", "on"}

        mode = os.getenv("CHRONOS_COMPILE_MODE", SAFEST_MODE)
        backend = os.getenv("CHRONOS_COMPILE_BACKEND", SAFEST_BACKEND)
        dtype = os.getenv("CHRONOS_DTYPE", SAFEST_DTYPE)
        cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")

        return cls(
            enabled=enabled,
            mode=mode,
            backend=backend,
            dtype=dtype,
            cache_dir=cache_dir,
        )

    @classmethod
    def safest(cls) -> ChronosCompileConfig:
        """Get the safest configuration (disabled by default)."""
        return cls(
            enabled=False,
            mode=SAFEST_MODE,
            backend=SAFEST_BACKEND,
            dtype=SAFEST_DTYPE,
            attn_implementation=SAFEST_ATTN_IMPL,
        )

    @classmethod
    def production_eager(cls) -> ChronosCompileConfig:
        """Production configuration without compilation (safest, slower)."""
        return cls(
            enabled=False,
            dtype=SAFEST_DTYPE,
            attn_implementation=SAFEST_ATTN_IMPL,
        )

    @classmethod
    def production_compiled(cls) -> ChronosCompileConfig:
        """Production configuration with compilation (faster, tested as safe)."""
        cache_dir = os.path.join(os.getcwd(), "compiled_models", "chronos2_torch_inductor")
        return cls(
            enabled=True,
            mode=SAFEST_MODE,
            backend=SAFEST_BACKEND,
            dtype=SAFEST_DTYPE,
            attn_implementation=SAFEST_ATTN_IMPL,
            cache_dir=cache_dir,
        )

    @classmethod
    def fast_testing(cls) -> ChronosCompileConfig:
        """Fast configuration for testing (no compilation overhead)."""
        return cls(
            enabled=False,
            dtype="float32",
            attn_implementation=SAFEST_ATTN_IMPL,
        )


# Convenience functions
def apply_safest_settings(verbose: bool = True) -> None:
    """Apply the safest Chronos compile settings (disabled by default)."""
    config = ChronosCompileConfig.safest()
    config.apply(verbose=verbose)
    config.configure_torch_backends()


def apply_production_eager(verbose: bool = True) -> None:
    """Apply production settings without compilation."""
    config = ChronosCompileConfig.production_eager()
    config.apply(verbose=verbose)
    config.configure_torch_backends()


def apply_production_compiled(verbose: bool = True) -> None:
    """Apply production settings with safe compilation."""
    config = ChronosCompileConfig.production_compiled()
    config.apply(verbose=verbose)
    config.configure_torch_backends()


def apply_fast_testing(verbose: bool = False) -> None:
    """Apply fast testing settings."""
    config = ChronosCompileConfig.fast_testing()
    config.apply(verbose=verbose)
    config.configure_torch_backends()


def get_current_config() -> ChronosCompileConfig:
    """Get current configuration from environment."""
    return ChronosCompileConfig.from_env()


def is_compilation_enabled() -> bool:
    """Check if torch.compile is currently enabled."""
    enabled_str = os.getenv("TORCH_COMPILED", os.getenv("CHRONOS_COMPILE", "0"))
    return enabled_str.lower() in {"1", "true", "yes", "on"}


# Diagnostic helpers
def print_current_config() -> None:
    """Print current compilation configuration."""
    config = get_current_config()
    print("=" * 60)
    print("Current Chronos Compile Configuration")
    print("=" * 60)
    print(f"Compilation enabled: {config.enabled}")
    if config.enabled:
        print(f"Compile mode:        {config.mode}")
        print(f"Compile backend:     {config.backend}")
        print(f"Cache directory:     {config.cache_dir or 'default'}")
    print(f"Dtype:               {config.dtype}")
    print(f"Attention impl:      {config.attn_implementation}")
    print("=" * 60)


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate current configuration and return (is_valid, warnings).

    Returns:
        (is_valid, warnings): Tuple of validation status and list of warning messages
    """
    config = get_current_config()
    warnings = []

    # Check for known problematic configurations
    if config.enabled:
        if config.mode not in {None, "default", "reduce-overhead", "max-autotune"}:
            warnings.append(f"Unknown compile mode: {config.mode}")

        if config.backend not in {"inductor", "aot_eager", "cudagraphs"}:
            warnings.append(f"Untested backend: {config.backend}")

        if config.dtype not in {"float32", "float16", "bfloat16"}:
            warnings.append(f"Unknown dtype: {config.dtype}")

        # Warn about potentially unstable configurations
        if config.mode == "max-autotune":
            warnings.append(
                "max-autotune mode can be unstable; use reduce-overhead for production"
            )

        if config.dtype in {"float16", "bfloat16"}:
            warnings.append(
                f"{config.dtype} may cause numerical instability; float32 recommended"
            )

    is_valid = len(warnings) == 0
    return is_valid, warnings


__all__ = [
    "ChronosCompileConfig",
    "apply_safest_settings",
    "apply_production_eager",
    "apply_production_compiled",
    "apply_fast_testing",
    "get_current_config",
    "is_compilation_enabled",
    "print_current_config",
    "validate_config",
    "SAFEST_MODE",
    "SAFEST_BACKEND",
    "SAFEST_DTYPE",
    "SAFEST_ATTN_IMPL",
]
