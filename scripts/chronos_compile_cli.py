#!/usr/bin/env python3
"""
CLI tool for managing Chronos2 torch.compile configuration.

Usage:
    python scripts/chronos_compile_cli.py status
    python scripts/chronos_compile_cli.py enable
    python scripts/chronos_compile_cli.py disable
    python scripts/chronos_compile_cli.py validate
    python scripts/chronos_compile_cli.py test
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chronos_compile_config import (
    ChronosCompileConfig,
    apply_production_compiled,
    apply_production_eager,
    get_current_config,
    is_compilation_enabled,
    print_current_config,
    validate_config,
)


def cmd_status():
    """Show current compilation configuration."""
    print_current_config()

    print("\nRecommendation:")
    if is_compilation_enabled():
        print("  ⚠️  Compilation is ENABLED")
        print("  Make sure you've tested thoroughly with your data")
        print("  Run: python scripts/chronos_compile_cli.py test")
    else:
        print("  ✅ Compilation is DISABLED (recommended default)")
        print("  This is the safest setting for production")


def cmd_enable():
    """Enable compilation with safe settings."""
    print("Enabling Chronos2 compilation with safe settings...")
    apply_production_compiled(verbose=True)

    print("\n✅ Compilation enabled!")
    print("\nSettings applied:")
    config = get_current_config()
    print(f"  Mode: {config.mode}")
    print(f"  Backend: {config.backend}")
    print(f"  Dtype: {config.dtype}")

    print("\n⚠️  Before using in production:")
    print("  1. Run tests: python scripts/chronos_compile_cli.py test")
    print("  2. Monitor predictions for numerical anomalies")
    print("  3. Compare results with eager mode")

    print("\nTo persist this setting, add to your environment:")
    print("  export TORCH_COMPILED=1")


def cmd_disable():
    """Disable compilation."""
    print("Disabling Chronos2 compilation...")
    apply_production_eager(verbose=True)

    print("\n✅ Compilation disabled!")
    print("Using eager mode (safest setting)")

    print("\nTo persist this setting, ensure environment has:")
    print("  export TORCH_COMPILED=0")
    print("Or simply unset TORCH_COMPILED")


def cmd_validate():
    """Validate current configuration."""
    print("Validating current Chronos2 compile configuration...")
    print()
    print_current_config()

    is_valid, warnings = validate_config()

    if is_valid:
        print("\n✅ Configuration is valid")
    else:
        print("\n⚠️  Configuration has warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\nValidation checks:")
    config = get_current_config()

    checks = [
        (
            "Compile mode",
            config.mode in {None, "default", "reduce-overhead", "max-autotune"},
            f"'{config.mode}' is a known mode" if config.enabled else "N/A (not enabled)",
        ),
        (
            "Backend",
            config.backend in {"inductor", "aot_eager", "cudagraphs"},
            f"'{config.backend}' is tested" if config.enabled else "N/A (not enabled)",
        ),
        (
            "Dtype",
            config.dtype == "float32",
            "Using float32 (safest)" if config.dtype == "float32" else f"Using {config.dtype} (may be unstable)",
        ),
        (
            "Attention impl",
            config.attn_implementation == "eager",
            "Using eager attention (safest)",
        ),
    ]

    for name, passed, message in checks:
        status = "✅" if passed else "⚠️"
        print(f"  {status} {name}: {message}")

    return is_valid


def cmd_test():
    """Run compilation tests."""
    import subprocess

    print("Running Chronos2 compilation tests...")
    print("=" * 60)

    tests = [
        ("Basic smoke tests", "pytest tests/test_chronos2_e2e_compile.py -v"),
        ("Accuracy tests", "pytest tests/test_chronos_compile_accuracy.py -v"),
        ("Fuzzing tests", "pytest tests/test_chronos2_compile_fuzzing.py -v"),
    ]

    failed = []

    for name, cmd in tests:
        print(f"\n### {name}")
        print(f"Running: {cmd}")
        print("-" * 60)

        result = subprocess.run(cmd, shell=True, capture_output=False)

        if result.returncode != 0:
            failed.append(name)
            print(f"❌ {name} FAILED")
        else:
            print(f"✅ {name} PASSED")

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if failed:
        print(f"❌ {len(failed)} test suite(s) failed:")
        for name in failed:
            print(f"  - {name}")
        print("\n⚠️  DO NOT enable compilation in production until tests pass")
        return False
    else:
        print("✅ All test suites passed!")
        print("\nYou can safely enable compilation if needed:")
        print("  python scripts/chronos_compile_cli.py enable")
        return True


def cmd_help():
    """Show help message."""
    print("Chronos2 torch.compile Configuration CLI")
    print("=" * 60)
    print()
    print("Usage: python scripts/chronos_compile_cli.py <command>")
    print()
    print("Commands:")
    print("  status    - Show current compilation configuration")
    print("  enable    - Enable compilation with safe settings")
    print("  disable   - Disable compilation (recommended default)")
    print("  validate  - Validate current configuration")
    print("  test      - Run compilation tests")
    print("  help      - Show this help message")
    print()
    print("Examples:")
    print("  python scripts/chronos_compile_cli.py status")
    print("  python scripts/chronos_compile_cli.py enable")
    print("  python scripts/chronos_compile_cli.py test")
    print()
    print("Documentation:")
    print("  docs/chronos_compilation_guide.md")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        cmd_help()
        return 0

    command = sys.argv[1].lower()

    commands = {
        "status": cmd_status,
        "enable": cmd_enable,
        "disable": cmd_disable,
        "validate": cmd_validate,
        "test": cmd_test,
        "help": cmd_help,
        "--help": cmd_help,
        "-h": cmd_help,
    }

    if command not in commands:
        print(f"Unknown command: {command}")
        print()
        cmd_help()
        return 1

    try:
        result = commands[command]()
        if result is False:
            return 1
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
