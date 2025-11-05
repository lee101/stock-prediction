#!/usr/bin/env python3
"""
Fix torch.compile issues with Toto model.

This script applies compilation fixes to the Toto model to resolve:
1. "skipping cudagraphs due to mutated inputs" warnings
2. Recompilation limit being hit (8+ recompilations)
3. Symbolic shapes warnings

The fix works by patching the KVCache implementation to use proper
graph breaks at mutation points, preventing cudagraphs from being
skipped and reducing unnecessary recompilations.

Usage:
    # Apply the fix and verify
    python fix_toto_compile.py

    # Apply fix and run full accuracy test
    python fix_toto_compile.py --test

    # Just show what would be fixed
    python fix_toto_compile.py --dry-run

    # Apply fix with backup
    python fix_toto_compile.py --backup
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_toto_install():
    """Find the Toto installation directory."""
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "toto" / "toto" / "model",
        repo_root / "toto" / "build" / "lib" / "toto" / "model",
    ]

    for candidate in candidates:
        if (candidate / "util.py").exists():
            return candidate

    return None


def backup_file(file_path: Path):
    """Create a backup of a file."""
    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    return backup_path


def apply_fix(model_dir: Path, dry_run: bool = False, create_backup: bool = True):
    """
    Apply the compilation fix to Toto.

    Strategy:
    1. Patch util.py to import the fixed KVCache
    2. Optionally patch util_optimized.py
    3. Verify the patch worked
    """
    util_py = model_dir / "util.py"
    util_optimized_py = model_dir / "util_optimized.py"
    util_compile_friendly_py = model_dir / "util_compile_friendly.py"

    if not util_py.exists():
        logger.error(f"Could not find util.py at {util_py}")
        return False

    # Check if already patched
    content = util_py.read_text()
    if "util_compile_friendly" in content or "KVCacheCompileFriendly" in content:
        logger.info("Fix already applied!")
        return True

    logger.info(f"Found Toto model directory: {model_dir}")

    if dry_run:
        logger.info("[DRY RUN] Would apply the following changes:")
        logger.info(f"  1. Add util_compile_friendly.py import to {util_py}")
        logger.info(f"  2. Replace KVCache with KVCacheCompileFriendly")
        if util_optimized_py.exists():
            logger.info(f"  3. Patch {util_optimized_py}")
        return True

    # Create backup if requested
    if create_backup:
        backup_file(util_py)
        if util_optimized_py.exists():
            backup_file(util_optimized_py)

    # Copy the fixed implementation (only if it doesn't exist)
    if not util_compile_friendly_py.exists():
        source_file = Path(__file__).parent / "toto" / "toto" / "model" / "util_compile_friendly.py"
        if not source_file.exists():
            logger.error(f"Could not find source file: {source_file}")
            return False

        shutil.copy2(source_file, util_compile_friendly_py)
        logger.info(f"Copied compile-friendly implementation to {util_compile_friendly_py}")
    else:
        logger.info(f"util_compile_friendly.py already exists at {util_compile_friendly_py}")

    # Patch util.py to import and use the fixed version
    logger.info(f"Patching {util_py}...")

    # Read current content
    lines = util_py.read_text().split("\n")

    # Find the KVCache class definition
    kv_cache_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("class KVCache"):
            kv_cache_line = i
            break

    if kv_cache_line is None:
        logger.error("Could not find KVCache class definition in util.py")
        return False

    # Insert import at the top of the file (after existing imports)
    import_insert_line = 0
    for i, line in enumerate(lines):
        if line.startswith("from") or line.startswith("import"):
            import_insert_line = i + 1

    # Add import after the last import
    import_statement = "\n# PATCH: Import compile-friendly KVCache\ntry:\n    from .util_compile_friendly import KVCacheCompileFriendly as _KVCacheCompileFriendly\n    _COMPILE_FRIENDLY_AVAILABLE = True\nexcept ImportError:\n    _COMPILE_FRIENDLY_AVAILABLE = False\n"

    lines.insert(import_insert_line, import_statement)

    # Add an alias at the end of the file to use the fixed version
    alias_statement = "\n# PATCH: Use compile-friendly KVCache when available\nif _COMPILE_FRIENDLY_AVAILABLE:\n    KVCache = _KVCacheCompileFriendly\n"

    lines.append(alias_statement)

    # Write back
    util_py.write_text("\n".join(lines))
    logger.info(f"✓ Patched {util_py}")

    # Patch util_optimized.py if it exists
    if util_optimized_py.exists():
        logger.info(f"Patching {util_optimized_py}...")
        opt_lines = util_optimized_py.read_text().split("\n")

        # Add the same import and alias
        opt_lines.insert(import_insert_line, import_statement)
        opt_lines.append(alias_statement)

        util_optimized_py.write_text("\n".join(opt_lines))
        logger.info(f"✓ Patched {util_optimized_py}")

    logger.info("✓ Fix applied successfully!")
    return True


def verify_fix(model_dir: Path):
    """Verify the fix was applied correctly."""
    util_py = model_dir / "util.py"
    util_compile_friendly_py = model_dir / "util_compile_friendly.py"

    if not util_compile_friendly_py.exists():
        logger.error("util_compile_friendly.py not found")
        return False

    content = util_py.read_text()
    if "util_compile_friendly" not in content and "KVCacheCompileFriendly" not in content:
        logger.error("Fix not applied to util.py")
        return False

    logger.info("✓ Fix verification passed!")
    return True


def run_accuracy_test():
    """Run the accuracy test suite."""
    logger.info("Running accuracy test...")
    import subprocess

    result = subprocess.run(
        [sys.executable, "test_toto_compile_accuracy.py", "BTCUSD"],
        capture_output=False,
    )

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Fix torch.compile issues with Toto model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run accuracy test after applying fix",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that the fix is applied",
    )

    args = parser.parse_args()

    # Find Toto installation
    model_dir = find_toto_install()
    if model_dir is None:
        logger.error("Could not find Toto installation")
        logger.error("Expected to find toto/toto/model/util.py")
        return 1

    # Verify only mode
    if args.verify_only:
        return 0 if verify_fix(model_dir) else 1

    # Apply fix
    success = apply_fix(
        model_dir,
        dry_run=args.dry_run,
        create_backup=not args.no_backup,
    )

    if not success:
        logger.error("Failed to apply fix")
        return 1

    # Verify fix (unless dry run)
    if not args.dry_run:
        if not verify_fix(model_dir):
            logger.error("Fix verification failed")
            return 1

    # Run test if requested
    if args.test and not args.dry_run:
        if not run_accuracy_test():
            logger.error("Accuracy test failed")
            return 1

    logger.info("\n✓ Fix applied successfully!")
    logger.info("\nNext steps:")
    logger.info("  1. Run the test harness: python test_toto_compile_accuracy.py")
    logger.info("  2. Check for compilation warnings in your logs")
    logger.info("  3. Verify MAE equivalence between compiled and uncompiled")

    return 0


if __name__ == "__main__":
    sys.exit(main())
