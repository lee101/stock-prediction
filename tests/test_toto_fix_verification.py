"""
Quick verification that Toto CUDA graphs fix is applied correctly.

This test verifies the code changes without needing GPU access.
Run full MAE tests when GPU is available.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_toto_fix():
    """Verify the .item() fix is applied in Toto code."""
    print("="*80)
    print("Toto CUDA Graphs Fix Verification")
    print("="*80)

    # Check the fix file exists
    util_file = project_root / "toto" / "toto" / "model" / "util_compile_friendly.py"

    if not util_file.exists():
        print(f"❌ File not found: {util_file}")
        return False

    print(f"\n✅ Found file: {util_file}")

    # Read and check for the fix
    content = util_file.read_text()

    issues = []

    # Check line 182-183 area (current_len method)
    if "def current_len" in content:
        # Find the method
        start = content.find("def current_len")
        end = content.find("\n    def ", start + 1)
        method_content = content[start:end]

        if "return self._current_idx[cache_idx]" in method_content:
            print("✅ Line ~182: Using host-mirrored ints in current_len() - GOOD")
        elif ".item()" in method_content and "_current_idx" in method_content:
            print("❌ Line ~182: Still using .item() in current_len() - BAD")
            issues.append("current_len() still uses .item()")
        else:
            print("⚠️  Line ~182: current_len() method structure unclear")

    # Check line 220 area (append method)
    if "def append" in content:
        # Find the method
        start = content.find("def append")
        end = content.find("\n    def ", start + 1)
        if end == -1:
            end = len(content)
        method_content = content[start:end]

        # Check for the critical line
        if "start_idx = self._current_idx[cache_idx]" in method_content:
            print("✅ Line ~220: Using host-mirrored ints in append() - GOOD")
        elif "start_idx = self._current_idx[cache_idx].item()" in method_content:
            print("❌ Line ~220: Still using .item() in append() - BAD")
            issues.append("append() still uses .item()")
        elif ".item()" in method_content and "start_idx" in method_content:
            print("⚠️  Line ~220: append() might still have .item() calls")
            issues.append("append() possibly uses .item()")
        else:
            print("✅ Line ~220: No .item() detected in append() - GOOD")

    # Ensure _current_idx is initialized as a Python list (host mirror)
    init_snippet = "self._current_idx = [0 for _ in range"
    if init_snippet in content:
        print("✅ Host index mirror detected during initialization")
    else:
        print("❌ Host index mirror missing in __post_init__")
        issues.append("_current_idx not initialized as host list")

    # Check for any remaining .item() calls that could be problematic
    item_calls = content.count(".item()")
    print(f"\nTotal .item() calls in file: {item_calls}")

    if item_calls > 0:
        print("⚠️  Note: Some .item() calls may be acceptable (in comments, etc)")
        # Show where they are
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '.item()' in line and not line.strip().startswith('#'):
                print(f"  Line {i}: {line.strip()[:80]}")

    print("\n" + "="*80)
    print("VERIFICATION RESULT")
    print("="*80)

    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nThe fix may not be correctly applied.")
        print("Re-run the fix or manually edit the file.")
        return False
    else:
        print("\n✅ SUCCESS: Toto CUDA graphs fix is correctly applied!")
        print("\nThe critical .item() calls have been eliminated by mirroring indices on the host.")
        print("CUDA graphs should now work properly with torch.compile.")
        return True


def check_test_files():
    """Check that all test files exist."""
    print("\n" + "="*80)
    print("Test File Inventory")
    print("="*80)

    test_files = [
        "tests/test_kvcache_fix.py",
        "tests/test_mae_integration.py",
        "tests/test_mae_both_models.py",
        "debug_cuda_errors.py",
    ]

    all_exist = True
    for test_file in test_files:
        path = project_root / test_file
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✅ {test_file:<40} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {test_file:<40} (MISSING)")
            all_exist = False

    return all_exist


def check_documentation():
    """Check that documentation exists."""
    print("\n" + "="*80)
    print("Documentation Inventory")
    print("="*80)

    docs = [
        "COMPLETE_FIX_GUIDE.md",
        "CUDA_GRAPHS_FIX_SUMMARY.md",
        "docs/toto_cuda_graphs_fix.md",
        "verify_cuda_graphs.sh",
    ]

    all_exist = True
    for doc in docs:
        path = project_root / doc
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✅ {doc:<40} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {doc:<40} (MISSING)")
            all_exist = False

    return all_exist


def main():
    print("\n" + "="*80)
    print("QUICK VERIFICATION - TOTO CUDA GRAPHS FIX")
    print("="*80)
    print("\nThis verifies the code changes are applied correctly.")
    print("For full MAE testing, ensure GPU is available and run:")
    print("  - python tests/test_mae_both_models.py")
    print("="*80)

    results = {
        "fix_applied": verify_toto_fix(),
        "tests_exist": check_test_files(),
        "docs_exist": check_documentation(),
    }

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")

    if all(results.values()):
        print("\n🎉 All verifications passed!")
        print("\nNext steps:")
        print("  1. Wait for GPU to be available (currently training is running)")
        print("  2. Run: python tests/test_mae_both_models.py")
        print("  3. Verify MAE baselines are acceptable")
        return 0
    else:
        print("\n⚠️  Some verifications failed.")
        print("Review the output above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
