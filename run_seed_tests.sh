#!/bin/bash
# Run seed tests sequentially to avoid resource exhaustion

echo "Running seed determinism tests..."
echo ""

# Test uncompiled with two seeds
echo "Testing UNCOMPILED..."
timeout 120 .venv/bin/python test_seed_determinism.py uncompiled 42
timeout 120 .venv/bin/python test_seed_determinism.py uncompiled 123

echo ""
echo "Testing COMPILED..."
# Test compiled with two seeds  
timeout 120 .venv/bin/python test_seed_determinism.py compiled 42
timeout 120 .venv/bin/python test_seed_determinism.py compiled 123

echo ""
echo "=" | head -c 80
echo ""
echo "ANALYSIS"
echo "=" | head -c 80
echo ""

# Analyze results
python3 << 'PYEOF'
import numpy as np

# Load results
unc_42 = np.load('/tmp/seed_test_uncompiled_seed42.npz')
unc_123 = np.load('/tmp/seed_test_uncompiled_seed123.npz')
comp_42 = np.load('/tmp/seed_test_compiled_seed42.npz')
comp_123 = np.load('/tmp/seed_test_compiled_seed123.npz')

print("MAE Values:")
print(f"  Uncompiled Seed 42:  {unc_42['mae']:.6f}")
print(f"  Uncompiled Seed 123: {unc_123['mae']:.6f}")
print(f"  Compiled Seed 42:    {comp_42['mae']:.6f}")
print(f"  Compiled Seed 123:   {comp_123['mae']:.6f}")
print()

# Seed-to-seed variance
seed_diff_unc = abs(unc_42['mae'] - unc_123['mae'])
seed_diff_comp = abs(comp_42['mae'] - comp_123['mae'])

print("Seed-to-Seed Variance (Sampling Randomness):")
print(f"  Uncompiled: {seed_diff_unc:.6f} ({seed_diff_unc/unc_42['mae']*100:.4f}%)")
print(f"  Compiled:   {seed_diff_comp:.6f} ({seed_diff_comp/comp_42['mae']*100:.4f}%)")
print()

# Same-seed compiled vs uncompiled
same_seed_42 = abs(comp_42['mae'] - unc_42['mae'])
same_seed_123 = abs(comp_123['mae'] - unc_123['mae'])

print("Same-Seed Compiled vs Uncompiled (Compilation Effect):")
print(f"  Seed 42:  {same_seed_42:.6f} ({same_seed_42/unc_42['mae']*100:.4f}%)")
print(f"  Seed 123: {same_seed_123:.6f} ({same_seed_123/unc_123['mae']*100:.4f}%)")
print()

# Verdict
avg_same_seed = (same_seed_42 + same_seed_123) / 2
avg_seed_variance = (seed_diff_unc + seed_diff_comp) / 2

print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

if avg_same_seed < avg_seed_variance:
    print("✓ COMPILATION IS DETERMINISTIC")
    print(f"  Same-seed difference ({avg_same_seed:.0f}) < Seed variance ({avg_seed_variance:.0f})")
    print("  Variance is from sampling, NOT compilation")
else:
    print("✗ COMPILATION MAY BE NON-DETERMINISTIC")
    print(f"  Same-seed difference ({avg_same_seed:.0f}) >= Seed variance ({avg_seed_variance:.0f})")
    
print()
print(f"Compilation impact: {avg_same_seed/unc_42['mae']*100:.4f}%")
PYEOF

