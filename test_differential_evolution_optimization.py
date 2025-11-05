import numpy as np
import torch
from loss_utils import (
    calculate_trading_profit_torch_with_entry_buysell,
)
from scipy.optimize import differential_evolution


def test_differential_evolution_vs_grid():
    """
    Compare differential_evolution to grid search for finding optimal multipliers
    """
    # Setup test data
    np.random.seed(42)
    torch.manual_seed(42)

    n = 50
    close_actual = torch.randn(n) * 0.02  # random returns
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01

    high_pred = torch.randn(n) * 0.01 + 0.005
    low_pred = torch.randn(n) * 0.01 - 0.005

    # Position: buy when high > low
    positions = torch.where(high_pred > low_pred, torch.ones(n), -torch.ones(n))

    print("Testing MaxDiff-style optimization")
    print("=" * 60)

    # Method 1: Grid search (old way)
    print("\n1. Grid search (500 x 500 = 250k evaluations)")
    import time

    start = time.time()

    best_grid_profit = float("-inf")
    best_grid_h = 0.0
    best_grid_l = 0.0

    for h_mult in np.linspace(-0.03, 0.03, 500):
        for l_mult in np.linspace(-0.03, 0.03, 500):
            profit = calculate_trading_profit_torch_with_entry_buysell(
                None,
                None,
                close_actual,
                positions,
                high_actual,
                high_pred + float(h_mult),
                low_actual,
                low_pred + float(l_mult),
            ).item()
            if profit > best_grid_profit:
                best_grid_profit = profit
                best_grid_h = h_mult
                best_grid_l = l_mult

    grid_time = time.time() - start
    print(f"  Time: {grid_time:.2f}s")
    print(f"  Best profit: {best_grid_profit:.6f}")
    print(f"  Best multipliers: h={best_grid_h:.6f}, l={best_grid_l:.6f}")

    # Method 2: Differential evolution (new way)
    print("\n2. Differential evolution (~500 evaluations)")
    start = time.time()

    def objective(multipliers):
        h_mult, l_mult = multipliers
        profit = calculate_trading_profit_torch_with_entry_buysell(
            None,
            None,
            close_actual,
            positions,
            high_actual,
            high_pred + float(h_mult),
            low_actual,
            low_pred + float(l_mult),
        ).item()
        return -profit

    result = differential_evolution(
        objective,
        bounds=[(-0.03, 0.03), (-0.03, 0.03)],
        maxiter=50,
        popsize=10,
        atol=1e-5,
        seed=42,
        workers=1,
    )

    de_time = time.time() - start
    de_profit = -result.fun
    de_h = result.x[0]
    de_l = result.x[1]

    print(f"  Time: {de_time:.2f}s")
    print(f"  Best profit: {de_profit:.6f}")
    print(f"  Best multipliers: h={de_h:.6f}, l={de_l:.6f}")
    print(f"  Function evaluations: {result.nfev}")

    print("\n" + "=" * 60)
    print("Comparison:")
    print(f"  Speedup: {grid_time / de_time:.1f}x faster")
    print(f"  Profit difference: {abs(de_profit - best_grid_profit):.6f}")
    print(f"  Evaluations: {result.nfev} vs 250,000")

    # Verify profit is close
    assert abs(de_profit - best_grid_profit) < 0.001, (
        f"DE profit {de_profit} too different from grid {best_grid_profit}"
    )

    print("\n✓ Differential evolution finds similar optimum with ~500x fewer evaluations")


if __name__ == "__main__":
    test_differential_evolution_vs_grid()
