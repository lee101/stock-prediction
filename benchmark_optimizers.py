#!/usr/bin/env python3
"""
Benchmark different optimizers for entry/exit multiplier optimization.
Compares scipy DE (baseline), BoTorch, Nevergrad, and EvoX.
"""

import time
import numpy as np
import torch
from typing import Tuple, Callable
from loss_utils import calculate_trading_profit_torch_with_entry_buysell

# Generate test data
np.random.seed(42)
torch.manual_seed(42)

n = 100
close_actual = torch.randn(n, device='cuda' if torch.cuda.is_available() else 'cpu') * 0.02
high_actual = close_actual + torch.abs(torch.randn(n, device=close_actual.device)) * 0.01
low_actual = close_actual - torch.abs(torch.randn(n, device=close_actual.device)) * 0.01
high_pred = torch.randn(n, device=close_actual.device) * 0.01 + 0.005
low_pred = torch.randn(n, device=close_actual.device) * 0.01 - 0.005
positions = torch.where(high_pred > low_pred, torch.ones(n, device=close_actual.device), -torch.ones(n, device=close_actual.device))

bounds = [(-0.03, 0.03), (-0.03, 0.03)]

def objective(h_mult: float, l_mult: float) -> float:
    """Single evaluation objective"""
    profit = calculate_trading_profit_torch_with_entry_buysell(
        None, None, close_actual, positions,
        high_actual, high_pred + h_mult,
        low_actual, low_pred + l_mult,
    ).item()
    return profit

def objective_batch(params_batch: torch.Tensor) -> torch.Tensor:
    """Batched objective for GPU - params_batch shape (batch_size, 2)"""
    profits = []
    for params in params_batch:
        h_mult, l_mult = params[0].item(), params[1].item()
        profit = calculate_trading_profit_torch_with_entry_buysell(
            None, None, close_actual, positions,
            high_actual, high_pred + h_mult,
            low_actual, low_pred + l_mult,
        ).item()
        profits.append(profit)
    return torch.tensor(profits, device=params_batch.device)


# 1. Baseline: scipy differential_evolution
def benchmark_scipy_de():
    from scipy.optimize import differential_evolution

    def obj_fn(x):
        return -objective(x[0], x[1])

    start = time.time()
    result = differential_evolution(
        obj_fn,
        bounds=bounds,
        maxiter=50,
        popsize=10,
        seed=42,
        workers=1,  # sequential for simplicity in benchmark
    )
    elapsed = time.time() - start

    return {
        'name': 'scipy DE',
        'time': elapsed,
        'profit': -result.fun,
        'params': result.x,
        'n_evals': result.nfev,
    }


# 2. BoTorch (Bayesian Optimization)
def benchmark_botorch():
    try:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import UpperConfidenceBound
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32

        # Convert bounds
        bounds_t = torch.tensor(bounds, device=device, dtype=dtype).T

        # Initial random samples
        n_init = 10
        train_x = torch.rand(n_init, 2, device=device, dtype=dtype) * 0.06 - 0.03
        train_y = objective_batch(train_x).unsqueeze(-1)

        start = time.time()

        # Optimization loop
        n_iterations = 20
        for i in range(n_iterations):
            # Fit GP model
            gp = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Get next candidate
            UCB = UpperConfidenceBound(gp, beta=0.1)
            candidate, _ = optimize_acqf(
                UCB,
                bounds=bounds_t,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )

            # Evaluate
            new_y = objective_batch(candidate).unsqueeze(-1)

            # Update
            train_x = torch.cat([train_x, candidate])
            train_y = torch.cat([train_y, new_y])

        elapsed = time.time() - start
        best_idx = train_y.argmax()

        return {
            'name': 'BoTorch',
            'time': elapsed,
            'profit': train_y[best_idx].item(),
            'params': train_x[best_idx].cpu().numpy(),
            'n_evals': len(train_y),
        }
    except ImportError as e:
        return {'name': 'BoTorch', 'error': f'Not installed: {e}'}


# 3. Nevergrad
def benchmark_nevergrad():
    try:
        import nevergrad as ng

        start = time.time()

        # Create optimizer
        parametrization = ng.p.Array(shape=(2,), lower=-0.03, upper=0.03)
        optimizer = ng.optimizers.TwoPointsDE(parametrization=parametrization, budget=300)

        # Can batch evaluations here
        batch_size = 10
        n_evals = 0

        for _ in range(optimizer.budget // batch_size):
            # Ask for batch of candidates
            candidates = [optimizer.ask() for _ in range(batch_size)]

            # Evaluate batch (could parallelize on GPU)
            params_batch = torch.tensor([c.value for c in candidates], dtype=torch.float32)
            if torch.cuda.is_available():
                params_batch = params_batch.cuda()

            profits = objective_batch(params_batch).cpu().numpy()

            # Tell results
            for cand, profit in zip(candidates, profits):
                optimizer.tell(cand, -profit)

            n_evals += batch_size

        elapsed = time.time() - start
        recommendation = optimizer.provide_recommendation()

        return {
            'name': 'Nevergrad',
            'time': elapsed,
            'profit': -optimizer.current_bests["pessimistic"].mean,
            'params': recommendation.value,
            'n_evals': n_evals,
        }
    except ImportError as e:
        return {'name': 'Nevergrad', 'error': f'Not installed: {e}'}


# 4. EvoX (GPU-accelerated DE)
def benchmark_evox():
    try:
        import evox
        from evox import algorithms, problems, workflows

        # EvoX expects minimization and specific problem structure
        # This is a simplified example - actual integration needs more work
        return {'name': 'EvoX', 'error': 'Integration pending (complex setup)'}
    except ImportError as e:
        return {'name': 'EvoX', 'error': f'Not installed: {e}'}


def main():
    print("="*80)
    print("OPTIMIZER BENCHMARK")
    print("="*80)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Data size: {n} timesteps")
    print(f"Bounds: {bounds}")
    print()

    results = []

    # Run benchmarks
    benchmarks = [
        ('Scipy DE (baseline)', benchmark_scipy_de),
        ('BoTorch', benchmark_botorch),
        ('Nevergrad', benchmark_nevergrad),
        ('EvoX', benchmark_evox),
    ]

    for name, benchmark_fn in benchmarks:
        print(f"Running {name}...")
        try:
            result = benchmark_fn()
            results.append(result)

            if 'error' in result:
                print(f"  ✗ {result['error']}")
            else:
                print(f"  ✓ Time: {result['time']:.3f}s")
                print(f"    Profit: {result['profit']:.6f}")
                print(f"    Params: {result['params']}")
                print(f"    Evals: {result['n_evals']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({'name': name, 'error': str(e)})
        print()

    # Summary table
    print("="*80)
    print("SUMMARY")
    print("="*80)

    successful_results = [r for r in results if 'error' not in r]
    if len(successful_results) > 1:
        baseline = next((r for r in successful_results if r['name'] == 'scipy DE'), successful_results[0])

        print(f"{'Optimizer':<20} {'Time (s)':<12} {'Speedup':<10} {'Profit':<12} {'Evals':<10}")
        print("-"*80)

        for r in successful_results:
            speedup = baseline['time'] / r['time'] if r['time'] > 0 else float('inf')
            print(f"{r['name']:<20} {r['time']:<12.3f} {speedup:<10.2f}x {r['profit']:<12.6f} {r['n_evals']:<10}")

    print()
    print("Recommendation:")
    print("1. If BoTorch works: Use it (most sample-efficient for 2D)")
    print("2. Otherwise Nevergrad: Easy batching, good speedup")
    print("3. Keep scipy as fallback")


if __name__ == '__main__':
    main()
