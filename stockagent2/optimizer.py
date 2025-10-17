from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize

from .config import OptimizationConfig

try:  # pragma: no cover - cvxpy is optional at import time, required at runtime
    import cvxpy as cp
except Exception:  # pragma: no cover - defer error until solve() is called
    cp = None  # type: ignore


@dataclass(frozen=True)
class OptimizerResult:
    weights: np.ndarray
    expected_return: float
    risk: float
    objective_value: float
    turnover: float
    status: str
    solver: str
    sector_exposures: Dict[str, float]


class CostAwareOptimizer:
    """
    Convex optimiser that penalises variance, turnover, and transaction costs
    while honouring exposure constraints.
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def _build_sector_constraints(
        self,
        variable: "cp.Expression",
        universe: Sequence[str],
        sector_map: Optional[Mapping[str, str]],
    ):
        if not self.config.sector_exposure_limits:
            return []
        if not sector_map:
            return []

        constraints = []
        weights_by_sector: Dict[str, np.ndarray] = {}
        for idx, symbol in enumerate(universe):
            sector = sector_map.get(symbol.upper())
            if sector is None:
                continue
            weights_by_sector.setdefault(sector, np.zeros(len(universe), dtype=float))[idx] = 1.0

        for sector, mask in weights_by_sector.items():
            if sector not in self.config.sector_exposure_limits:
                continue
            limit = float(self.config.sector_exposure_limits[sector])
            if limit <= 0:
                continue
            if np.allclose(mask, 0.0):
                continue
            mask_const = cp.Constant(mask)
            constraints.append(mask_const @ variable <= limit)
            constraints.append(mask_const @ variable >= -limit)
        return constraints

    def solve(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        *,
        previous_weights: Optional[np.ndarray] = None,
        universe: Sequence[str],
        sector_map: Optional[Mapping[str, str]] = None,
        solver: str = "OSQP",
    ) -> OptimizerResult:
        mu_vec = np.asarray(mu, dtype=float)
        cov = np.asarray(sigma, dtype=float)
        n = mu_vec.shape[0]
        if cov.shape != (n, n):
            raise ValueError("mu and sigma dimension mismatch.")
        if previous_weights is None:
            previous_weights = np.zeros(n, dtype=float)
        prev = np.asarray(previous_weights, dtype=float)
        if prev.shape != (n,):
            raise ValueError("previous_weights dimension mismatch.")

        # Symmetrise covariance to avoid solver noise.
        cov = (cov + cov.T) * 0.5

        sector_norm = self._normalise_sector_map(sector_map)
        penalty_scale = (self.config.transaction_cost_bps + self.config.turnover_penalty_bps) / 1e4
        net_target = float(self.config.net_exposure_target)
        gross_limit = float(self.config.gross_exposure_limit)
        lower_bound = max(-self.config.short_cap, self.config.min_weight)
        upper_bound = min(self.config.long_cap, self.config.max_weight)

        if cp is not None:
            try:
                return self._solve_with_cvxpy(
                    mu_vec,
                    cov,
                    prev,
                    universe,
                    sector_norm,
                    penalty_scale,
                    net_target,
                    gross_limit,
                    lower_bound,
                    upper_bound,
                    solver,
                )
            except Exception:
                pass

        return self._solve_with_slsqp(
            mu_vec,
            cov,
            prev,
            universe,
            sector_norm,
            penalty_scale,
            net_target,
            gross_limit,
            lower_bound,
            upper_bound,
        )

    def _solve_with_cvxpy(
        self,
        mu_vec: np.ndarray,
        cov: np.ndarray,
        prev: np.ndarray,
        universe: Sequence[str],
        sector_map: Optional[Dict[str, str]],
        penalty_scale: float,
        net_target: float,
        gross_limit: float,
        lower_bound: float,
        upper_bound: float,
        solver: str,
    ) -> OptimizerResult:
        w = cp.Variable(mu_vec.shape[0])
        risk_term = cp.quad_form(w, cov)
        turnover = cp.norm1(w - prev)

        objective = cp.Maximize(
            mu_vec @ w - self.config.risk_aversion * risk_term - penalty_scale * turnover
        )

        constraints = [
            cp.sum(w) == net_target,
            cp.norm1(w) <= gross_limit,
            w >= lower_bound,
            w <= upper_bound,
        ]
        constraints.extend(self._build_sector_constraints(w, universe, sector_map))

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=solver, warm_start=True)
        except Exception:
            problem.solve(solver="SCS", warm_start=True, verbose=False)

        if w.value is None:
            raise RuntimeError(f"Optimizer failed to converge (status={problem.status}).")

        weights = np.asarray(w.value, dtype=float)
        expected_return = float(mu_vec @ weights)
        risk = float(weights @ cov @ weights)
        turnover_value = float(np.sum(np.abs(weights - prev)))

        sector_exposures = self._compute_sector_exposures(weights, universe, sector_map)

        return OptimizerResult(
            weights=weights,
            expected_return=expected_return,
            risk=risk,
            objective_value=float(problem.value),
            turnover=turnover_value,
            status=str(problem.status),
            solver=str(problem.solver_stats.solver_name) if problem.solver_stats else solver,
            sector_exposures=sector_exposures,
        )

    def _solve_with_slsqp(
        self,
        mu_vec: np.ndarray,
        cov: np.ndarray,
        prev: np.ndarray,
        universe: Sequence[str],
        sector_map: Optional[Dict[str, str]],
        penalty_scale: float,
        net_target: float,
        gross_limit: float,
        lower_bound: float,
        upper_bound: float,
    ) -> OptimizerResult:
        n = mu_vec.shape[0]
        bounds = [(lower_bound, upper_bound)] * n
        eps = 1e-6

        def smooth_abs(x: np.ndarray) -> np.ndarray:
            return np.sqrt(x**2 + eps)

        def objective(w: np.ndarray) -> float:
            risk = w @ cov @ w
            turnover = np.sum(smooth_abs(w - prev))
            return -float(mu_vec @ w - self.config.risk_aversion * risk - penalty_scale * turnover)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - net_target},
            {"type": "ineq", "fun": lambda w: gross_limit - np.sum(smooth_abs(w))},
        ]

        if sector_map:
            for sector, limit in self.config.sector_exposure_limits.items():
                if limit <= 0:
                    continue
                mask = np.array(
                    [1.0 if sector_map.get(symbol.upper()) == sector else 0.0 for symbol in universe],
                    dtype=float,
                )
                if not np.any(mask):
                    continue
                constraints.append({"type": "ineq", "fun": lambda w, m=mask, lim=limit: lim - m @ w})
                constraints.append({"type": "ineq", "fun": lambda w, m=mask, lim=limit: lim + m @ w})

        result = optimize.minimize(
            objective,
            x0=np.clip(prev, lower_bound, upper_bound),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"SLSQP failed to converge: {result.message}")

        weights = np.asarray(result.x, dtype=float)
        expected_return = float(mu_vec @ weights)
        risk = float(weights @ cov @ weights)
        turnover_value = float(np.sum(np.abs(weights - prev)))
        sector_exposures = self._compute_sector_exposures(weights, universe, sector_map)

        return OptimizerResult(
            weights=weights,
            expected_return=expected_return,
            risk=risk,
            objective_value=-float(result.fun),
            turnover=turnover_value,
            status="SLSQP_success",
            solver="SLSQP",
            sector_exposures=sector_exposures,
        )

    def _normalise_sector_map(
        self,
        sector_map: Optional[Mapping[str, str]],
    ) -> Optional[Dict[str, str]]:
        if sector_map is None:
            return None
        return {symbol.upper(): sector for symbol, sector in sector_map.items()}

    def _compute_sector_exposures(
        self,
        weights: np.ndarray,
        universe: Sequence[str],
        sector_map: Optional[Mapping[str, str]],
    ) -> Dict[str, float]:
        if not sector_map:
            return {}
        exposures: Dict[str, float] = {}
        for weight, symbol in zip(weights, universe):
            sector = sector_map.get(symbol.upper())
            if sector is None:
                continue
            exposures[sector] = exposures.get(sector, 0.0) + float(weight)
        return exposures
