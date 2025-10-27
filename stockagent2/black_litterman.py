from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .views_schema import LLMViews


@dataclass(frozen=True)
class BlackLittermanResult:
    """Posterior mean/covariance after injecting LLM views."""

    mu_prior: np.ndarray
    mu_market_equilibrium: np.ndarray
    mu_posterior: np.ndarray
    sigma_prior: np.ndarray
    sigma_posterior: np.ndarray
    tau: float
    market_weight: float


def equilibrium_excess_returns(
    sigma: np.ndarray,
    market_weights: np.ndarray,
    *,
    risk_aversion: float,
) -> np.ndarray:
    """
    Reverse-optimise the implied excess returns that would make the market
    portfolio optimal under mean-variance utility with risk_aversion λ.
    """
    cov = np.asarray(sigma, dtype=float)
    weights = np.asarray(market_weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("market_weights must be a 1-D vector.")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("sigma must be a square covariance matrix.")
    if cov.shape[0] != weights.shape[0]:
        raise ValueError("Covariance and weights dimension mismatch.")
    lam = float(risk_aversion)
    if lam <= 0:
        raise ValueError("risk_aversion must be positive.")
    return lam * cov @ weights


def black_litterman_posterior(
    sigma: np.ndarray,
    tau: float,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Black–Litterman posterior expected returns and covariance.

    Parameters use the original notation from the seminal paper.
    """
    cov = np.asarray(sigma, dtype=float)
    prior = np.asarray(pi, dtype=float)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    n = cov.shape[0]
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if prior.shape != (n,):
        raise ValueError("Implied returns must match covariance dimension.")
    if P.ndim != 2 or P.shape[1] != n:
        raise ValueError("Pick matrix P has incompatible dimensions.")
    if Q.shape != (P.shape[0],):
        raise ValueError("View vector Q must align with pick matrix rows.")
    if Omega.shape != (P.shape[0], P.shape[0]):
        raise ValueError("Omega must be square with size equal to number of views.")
    if tau <= 0:
        raise ValueError("Tau must be positive.")

    tau_sigma_inv = np.linalg.inv(tau * cov)
    omega_inv = np.linalg.inv(Omega)

    middle = P.T @ omega_inv @ P
    sigma_post = np.linalg.inv(tau_sigma_inv + middle)
    mu_post = sigma_post @ (tau_sigma_inv @ prior + P.T @ omega_inv @ Q)
    sigma_post = (sigma_post + sigma_post.T) * 0.5  # enforce symmetry
    return mu_post, sigma_post


class BlackLittermanFuser:
    """
    Convenience wrapper that validates dimensions and gracefully handles the
    absence of discretionary views.
    """

    def __init__(self, *, tau: float = 0.05, market_prior_weight: float = 0.5) -> None:
        if tau <= 0:
            raise ValueError("Tau must be strictly positive.")
        if not 0.0 <= market_prior_weight <= 1.0:
            raise ValueError("market_prior_weight must lie in [0, 1].")
        self.tau = float(tau)
        self.market_prior_weight = float(market_prior_weight)

    def fuse(
        self,
        mu_prior: np.ndarray,
        sigma_prior: np.ndarray,
        *,
        market_weights: Optional[np.ndarray],
        risk_aversion: float,
        views: Optional[LLMViews],
        universe: Sequence[str],
    ) -> BlackLittermanResult:
        prior = np.asarray(mu_prior, dtype=float)
        cov = np.asarray(sigma_prior, dtype=float)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("sigma_prior must be square.")
        if prior.shape != (cov.shape[0],):
            raise ValueError("mu_prior and sigma_prior dimension mismatch.")

        if market_weights is None:
            market_weights = np.full_like(prior, 1.0 / prior.size)
        else:
            market_weights = np.asarray(market_weights, dtype=float)
            if market_weights.shape != prior.shape:
                raise ValueError("market_weights dimension mismatch.")
            if not np.isclose(market_weights.sum(), 1.0):
                market_weights = market_weights / market_weights.sum()

        pi_market = equilibrium_excess_returns(
            cov,
            market_weights,
            risk_aversion=risk_aversion,
        )
        pi = self.market_prior_weight * pi_market + (1.0 - self.market_prior_weight) * prior

        if views is None:
            return BlackLittermanResult(
                mu_prior=prior,
                mu_market_equilibrium=pi_market,
                mu_posterior=pi,
                sigma_prior=cov,
                sigma_posterior=cov,
                tau=self.tau,
                market_weight=self.market_prior_weight,
            )

        P, Q, Omega, _ = views.black_litterman_inputs(universe)
        if P.size == 0:
            return BlackLittermanResult(
                mu_prior=prior,
                mu_market_equilibrium=pi_market,
                mu_posterior=pi,
                sigma_prior=cov,
                sigma_posterior=cov,
                tau=self.tau,
                market_weight=self.market_prior_weight,
            )

        mu_post, sigma_post = black_litterman_posterior(
            cov,
            self.tau,
            pi,
            P,
            Q,
            Omega,
        )
        return BlackLittermanResult(
            mu_prior=prior,
            mu_market_equilibrium=pi_market,
            mu_posterior=mu_post,
            sigma_prior=cov,
            sigma_posterior=sigma_post,
            tau=self.tau,
            market_weight=self.market_prior_weight,
        )
