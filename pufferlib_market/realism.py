"""Shared production-realism defaults for pufferlib-market entrypoints."""

PRODUCTION_DECISION_LAG = 2
PRODUCTION_FEE_BPS = 10.0
PRODUCTION_FILL_BUFFER_BPS = 5.0
PRODUCTION_SHORT_BORROW_APR = 0.0625


def require_production_decision_lag(decision_lag: int, *, allow_low_lag_diagnostics: bool) -> int:
    """Validate evaluator decision lag and require explicit opt-in below prod lag."""
    lag = int(decision_lag)
    if lag < 0:
        raise ValueError("--decision-lag must be >= 0")
    if lag < PRODUCTION_DECISION_LAG and not bool(allow_low_lag_diagnostics):
        raise ValueError("decision_lag below 2 requires --allow-low-lag-diagnostics")
    return lag


__all__ = [
    "PRODUCTION_DECISION_LAG",
    "PRODUCTION_FEE_BPS",
    "PRODUCTION_FILL_BUFFER_BPS",
    "PRODUCTION_SHORT_BORROW_APR",
    "require_production_decision_lag",
]
