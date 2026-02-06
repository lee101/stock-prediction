"""
Binance margin loan interest rates (hardcoded snapshot).

These numbers are intentionally hardcoded to make experiments reproducible.
They will drift from Binance over time; update the table when you want your
training/sim assumptions to reflect current rates.

All values below are *percent* rates, matching the Binance UI:
  - hourly_pct: percent per hour (e.g. 0.000441 means 0.000441% per hour)
  - yearly_pct: annualised percent (approximately hourly_pct * 24 * 365)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping


HOURS_PER_YEAR: int = 24 * 365


@dataclass(frozen=True)
class MarginInterestRatePct:
    """Margin interest rates expressed in percent units."""

    hourly_pct: float
    yearly_pct: float

    @property
    def hourly(self) -> float:
        """Hourly rate as a decimal fraction (e.g. 0.00000441 == 0.000441%)."""
        return float(self.hourly_pct) / 100.0

    @property
    def yearly(self) -> float:
        """Yearly rate as a decimal fraction (e.g. 0.0386 == 3.86%)."""
        return float(self.yearly_pct) / 100.0

    def yearly_pct_from_hourly(self, *, hours_per_year: int = HOURS_PER_YEAR) -> float:
        """Compute annualised percent implied by the hourly percent rate."""
        return float(self.hourly_pct) * float(hours_per_year)


def _normalize_asset(asset: str) -> str:
    if not isinstance(asset, str):
        raise TypeError(f"Borrowed asset must be a string, received {type(asset).__name__}.")
    normalized = asset.strip().upper()
    if not normalized:
        raise ValueError("Borrowed asset must be a non-empty string.")
    if not normalized.isalnum():
        raise ValueError(
            f"Borrowed asset must be alphanumeric (e.g. 'USDT'), received {asset!r}."
        )
    return normalized


def _normalize_tier(tier: str) -> str:
    if not isinstance(tier, str):
        raise TypeError(f"Tier must be a string, received {type(tier).__name__}.")
    cleaned = tier.strip().lower().replace("-", "").replace("_", "")
    if cleaned in {"vip1", "1"}:
        return "vip1"
    if cleaned in {"standard", "vip0", "normal", "0"}:
        return "standard"
    raise ValueError(f"Unknown Binance margin tier {tier!r}. Expected 'standard' or 'VIP1'.")


# Snapshot taken from a user-provided table (VIP1 displayed alongside standard).
# Values are percent rates as shown by Binance (hourly / yearly).
BINANCE_MARGIN_INTEREST_RATES_PCT: Mapping[str, Mapping[str, MarginInterestRatePct]] = {
    "USDT": {
        "standard": MarginInterestRatePct(hourly_pct=0.00044504, yearly_pct=3.90),
        "vip1": MarginInterestRatePct(hourly_pct=0.000441, yearly_pct=3.86),
    },
    "USDC": {
        "standard": MarginInterestRatePct(hourly_pct=0.00059458, yearly_pct=5.21),
        "vip1": MarginInterestRatePct(hourly_pct=0.000589, yearly_pct=5.16),
    },
    "BTC": {
        "standard": MarginInterestRatePct(hourly_pct=0.00004200, yearly_pct=0.37),
        "vip1": MarginInterestRatePct(hourly_pct=0.000040, yearly_pct=0.35),
    },
    "SOL": {
        "standard": MarginInterestRatePct(hourly_pct=0.00098683, yearly_pct=8.64),
        "vip1": MarginInterestRatePct(hourly_pct=0.000962, yearly_pct=8.43),
    },
    "ETH": {
        "standard": MarginInterestRatePct(hourly_pct=0.00026471, yearly_pct=2.32),
        "vip1": MarginInterestRatePct(hourly_pct=0.000262, yearly_pct=2.30),
    },
    "XRP": {
        "standard": MarginInterestRatePct(hourly_pct=0.00046904, yearly_pct=4.11),
        "vip1": MarginInterestRatePct(hourly_pct=0.000457, yearly_pct=4.01),
    },
    "BCH": {
        "standard": MarginInterestRatePct(hourly_pct=0.00101708, yearly_pct=8.91),
        "vip1": MarginInterestRatePct(hourly_pct=0.000992, yearly_pct=8.69),
    },
    "DOGE": {
        "standard": MarginInterestRatePct(hourly_pct=0.00049179, yearly_pct=4.31),
        "vip1": MarginInterestRatePct(hourly_pct=0.000480, yearly_pct=4.20),
    },
    "ADA": {
        "standard": MarginInterestRatePct(hourly_pct=0.00069546, yearly_pct=6.09),
        "vip1": MarginInterestRatePct(hourly_pct=0.000678, yearly_pct=5.94),
    },
    "SUI": {
        "standard": MarginInterestRatePct(hourly_pct=0.00041883, yearly_pct=3.67),
        "vip1": MarginInterestRatePct(hourly_pct=0.000408, yearly_pct=3.58),
    },
    "TRUMP": {
        "standard": MarginInterestRatePct(hourly_pct=0.00446513, yearly_pct=39.11),
        "vip1": MarginInterestRatePct(hourly_pct=0.004353, yearly_pct=38.14),
    },
    "AXS": {
        "standard": MarginInterestRatePct(hourly_pct=0.01490733, yearly_pct=130.59),
        "vip1": MarginInterestRatePct(hourly_pct=0.014535, yearly_pct=127.32),
    },
    "PAXG": {
        "standard": MarginInterestRatePct(hourly_pct=0.00027317, yearly_pct=2.39),
        "vip1": MarginInterestRatePct(hourly_pct=0.000266, yearly_pct=2.33),
    },
    "LTC": {
        "standard": MarginInterestRatePct(hourly_pct=0.00034371, yearly_pct=3.01),
        "vip1": MarginInterestRatePct(hourly_pct=0.000335, yearly_pct=2.94),
    },
    "LINK": {
        "standard": MarginInterestRatePct(hourly_pct=0.00012983, yearly_pct=1.14),
        "vip1": MarginInterestRatePct(hourly_pct=0.000127, yearly_pct=1.11),
    },
    "POL": {
        "standard": MarginInterestRatePct(hourly_pct=0.00246700, yearly_pct=21.61),
        "vip1": MarginInterestRatePct(hourly_pct=0.002405, yearly_pct=21.07),
    },
    "ZEC": {
        "standard": MarginInterestRatePct(hourly_pct=0.00480904, yearly_pct=42.13),
        "vip1": MarginInterestRatePct(hourly_pct=0.004689, yearly_pct=41.07),
    },
    "BNB": {
        "standard": MarginInterestRatePct(hourly_pct=0.00913200, yearly_pct=80.00),
        "vip1": MarginInterestRatePct(hourly_pct=0.009132, yearly_pct=80.00),
    },
    "TON": {
        "standard": MarginInterestRatePct(hourly_pct=0.00180025, yearly_pct=15.77),
        "vip1": MarginInterestRatePct(hourly_pct=0.001800, yearly_pct=15.77),
    },
    "AVAX": {
        "standard": MarginInterestRatePct(hourly_pct=0.00063750, yearly_pct=5.58),
        "vip1": MarginInterestRatePct(hourly_pct=0.000622, yearly_pct=5.44),
    },
}


def get_binance_margin_interest_rate_pct(
    borrowed_asset: str,
    *,
    tier: str = "VIP1",
) -> MarginInterestRatePct:
    """Fetch hardcoded Binance margin interest rates for a borrowed asset."""
    asset = _normalize_asset(borrowed_asset)
    tier_norm = _normalize_tier(tier)
    tiers = BINANCE_MARGIN_INTEREST_RATES_PCT.get(asset)
    if tiers is None:
        raise KeyError(
            f"Unsupported borrowed asset {borrowed_asset!r} for hardcoded Binance margin rates."
        )
    rate = tiers.get(tier_norm)
    if rate is None:
        raise KeyError(
            f"Unsupported tier {tier!r} for borrowed asset {borrowed_asset!r}. "
            f"Supported: {sorted(tiers.keys())}"
        )
    return rate


def compute_simple_margin_interest(
    principal: float,
    hours: float,
    *,
    borrowed_asset: str,
    tier: str = "VIP1",
) -> float:
    """
    Compute simple (non-compounded) interest cost for a Binance margin loan.

    Interest is calculated as:
        principal * hourly_rate_decimal * hours
    """
    principal_f = float(principal)
    hours_f = float(hours)
    if not math.isfinite(principal_f) or principal_f < 0.0:
        raise ValueError(f"principal must be a finite non-negative float, received {principal!r}.")
    if not math.isfinite(hours_f) or hours_f < 0.0:
        raise ValueError(f"hours must be a finite non-negative float, received {hours!r}.")
    if principal_f == 0.0 or hours_f == 0.0:
        return 0.0
    rate = get_binance_margin_interest_rate_pct(borrowed_asset, tier=tier)
    return principal_f * rate.hourly * hours_f


def compute_compound_margin_interest(
    principal: float,
    hours: float,
    *,
    borrowed_asset: str,
    tier: str = "VIP1",
) -> float:
    """
    Compute interest cost assuming hourly compounding.

    This models interest being added to the liability each hour. For small rates
    and short horizons this is close to simple interest; for long horizons the
    difference can matter.
    """
    principal_f = float(principal)
    hours_f = float(hours)
    if not math.isfinite(principal_f) or principal_f < 0.0:
        raise ValueError(f"principal must be a finite non-negative float, received {principal!r}.")
    if not math.isfinite(hours_f) or hours_f < 0.0:
        raise ValueError(f"hours must be a finite non-negative float, received {hours!r}.")
    if principal_f == 0.0 or hours_f == 0.0:
        return 0.0
    rate = get_binance_margin_interest_rate_pct(borrowed_asset, tier=tier)
    hourly = rate.hourly
    if hourly <= 0.0:
        return 0.0
    # Use exp(log1p(x)*t) for numerical stability for very small hourly rates.
    growth = math.exp(math.log1p(hourly) * hours_f)
    return principal_f * (growth - 1.0)


def get_supported_borrowed_assets() -> Dict[str, Dict[str, MarginInterestRatePct]]:
    """Return a copy of the supported asset->tier->rate mapping."""
    return {
        asset: dict(tiers)
        for asset, tiers in BINANCE_MARGIN_INTEREST_RATES_PCT.items()
    }


__all__ = [
    "HOURS_PER_YEAR",
    "MarginInterestRatePct",
    "BINANCE_MARGIN_INTEREST_RATES_PCT",
    "get_binance_margin_interest_rate_pct",
    "compute_simple_margin_interest",
    "compute_compound_margin_interest",
    "get_supported_borrowed_assets",
]

