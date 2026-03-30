from __future__ import annotations

FDUSD_PAIRS: frozenset[str] = frozenset({
    "BTCFDUSD", "ETHFDUSD", "SOLFDUSD", "BNBFDUSD",
})

DEFAULT_USDT_FEE_BPS = 10.0
DEFAULT_FDUSD_FEE_BPS = 0.0
MARGIN_ANNUAL_RATE = 0.0625


def get_fee_bps(pair: str, *, custom_fees: dict[str, float] | None = None) -> float:
    pair = pair.upper().replace("/", "").replace("-", "")
    if custom_fees and pair in custom_fees:
        return custom_fees[pair]
    return DEFAULT_FDUSD_FEE_BPS if pair in FDUSD_PAIRS else DEFAULT_USDT_FEE_BPS


def fee_fraction(pair: str, *, custom_fees: dict[str, float] | None = None) -> float:
    return get_fee_bps(pair, custom_fees=custom_fees) / 10_000.0


def margin_cost_per_hour(position_value: float, annual_rate: float = MARGIN_ANNUAL_RATE) -> float:
    return abs(position_value) * annual_rate / 8760.0
