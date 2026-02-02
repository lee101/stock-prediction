"""Configuration dataclasses for Bags.fm trading bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

# Try to import credentials from env_real, fall back to env vars
try:
    from env_real import (
        BAGS_API_KEY,
        BIRDSEYE_API_KEY,
        HELIUS_API_KEY,
        JUP_API_KEY,
        SOLANA_PRIVATE_KEY,
        SOLANA_PUBLIC_KEY,
    )
except ImportError:
    BAGS_API_KEY = os.getenv("BAGS_API_KEY", "")
    BIRDSEYE_API_KEY = os.getenv("BIRDSEYE_API_KEY", "")
    HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
    JUP_API_KEY = os.getenv("JUP_API_KEY", "")
    # Support both SOLANA_PRIVATE_KEY and PRIVATE_KEY (Bags SDK convention)
    SOLANA_PRIVATE_KEY = os.getenv("SOLANA_PRIVATE_KEY", "") or os.getenv("PRIVATE_KEY", "")
    SOLANA_PUBLIC_KEY = os.getenv("SOLANA_PUBLIC_KEY", "")


# Well-known Solana token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
# Bags.fm token mints
CODEX_MINT = "HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS"
BLON_MINT = "E2KVeSMV6GzRxMuVa4GXA42RwQeCL959VwFxTbmqBAGS"

# Bags.fm fee structure (bps)
BAGS_PLATFORM_FEE_BPS: float = 100.0
BAGS_CREATOR_FEE_BPS: float = 100.0
BAGS_TOTAL_FEE_BPS: float = BAGS_PLATFORM_FEE_BPS + BAGS_CREATOR_FEE_BPS


@dataclass
class BagsConfig:
    """Configuration for Bags.fm API."""

    api_key: str = field(default_factory=lambda: BAGS_API_KEY)
    base_url: str = "https://public-api-v2.bags.fm/api/v1"

    # RPC configuration
    rpc_url: str = field(
        default_factory=lambda: os.getenv(
            "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
        )
    )

    # Helius RPC (higher rate limits, priority fee estimation)
    helius_api_key: str = field(default_factory=lambda: HELIUS_API_KEY)

    # Wallet configuration
    public_key: str = field(
        default_factory=lambda: SOLANA_PUBLIC_KEY
    )
    private_key_b58: str = field(
        default_factory=lambda: SOLANA_PRIVATE_KEY
    )

    # Request configuration
    timeout_seconds: int = 30
    max_retries: int = 5
    retry_delay_seconds: float = 0.5

    # RPC rate limiting
    rpc_rate_limit_rps: float = 10.0
    rpc_backoff_multiplier: float = 2.0
    rpc_max_backoff_seconds: float = 30.0


@dataclass
class TokenConfig:
    """Configuration for a tradeable token."""

    symbol: str
    mint: str
    decimals: int = 9
    name: str = ""

    # Trading parameters
    min_trade_amount: float = 0.001  # Minimum trade amount in token units
    max_position_pct: float = 0.25  # Max % of portfolio in this token
    entry_fee_bps: float = 0.0  # Token-specific entry fee/tax in bps
    exit_fee_bps: float = 0.0  # Token-specific exit fee/tax in bps
    spread_bps: float = 0.0  # Full bid/ask spread in bps (round-trip)
    creator_rebate_bps: float = 0.0  # Rebate to creator in bps (credited on volume)


# Default tokens to trade
DEFAULT_TOKENS: Dict[str, TokenConfig] = {
    "SOL": TokenConfig(
        symbol="SOL",
        mint=SOL_MINT,
        decimals=9,
        name="Solana",
        min_trade_amount=0.01,
    ),
    "USDC": TokenConfig(
        symbol="USDC",
        mint=USDC_MINT,
        decimals=6,
        name="USD Coin",
        min_trade_amount=1.0,
    ),
    "CODEX": TokenConfig(
        symbol="CODEX",
        mint=CODEX_MINT,
        decimals=9,
        name="CODEX (Bags.fm)",
        min_trade_amount=1.0,
        entry_fee_bps=BAGS_TOTAL_FEE_BPS,
        exit_fee_bps=BAGS_TOTAL_FEE_BPS,
        creator_rebate_bps=BAGS_CREATOR_FEE_BPS,
    ),
    "BLON": TokenConfig(
        symbol="BLON",
        mint=BLON_MINT,
        decimals=9,
        name="BLON",
        min_trade_amount=1.0,
        entry_fee_bps=BAGS_TOTAL_FEE_BPS,
        exit_fee_bps=BAGS_TOTAL_FEE_BPS,
    ),
}


@dataclass
class DataConfig:
    """Configuration for data collection and storage."""

    # Storage paths
    data_dir: Path = field(default_factory=lambda: Path("bagstraining"))
    price_history_file: str = "price_history.csv"
    ohlc_file: str = "ohlc_data.csv"

    # Tokens to track
    tracked_tokens: List[TokenConfig] = field(default_factory=lambda: list(DEFAULT_TOKENS.values()))

    # Base token for quotes (usually SOL or USDC)
    quote_token: TokenConfig = field(default_factory=lambda: DEFAULT_TOKENS["SOL"])

    # Data collection settings
    collection_interval_minutes: int = 10  # How often to collect prices
    ohlc_interval_minutes: int = 10  # OHLC bar interval (matches collection)

    # History settings
    context_bars: int = 512  # Number of OHLC bars for forecasting context
    max_history_days: int = 365  # Max days of history to keep

    @property
    def data_path(self) -> Path:
        """Full path to data directory."""
        return self.data_dir

    @property
    def price_history_path(self) -> Path:
        """Full path to price history file."""
        return self.data_dir / self.price_history_file

    @property
    def ohlc_path(self) -> Path:
        """Full path to OHLC file."""
        return self.data_dir / self.ohlc_file


@dataclass
class ForecastConfig:
    """Configuration for Chronos2 forecasting."""

    model_id: str = "amazon/chronos-2"
    device_map: str = "cuda"
    prediction_length: int = 6  # Predict 6 bars ahead (1 hour at 10min intervals)
    context_length: int = 512
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 32

    # Use univariate mode for crypto (typically better than multivariate)
    use_multivariate: bool = False

    # Pre-augmentation
    use_preaugmentation: bool = True
    preaugmentation_dirs: Tuple[str, ...] = (
        "preaugstrategies/chronos2",
        "preaugstrategies/best",
    )


@dataclass
class CostConfig:
    """Configuration for transaction cost modeling."""

    # AMM/DEX fees (embedded in quotes, but useful for estimation)
    estimated_swap_fee_bps: int = 30  # ~0.3% typical AMM fee

    # Slippage tolerance for swaps
    default_slippage_bps: int = 100  # 1% default slippage tolerance

    # Network fees (in lamports)
    estimated_base_fee_lamports: int = 5000  # ~0.000005 SOL base fee
    estimated_priority_fee_lamports: int = 100000  # ~0.0001 SOL priority

    # ATA (Associated Token Account) creation cost
    ata_rent_lamports: int = 2039280  # ~0.00204 SOL for new token account

    @property
    def estimated_total_fee_sol(self) -> float:
        """Estimated total network fee in SOL."""
        return (
            self.estimated_base_fee_lamports + self.estimated_priority_fee_lamports
        ) / 1e9


@dataclass
class SimulationConfig:
    """Configuration for market simulation."""

    # Initial portfolio
    initial_sol: float = 1.0  # Starting SOL balance
    initial_usdc: float = 0.0  # Starting USDC balance

    # Cost modeling
    costs: CostConfig = field(default_factory=CostConfig)

    # Trading rules
    min_trade_value_sol: float = 0.01  # Minimum trade value in SOL
    max_position_pct: float = 0.5  # Max % of portfolio in any single position
    max_position_sol: float = 1.0  # Max SOL notional across all positions

    # Risk controls
    max_daily_loss_pct: float = 0.10  # Stop trading if daily loss > 10%
    max_drawdown_pct: float = 0.20  # Stop trading if drawdown > 20%
    max_trades_per_day: Optional[int] = None  # Limit entries per day (None = no limit)

    # Simulation dates
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class TradingConfig:
    """Configuration for live trading."""

    # API configurations
    bags: BagsConfig = field(default_factory=BagsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    # Trading parameters
    strategy: Literal["forecast", "daily-range"] = "forecast"
    check_interval_minutes: int = 10  # How often to check for trades
    min_predicted_return: float = 0.005  # Minimum predicted return to trade (0.5%)
    min_confidence: float = 0.6  # Minimum forecast confidence
    daily_range_min_bps: float = 100.0  # Min prior-day range in bps to trade
    daily_range_max_actions_per_day: int = 2  # Per-token actions/day for daily-range
    daily_range_use_previous_day: bool = True  # Use prior-day levels if True

    # Position sizing
    position_size_pct: float = 0.25  # % of portfolio per position
    max_positions: int = 4  # Maximum concurrent positions
    max_position_sol: float = 1.0  # Max SOL notional across all positions

    # Slippage
    slippage_bps: int = 100  # 1% slippage tolerance

    # Risk controls
    max_daily_loss_pct: float = 0.05  # Stop trading if daily loss > 5%
    max_daily_trades: int = 20  # Max trades per day

    # Execution
    dry_run: bool = True  # If True, don't execute actual swaps
    log_level: str = "INFO"

    # State persistence
    state_file: Path = field(default_factory=lambda: Path("bagsfm_state.json"))


def create_default_config() -> TradingConfig:
    """Create a default trading configuration."""
    return TradingConfig()


def load_config_from_env() -> TradingConfig:
    """Load configuration from environment variables."""
    config = TradingConfig()

    # Override from environment
    if os.getenv("BAGS_DRY_RUN", "1") in ("0", "false", "FALSE"):
        config.dry_run = False

    if os.getenv("BAGS_CHECK_INTERVAL"):
        config.check_interval_minutes = int(os.getenv("BAGS_CHECK_INTERVAL"))

    if os.getenv("BAGS_SLIPPAGE_BPS"):
        config.slippage_bps = int(os.getenv("BAGS_SLIPPAGE_BPS"))

    if os.getenv("BAGS_MIN_RETURN"):
        config.min_predicted_return = float(os.getenv("BAGS_MIN_RETURN"))

    if os.getenv("BAGS_MAX_POSITION_SOL"):
        config.max_position_sol = float(os.getenv("BAGS_MAX_POSITION_SOL"))

    if os.getenv("BAGS_STRATEGY"):
        strategy = os.getenv("BAGS_STRATEGY", "").strip().lower()
        if strategy in ("forecast", "daily-range"):
            config.strategy = strategy  # type: ignore[assignment]

    if os.getenv("BAGS_DAILY_RANGE_MIN_BPS"):
        config.daily_range_min_bps = float(os.getenv("BAGS_DAILY_RANGE_MIN_BPS"))

    if os.getenv("BAGS_DAILY_RANGE_MAX_ACTIONS"):
        config.daily_range_max_actions_per_day = int(os.getenv("BAGS_DAILY_RANGE_MAX_ACTIONS"))

    if os.getenv("BAGS_DAILY_RANGE_USE_PREVIOUS"):
        value = os.getenv("BAGS_DAILY_RANGE_USE_PREVIOUS", "true")
        config.daily_range_use_previous_day = value.lower() not in ("0", "false", "no")

    return config
