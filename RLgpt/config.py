from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RLGPT_DATA_ROOT = Path("trainingdatahourly") / "crypto"
DEFAULT_RLGPT_FORECAST_CACHE_ROOT = Path("binanceneural") / "forecast_cache"
DEFAULT_RLGPT_FORECAST_HORIZONS = (1, 24)
DEFAULT_RLGPT_SEQUENCE_LENGTH = 72
DEFAULT_RLGPT_MIN_HISTORY_HOURS = 24 * 45
DEFAULT_RLGPT_MAX_FEATURE_LOOKBACK_HOURS = 24 * 7
DEFAULT_RLGPT_MIN_BARS_PER_DAY = 1
DEFAULT_RLGPT_VALIDATION_DAYS = 30
DEFAULT_RLGPT_CACHE_ONLY = True

DEFAULT_RLGPT_HIDDEN_DIM = 128
DEFAULT_RLGPT_DEPTH = 3
DEFAULT_RLGPT_HEADS = 4
DEFAULT_RLGPT_DROPOUT = 0.1
DEFAULT_RLGPT_MAX_CENTER_OFFSET_BPS = 250.0
DEFAULT_RLGPT_MIN_HALF_SPREAD_BPS = 5.0
DEFAULT_RLGPT_MAX_HALF_SPREAD_BPS = 300.0

DEFAULT_RLGPT_INITIAL_CASH = 100_000.0
DEFAULT_RLGPT_SHARED_UNIT_BUDGET = 20.0
DEFAULT_RLGPT_MAX_UNITS_PER_ASSET = 10.0
DEFAULT_RLGPT_MAKER_FEE_BPS = 10.0
DEFAULT_RLGPT_SLIPPAGE_BPS = 5.0
DEFAULT_RLGPT_FILL_BUFFER_BPS = 5.0
DEFAULT_RLGPT_FILL_TEMPERATURE_BPS = 8.0
DEFAULT_RLGPT_CARRY_INVENTORY = False
DEFAULT_RLGPT_MIN_TRADE_UNITS = 0.0
DEFAULT_RLGPT_RAW_PNL_WEIGHT = 1.0
DEFAULT_RLGPT_SMOOTH_PNL_WEIGHT = 0.05
DEFAULT_RLGPT_DOWNSIDE_PENALTY = 0.25
DEFAULT_RLGPT_TURNOVER_PENALTY = 1e-5
DEFAULT_RLGPT_INVENTORY_PENALTY = 1e-3

DEFAULT_RLGPT_EPOCHS = 25
DEFAULT_RLGPT_BATCH_SIZE = 16
DEFAULT_RLGPT_LEARNING_RATE = 3e-4
DEFAULT_RLGPT_WEIGHT_DECAY = 1e-4
DEFAULT_RLGPT_GRAD_CLIP = 1.0
DEFAULT_RLGPT_SEED = 1337
DEFAULT_RLGPT_DEVICE: str | None = None
DEFAULT_RLGPT_NUM_WORKERS = 0
DEFAULT_RLGPT_OUTPUT_ROOT = Path("experiments") / "RLgpt"


@dataclass(frozen=True)
class DailyPlanDataConfig:
    symbols: tuple[str, ...]
    data_root: Path = DEFAULT_RLGPT_DATA_ROOT
    forecast_cache_root: Path = DEFAULT_RLGPT_FORECAST_CACHE_ROOT
    forecast_horizons: tuple[int, ...] = DEFAULT_RLGPT_FORECAST_HORIZONS
    sequence_length: int = DEFAULT_RLGPT_SEQUENCE_LENGTH
    min_history_hours: int = DEFAULT_RLGPT_MIN_HISTORY_HOURS
    max_feature_lookback_hours: int = DEFAULT_RLGPT_MAX_FEATURE_LOOKBACK_HOURS
    min_bars_per_day: int = DEFAULT_RLGPT_MIN_BARS_PER_DAY
    validation_days: int = DEFAULT_RLGPT_VALIDATION_DAYS
    cache_only: bool = DEFAULT_RLGPT_CACHE_ONLY


@dataclass(frozen=True)
class PlannerConfig:
    hidden_dim: int = DEFAULT_RLGPT_HIDDEN_DIM
    depth: int = DEFAULT_RLGPT_DEPTH
    heads: int = DEFAULT_RLGPT_HEADS
    dropout: float = DEFAULT_RLGPT_DROPOUT
    max_center_offset_bps: float = DEFAULT_RLGPT_MAX_CENTER_OFFSET_BPS
    min_half_spread_bps: float = DEFAULT_RLGPT_MIN_HALF_SPREAD_BPS
    max_half_spread_bps: float = DEFAULT_RLGPT_MAX_HALF_SPREAD_BPS


@dataclass(frozen=True)
class SimulatorConfig:
    initial_cash: float = DEFAULT_RLGPT_INITIAL_CASH
    shared_unit_budget: float = DEFAULT_RLGPT_SHARED_UNIT_BUDGET
    max_units_per_asset: float = DEFAULT_RLGPT_MAX_UNITS_PER_ASSET
    maker_fee_bps: float = DEFAULT_RLGPT_MAKER_FEE_BPS
    slippage_bps: float = DEFAULT_RLGPT_SLIPPAGE_BPS
    fill_buffer_bps: float = DEFAULT_RLGPT_FILL_BUFFER_BPS
    fill_temperature_bps: float = DEFAULT_RLGPT_FILL_TEMPERATURE_BPS
    carry_inventory: bool = DEFAULT_RLGPT_CARRY_INVENTORY
    min_trade_units: float = DEFAULT_RLGPT_MIN_TRADE_UNITS
    raw_pnl_weight: float = DEFAULT_RLGPT_RAW_PNL_WEIGHT
    smooth_pnl_weight: float = DEFAULT_RLGPT_SMOOTH_PNL_WEIGHT
    downside_penalty: float = DEFAULT_RLGPT_DOWNSIDE_PENALTY
    turnover_penalty: float = DEFAULT_RLGPT_TURNOVER_PENALTY
    inventory_penalty: float = DEFAULT_RLGPT_INVENTORY_PENALTY


@dataclass(frozen=True)
class TrainingConfig:
    data: DailyPlanDataConfig
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    epochs: int = DEFAULT_RLGPT_EPOCHS
    batch_size: int = DEFAULT_RLGPT_BATCH_SIZE
    learning_rate: float = DEFAULT_RLGPT_LEARNING_RATE
    weight_decay: float = DEFAULT_RLGPT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_RLGPT_GRAD_CLIP
    seed: int = DEFAULT_RLGPT_SEED
    device: str | None = DEFAULT_RLGPT_DEVICE
    num_workers: int = DEFAULT_RLGPT_NUM_WORKERS
    max_train_days: int | None = None
    max_val_days: int | None = None
    run_name: str | None = None
    output_root: Path = DEFAULT_RLGPT_OUTPUT_ROOT


def default_forecast_horizons_csv() -> str:
    return ",".join(str(value) for value in DEFAULT_RLGPT_FORECAST_HORIZONS)


def normalize_symbol_list(symbols: Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        token = str(symbol or "").strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
    return tuple(cleaned)


def parse_horizon_list(values: str | Iterable[int | str]) -> tuple[int, ...]:
    if isinstance(values, str):
        raw_values = (token.strip() for token in values.split(","))
    else:
        raw_values = (str(value).strip() for value in values)

    parsed: list[int] = []
    seen: set[int] = set()
    for token in raw_values:
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        seen.add(value)
        parsed.append(value)
    return tuple(parsed)


def validate_training_config(config: TrainingConfig) -> list[str]:
    errors: list[str] = []

    if not normalize_symbol_list(config.data.symbols):
        errors.append("At least one symbol is required.")
    if not config.data.forecast_horizons:
        errors.append("At least one forecast horizon is required.")
    elif any(int(horizon) <= 0 for horizon in config.data.forecast_horizons):
        errors.append("Forecast horizons must be positive integers.")

    if int(config.data.validation_days) <= 0:
        errors.append("validation_days must be >= 1.")
    if int(config.data.sequence_length) <= 0:
        errors.append("sequence_length must be >= 1.")
    if int(config.data.min_history_hours) <= 0:
        errors.append("min_history_hours must be >= 1.")
    if int(config.data.max_feature_lookback_hours) <= 0:
        errors.append("max_feature_lookback_hours must be >= 1.")
    if int(config.data.min_bars_per_day) <= 0:
        errors.append("min_bars_per_day must be >= 1.")

    if int(config.planner.hidden_dim) <= 0:
        errors.append("planner.hidden_dim must be >= 1.")
    if int(config.planner.depth) <= 0:
        errors.append("planner.depth must be >= 1.")
    if int(config.planner.heads) <= 0:
        errors.append("planner.heads must be >= 1.")
    if not 0.0 <= float(config.planner.dropout) < 1.0:
        errors.append("planner.dropout must be in [0.0, 1.0).")

    if float(config.simulator.initial_cash) <= 0.0:
        errors.append("simulator.initial_cash must be > 0.")
    if float(config.simulator.shared_unit_budget) <= 0.0:
        errors.append("simulator.shared_unit_budget must be > 0.")
    if float(config.simulator.max_units_per_asset) <= 0.0:
        errors.append("simulator.max_units_per_asset must be > 0.")

    if int(config.epochs) <= 0:
        errors.append("epochs must be >= 1.")
    if int(config.batch_size) <= 0:
        errors.append("batch_size must be >= 1.")
    if float(config.learning_rate) <= 0.0:
        errors.append("learning_rate must be > 0.")
    if float(config.weight_decay) < 0.0:
        errors.append("weight_decay must be >= 0.")
    if float(config.grad_clip) <= 0.0:
        errors.append("grad_clip must be > 0.")
    if int(config.num_workers) < 0:
        errors.append("num_workers must be >= 0.")

    if config.max_train_days is not None and int(config.max_train_days) <= 0:
        errors.append("max_train_days must be >= 1 when provided.")
    if config.max_val_days is not None and int(config.max_val_days) <= 0:
        errors.append("max_val_days must be >= 1 when provided.")

    return errors
