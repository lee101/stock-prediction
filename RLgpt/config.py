from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DailyPlanDataConfig:
    symbols: tuple[str, ...]
    data_root: Path = Path("trainingdatahourly") / "crypto"
    forecast_cache_root: Path = Path("binanceneural") / "forecast_cache"
    forecast_horizons: tuple[int, ...] = (1, 24)
    sequence_length: int = 72
    min_history_hours: int = 24 * 45
    max_feature_lookback_hours: int = 24 * 7
    min_bars_per_day: int = 1
    validation_days: int = 30
    cache_only: bool = True


@dataclass(frozen=True)
class PlannerConfig:
    hidden_dim: int = 128
    depth: int = 3
    heads: int = 4
    dropout: float = 0.1
    max_center_offset_bps: float = 250.0
    min_half_spread_bps: float = 5.0
    max_half_spread_bps: float = 300.0


@dataclass(frozen=True)
class SimulatorConfig:
    initial_cash: float = 100_000.0
    shared_unit_budget: float = 20.0
    max_units_per_asset: float = 10.0
    maker_fee_bps: float = 10.0
    slippage_bps: float = 5.0
    fill_buffer_bps: float = 5.0
    fill_temperature_bps: float = 8.0
    carry_inventory: bool = False
    min_trade_units: float = 0.0
    raw_pnl_weight: float = 1.0
    smooth_pnl_weight: float = 0.05
    downside_penalty: float = 0.25
    turnover_penalty: float = 1e-5
    inventory_penalty: float = 1e-3


@dataclass(frozen=True)
class TrainingConfig:
    data: DailyPlanDataConfig
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 1337
    device: str | None = None
    num_workers: int = 0
    max_train_days: int | None = None
    max_val_days: int | None = None
    run_name: str | None = None
    output_root: Path = Path("experiments") / "RLgpt"
