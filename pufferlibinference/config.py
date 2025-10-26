from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

from src.leverage_settings import get_leverage_settings
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import used for type checking only
    from hftraining.toto_features import TotoOptions


PathLike = Union[str, Path]


@dataclass(slots=True)
class InferenceDataConfig:
    """
    Declarative configuration describing how market data should be loaded and
    transformed prior to feeding the PufferLib allocator.

    Attributes:
        symbols: Ordered collection of tickers that the allocator expects.
        data_dir: Root directory containing per-symbol CSV files.
        start_date: Optional inclusive start date (UTC) used to trim history.
        end_date: Optional inclusive end date (UTC) used to trim history.
        resample_rule: Optional pandas offset alias (e.g. '1D', '30T').
        fill_method: Missing data reconciliation strategy ('ffill', 'bfill', or None).
        use_toto_forecasts: Toggle Toto feature generation to mirror training.
        toto_options: Optional Toto inference options (context length, horizon, etc.).
        toto_prediction_frames: Pre-computed Toto predictions keyed by ticker symbol.
        max_rows: Optional hard cap on the number of rows loaded per asset.
        enforce_common_index: If True, restrict all assets to a shared timestamp index.
    """

    symbols: Sequence[str]
    data_dir: Path = Path("trainingdata")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    resample_rule: Optional[str] = None
    fill_method: Optional[str] = "ffill"
    use_toto_forecasts: bool = False
    toto_options: Optional["TotoOptions"] = None
    toto_prediction_frames: Optional[Dict[str, "object"]] = None
    max_rows: Optional[int] = None
    enforce_common_index: bool = True

    def normalised_symbols(self) -> Sequence[str]:
        return [sym.upper() for sym in self.symbols]

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()


@dataclass(slots=True)
class PufferInferenceConfig:
    """
    Parameters controlling how a saved Portfolio RL checkpoint is loaded and
    how allocations translate into portfolio metrics.

    Attributes:
        checkpoint_path: Torch checkpoint produced by ``pufferlibtraining``.
        processor_path: Joblib file with the fitted ``StockDataProcessor`` scalers.
        device: 'auto', 'cpu', or 'cuda' override.
        leverage_limit: Maximum absolute exposure enforced during inference.
        transaction_cost_bps: Linear transaction fees in basis points per unit turnover.
        borrowing_cost: Annualised financing cost applied to exposure above 1×.
        trading_days_per_year: Calendar used to annualise borrowing cost.
        allow_short: Whether negative allocations are permitted.
        enforce_leverage_limit: If True, allocations are re-scaled to stay within leverage cap.
        clamp_action: Optional per-asset clamp applied pre-normalisation (e.g., 1.5).
        min_cash_buffer: Optional residual cash allocation to preserve (0.0-1.0).
    """

    checkpoint_path: PathLike
    processor_path: Optional[PathLike] = None
    device: str = "auto"
    leverage_limit: float = field(default_factory=lambda: get_leverage_settings().max_gross_leverage)
    transaction_cost_bps: float = 10.0
    borrowing_cost: float = field(default_factory=lambda: get_leverage_settings().annual_cost)
    trading_days_per_year: int = field(default_factory=lambda: get_leverage_settings().trading_days_per_year)
    allow_short: bool = True
    enforce_leverage_limit: bool = True
    clamp_action: Optional[float] = None
    min_cash_buffer: float = 0.0

    def resolved_checkpoint(self) -> Path:
        return Path(self.checkpoint_path).expanduser().resolve()

    def resolved_processor(self) -> Optional[Path]:
        if self.processor_path is None:
            return None
        return Path(self.processor_path).expanduser().resolve()

    @property
    def transaction_cost_decimal(self) -> float:
        return self.transaction_cost_bps / 1e4

