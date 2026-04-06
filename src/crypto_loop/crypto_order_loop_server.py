""" fastapi server supporting

adding an order for a stock pair/side/price

polling untill the market is ready to accept the order then making a market order
cancelling all orders
cancelling an order
getting the current orders

"""
import csv
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Dict, Iterable, List, Literal, NotRequired, Optional, Tuple, TypedDict

from fastapi import FastAPI, Query
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse

from alpaca_wrapper import open_order_at_price_or_all
from jsonshelve import FlatShelf
from src.binan import binance_wrapper
from src.stock_utils import unmap_symbols
from src.trade_stock_utils import coerce_optional_float

data_dir = Path(__file__).parent.parent / 'data'

dynamic_config_ = data_dir / "dynamic_config"
dynamic_config_.mkdir(exist_ok=True, parents=True)

crypto_symbol_to_order = FlatShelf(str(dynamic_config_ / "crypto_symbol_to_order.db.json"))

symbols = [
    "BTCUSD",
    "ETHUSD",
    "LTCUSD",
    "UNIUSD",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORECAST_RESULTS_DIRS = (
    _REPO_ROOT / "strategy_state" / "results",
    _REPO_ROOT / "results",
)
_PREDICTION_FILENAMES = ("predictions.csv", "predictions-sim.csv")
_STRATEGY_FIELDS = {
    "entry_takeprofit": ("entry_takeprofit_profit", "entry_takeprofit_high_price", "entry_takeprofit_low_price"),
    "maxdiffprofit": ("maxdiffprofit_profit", "maxdiffprofit_high_price", "maxdiffprofit_low_price"),
    "takeprofit": ("takeprofit_profit", "takeprofit_high_price", "takeprofit_low_price"),
}
_forecast_cache_lock = Lock()
_order_book_lock = Lock()
_BINANCE_ALL_IN_MIRROR_MIN_QTY = 0.03
_BINANCE_ORDER_MIRROR_SYMBOLS = {
    "BTCUSD": "BTCUSDT",
}

OrderSide = Literal["buy", "sell"]
BinanceOrderSide = Literal["BUY", "SELL"]


class QueuedOrderPayload(TypedDict):
    symbol: str
    side: OrderSide
    price: float
    qty: float
    created_at: NotRequired[str]


class StockOrderLogPayload(TypedDict):
    symbol: str
    side: OrderSide
    price: float
    qty: float
    created_at: str
    mirrored_to_binance: bool
    mirror_symbol: Optional[str]


@dataclass(frozen=True)
class ForecastFileCacheEntry:
    source_file: str
    stat_key: Tuple[int, int]
    generated_at: Optional[str]
    records: List[Dict[str, object]]


_forecast_cache_entry: Optional[ForecastFileCacheEntry] = None


@dataclass(frozen=True)
class CryptoOrderBrokerApi:
    submit_price_order: Callable[[str, float, OrderSide, float], object]
    cancel_all_binance_orders: Callable[[], object]
    mirror_all_in_order: Callable[[str, BinanceOrderSide, float], object]


class ForecastSourceLoadError(RuntimeError):
    def __init__(self, message: str, source_file: Optional[str] = None):
        super().__init__(message)
        self.source_file = source_file


def _parse_symbols(raw_symbols: Optional[str]) -> Optional[List[str]]:
    if not raw_symbols:
        return None
    parsed_symbols: List[str] = []
    seen_symbols = set()
    for raw_symbol in raw_symbols.split(","):
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        parsed_symbols.append(symbol)
    return parsed_symbols


def _normalize_scalar(raw: object) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return coerce_optional_float(raw)
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
        if text.endswith(","):
            text = text[:-1].strip()
    return coerce_optional_float(text)


def _candidate_prediction_files(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    candidates = list(results_dir.glob("predictions-*.csv"))
    for name in _PREDICTION_FILENAMES:
        path = results_dir / name
        if path.exists():
            candidates.append(path)
    return candidates


def _find_latest_prediction_file() -> Optional[Path]:
    candidates: List[Path] = []
    for results_dir in _FORECAST_RESULTS_DIRS:
        candidates.extend(_candidate_prediction_files(results_dir))
    if not candidates:
        return None
    latest_candidate: Optional[Path] = None
    latest_mtime: Optional[float] = None
    for candidate in candidates:
        try:
            candidate_mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if latest_candidate is None or latest_mtime is None or candidate_mtime > latest_mtime:
            latest_candidate = candidate
            latest_mtime = candidate_mtime
    return latest_candidate


def _prediction_file_stat_key(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _extract_generated_at(rows: List[Dict[str, str]], source_file: Path) -> str:
    for row in rows:
        generated_at = row.get("generated_at")
        if generated_at:
            return str(generated_at)
    return datetime.fromtimestamp(source_file.stat().st_mtime, tz=timezone.utc).isoformat()


def _forecast_metadata_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_forecast_response_metadata(source_file: Optional[str], generated_at: Optional[str]) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "generated_at": generated_at,
        "source_file": source_file,
        "source_filename": None,
        "source_file_updated_at": None,
        "source_file_age_seconds": None,
    }
    if source_file is None:
        return metadata

    source_path = Path(source_file)
    metadata["source_filename"] = source_path.name
    try:
        source_stat = source_path.stat()
    except OSError:
        return metadata

    source_updated_at = datetime.fromtimestamp(source_stat.st_mtime, tz=timezone.utc)
    source_age_seconds = max(0, int((_forecast_metadata_now() - source_updated_at).total_seconds()))
    metadata["source_file_updated_at"] = source_updated_at.isoformat()
    metadata["source_file_age_seconds"] = source_age_seconds
    return metadata


def _forecast_search_paths() -> List[str]:
    return [str(path) for path in _FORECAST_RESULTS_DIRS]


def _build_missing_forecast_source_payload(
    payload_key: str,
    requested_symbols: Optional[List[str]],
    *,
    include_buy_list: bool = False,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        **_build_forecast_response_metadata(None, None),
        **_build_symbol_query_metadata(requested_symbols, ()),
        "count": 0,
        payload_key: [],
        "forecast_source_status": "missing",
        "error": "forecast_source_missing",
        "error_detail": "no prediction files found in configured results directories",
        "source_search_paths": _forecast_search_paths(),
        "source_search_filenames": list(_PREDICTION_FILENAMES),
        "next_steps": [
            f"write one of {', '.join(_PREDICTION_FILENAMES)} under a configured results directory",
            "rerun the forecast request after the predictions file is generated",
        ],
    }
    if include_buy_list:
        payload["buy_list"] = []
    return payload


def _build_symbol_query_metadata(
    requested_symbols: Optional[List[str]],
    returned_records: Iterable[Dict[str, object]],
) -> Dict[str, object]:
    if not requested_symbols:
        return {"symbol_query": {"applied": False}}

    returned_symbols = {
        str(record.get("symbol")).upper()
        for record in returned_records
        if record.get("symbol")
    }
    matched_symbols = [symbol for symbol in requested_symbols if symbol in returned_symbols]
    missing_symbols = [symbol for symbol in requested_symbols if symbol not in returned_symbols]
    return {
        "symbol_query": {
            "applied": True,
            "requested": list(requested_symbols),
            "matched": matched_symbols,
            "missing": missing_symbols,
        }
    }


def _build_forecast_source_error_payload(
    payload_key: str,
    requested_symbols: Optional[List[str]],
    error: ForecastSourceLoadError,
    *,
    include_buy_list: bool = False,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        **_build_forecast_response_metadata(error.source_file, None),
        **_build_symbol_query_metadata(requested_symbols, ()),
        "count": 0,
        payload_key: [],
        "forecast_source_status": "error",
        "error": "forecast_source_unavailable",
        "error_detail": str(error),
    }
    if include_buy_list:
        payload["buy_list"] = []
    return payload


def _build_ready_forecast_payload(
    requested_symbols: Optional[List[str]],
    source_file: str,
    generated_at: Optional[str],
    returned_records: Iterable[Dict[str, object]],
    payload_body: Dict[str, object],
) -> Dict[str, object]:
    return {
        **_build_forecast_response_metadata(source_file, generated_at),
        **_build_symbol_query_metadata(requested_symbols, returned_records),
        "forecast_source_status": "ready",
        **payload_body,
    }


def _serve_forecast_endpoint(
    endpoint: str,
    payload_key: str,
    requested_symbols: Optional[List[str]],
    payload_factory: Callable[[List[Dict[str, object]]], Dict[str, object]],
    *,
    include_buy_list: bool = False,
) -> object:
    try:
        records, source_file, generated_at = _load_forecast_records(requested_symbols)
    except ForecastSourceLoadError as exc:
        error_payload = _build_forecast_source_error_payload(
            payload_key,
            requested_symbols,
            exc,
            include_buy_list=include_buy_list,
        )
        _log_forecast_response(endpoint, error_payload, status_code=503)
        return JSONResponse(status_code=503, content=error_payload)
    if source_file is None:
        payload = _build_missing_forecast_source_payload(
            payload_key,
            requested_symbols,
            include_buy_list=include_buy_list,
        )
        _log_forecast_response(endpoint, payload, status_code=200)
        return payload
    payload = _build_ready_forecast_payload(
        requested_symbols,
        source_file,
        generated_at,
        records,
        payload_factory(records),
    )
    _log_forecast_response(endpoint, payload, status_code=200)
    return payload


def _log_forecast_response(endpoint: str, payload: Dict[str, object], *, status_code: int) -> None:
    event = {
        "endpoint": endpoint,
        "status_code": status_code,
        "forecast_source_status": payload.get("forecast_source_status", "ready"),
        "source_file": payload.get("source_file"),
        "source_filename": payload.get("source_filename"),
        "generated_at": payload.get("generated_at"),
        "source_file_age_seconds": payload.get("source_file_age_seconds"),
        "symbol_query": payload.get("symbol_query"),
        "count": payload.get("count"),
        "error": payload.get("error"),
        "error_detail": payload.get("error_detail"),
        "source_search_paths": payload.get("source_search_paths"),
    }
    if status_code >= 500:
        logger.warning("crypto_loop_forecast_response {}", event)
        return
    logger.info("crypto_loop_forecast_response {}", event)


def _log_stock_order(order_payload: StockOrderLogPayload) -> None:
    logger.info("crypto_loop_stock_order {}", order_payload)


def _load_crypto_order_broker_api() -> CryptoOrderBrokerApi:
    return CryptoOrderBrokerApi(
        submit_price_order=open_order_at_price_or_all,
        cancel_all_binance_orders=binance_wrapper.cancel_all_orders,
        mirror_all_in_order=binance_wrapper.create_all_in_order,
    )


def _binance_mirror_symbol_for(symbol: str) -> Optional[str]:
    return _BINANCE_ORDER_MIRROR_SYMBOLS.get(symbol)


def _binance_order_side_for(side: OrderSide) -> BinanceOrderSide:
    return "BUY" if side == "buy" else "SELL"


def _coerce_queued_order_payload(raw: object) -> Optional[QueuedOrderPayload]:
    if not isinstance(raw, dict):
        return None
    symbol = raw.get("symbol")
    side = raw.get("side")
    price = raw.get("price")
    qty = raw.get("qty")
    created_at = raw.get("created_at")
    if not isinstance(symbol, str) or not isinstance(side, str):
        return None
    if side not in ("buy", "sell"):
        return None
    normalized_price = _normalize_scalar(price)
    normalized_qty = _normalize_scalar(qty)
    if normalized_price is None or normalized_qty is None:
        return None
    payload: QueuedOrderPayload = {
        "symbol": symbol,
        "side": side,
        "price": normalized_price,
        "qty": normalized_qty,
    }
    if created_at is not None:
        if not isinstance(created_at, str):
            return None
        payload["created_at"] = created_at
    return payload


def _order_book_items_snapshot() -> List[Tuple[str, Optional[QueuedOrderPayload]]]:
    with _order_book_lock:
        if hasattr(crypto_symbol_to_order, "items"):
            raw_items = list(crypto_symbol_to_order.items())
        else:
            raw_items = [(str(key), crypto_symbol_to_order.get(key)) for key in list(crypto_symbol_to_order)]
    return [(str(symbol), _coerce_queued_order_payload(order)) for symbol, order in raw_items]


def _snapshot_order_book() -> Dict[str, QueuedOrderPayload]:
    return {
        str(symbol): order
        for symbol, order in _order_book_items_snapshot()
        if order is not None
    }


def _snapshot_order_symbols() -> List[str]:
    return [str(symbol) for symbol in _snapshot_order_book().keys()]


def _get_queued_order(symbol: str) -> Optional[QueuedOrderPayload]:
    with _order_book_lock:
        return _coerce_queued_order_payload(crypto_symbol_to_order.get(symbol))


def _set_queued_order(symbol: str, order_payload: QueuedOrderPayload) -> None:
    with _order_book_lock:
        crypto_symbol_to_order[symbol] = order_payload


def _clear_queued_order(symbol: str) -> None:
    with _order_book_lock:
        crypto_symbol_to_order[symbol] = None
        try:
            del crypto_symbol_to_order[symbol]
        except KeyError:
            pass


def _pop_queued_order(symbol: str) -> Optional[QueuedOrderPayload]:
    with _order_book_lock:
        order = _coerce_queued_order_payload(crypto_symbol_to_order.get(symbol))
        if order is None:
            try:
                del crypto_symbol_to_order[symbol]
            except KeyError:
                pass
            return None
        crypto_symbol_to_order[symbol] = None
        try:
            del crypto_symbol_to_order[symbol]
        except KeyError:
            pass
        return order


def _restore_queued_order_if_missing(symbol: str, order_payload: QueuedOrderPayload) -> bool:
    with _order_book_lock:
        if crypto_symbol_to_order.get(symbol) is not None:
            return False
        crypto_symbol_to_order[symbol] = order_payload
        return True


def _normalize_forecast_row(row: Dict[str, str]) -> Optional[Dict[str, object]]:
    raw_symbol = row.get("instrument") or row.get("symbol")
    if not raw_symbol:
        return None
    symbol = str(raw_symbol).strip().upper()
    record: Dict[str, object] = {"symbol": symbol}

    predicted_close = _normalize_scalar(row.get("close_predicted_price"))
    if predicted_close is None:
        predicted_close = _normalize_scalar(row.get("close_predicted_price_value"))
    predicted_high = _normalize_scalar(row.get("high_predicted_price"))
    if predicted_high is None:
        predicted_high = _normalize_scalar(row.get("high_predicted_price_value"))
    predicted_low = _normalize_scalar(row.get("low_predicted_price"))
    if predicted_low is None:
        predicted_low = _normalize_scalar(row.get("low_predicted_price_value"))

    last_close = _normalize_scalar(row.get("close_last_price"))

    predicted: Dict[str, float] = {}
    if predicted_close is not None:
        predicted["close"] = predicted_close
    if predicted_high is not None:
        predicted["high"] = predicted_high
    if predicted_low is not None:
        predicted["low"] = predicted_low
    if predicted:
        record["predicted"] = predicted

    last: Dict[str, float] = {}
    if last_close is not None:
        last["close"] = last_close
    if last:
        record["last"] = last

    strategies: Dict[str, Dict[str, float]] = {}
    for name, (profit_key, high_key, low_key) in _STRATEGY_FIELDS.items():
        profit = _normalize_scalar(row.get(profit_key))
        high_price = _normalize_scalar(row.get(high_key))
        low_price = _normalize_scalar(row.get(low_key))
        if profit is None and high_price is None and low_price is None:
            continue
        strategy_entry: Dict[str, float] = {}
        if profit is not None:
            strategy_entry["profit"] = profit
        if high_price is not None:
            strategy_entry["high_price"] = high_price
        if low_price is not None:
            strategy_entry["low_price"] = low_price
        if strategy_entry:
            strategies[name] = strategy_entry
    if strategies:
        record["strategy"] = strategies

    return record


def _load_forecast_records(symbols_filter: Optional[List[str]]) -> Tuple[List[Dict[str, object]], Optional[str], Optional[str]]:
    source_file = _find_latest_prediction_file()
    if source_file is None:
        return [], None, None
    source_file_str = str(source_file)
    try:
        stat_key = _prediction_file_stat_key(source_file)
    except OSError as exc:
        raise ForecastSourceLoadError(
            f"failed to stat forecast source: {exc}",
            source_file=source_file_str,
        ) from exc
    with _forecast_cache_lock:
        cached = _forecast_cache_entry
        if cached is not None and cached.source_file == source_file_str and cached.stat_key == stat_key:
            records = cached.records
            generated_at = cached.generated_at
        else:
            try:
                with source_file.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                generated_at = _extract_generated_at(rows, source_file)
                records = []
                for row in rows:
                    record = _normalize_forecast_row(row)
                    if record is not None:
                        records.append(record)
            except (OSError, csv.Error) as exc:
                raise ForecastSourceLoadError(
                    f"failed to load forecast source: {exc}",
                    source_file=source_file_str,
                ) from exc
            globals()["_forecast_cache_entry"] = ForecastFileCacheEntry(
                source_file=source_file_str,
                stat_key=stat_key,
                generated_at=generated_at,
                records=records,
            )

    if symbols_filter:
        records_by_symbol: Dict[str, List[Dict[str, object]]] = {}
        for record in records:
            symbol = str(record["symbol"]).upper()
            records_by_symbol.setdefault(symbol, []).append(record)
        filtered: List[Dict[str, object]] = []
        for symbol in symbols_filter:
            filtered.extend(records_by_symbol.get(symbol, ()))
        return filtered, source_file_str, generated_at
    return list(records), source_file_str, generated_at


def _build_recommendations(
    records: Iterable[Dict[str, object]],
    min_profit: float,
) -> Tuple[List[Dict[str, object]], List[str]]:
    recommendations: List[Dict[str, object]] = []
    buy_list: List[str] = []
    for record in records:
        symbol = str(record.get("symbol"))
        strategies = record.get("strategy", {})
        best_name = None
        best_profit = None
        best_payload = None
        if isinstance(strategies, dict):
            for name, payload in strategies.items():
                profit = None
                if isinstance(payload, dict):
                    profit = payload.get("profit")
                if profit is None:
                    continue
                if best_profit is None or profit > best_profit:
                    best_profit = profit
                    best_name = name
                    best_payload = payload

        predicted = record.get("predicted") if isinstance(record.get("predicted"), dict) else {}
        price_targets = {
            "low": None,
            "high": None,
            "close": None,
        }
        if isinstance(best_payload, dict):
            price_targets["low"] = best_payload.get("low_price")
            price_targets["high"] = best_payload.get("high_price")
        if isinstance(predicted, dict):
            price_targets["close"] = predicted.get("close")
            if price_targets["low"] is None:
                price_targets["low"] = predicted.get("low")
            if price_targets["high"] is None:
                price_targets["high"] = predicted.get("high")

        action = "HOLD"
        if best_profit is not None and best_profit > 0 and best_profit >= min_profit:
            action = "BUY"
            buy_list.append(symbol)

        recommendations.append(
            {
                "symbol": symbol,
                "recommendation": action,
                "strategy": best_name,
                "expected_profit": best_profit,
                "price_targets": price_targets,
            }
        )
    return recommendations, buy_list


def _process_queued_order(symbol: str, broker_api: CryptoOrderBrokerApi) -> None:
    order = _pop_queued_order(symbol)
    if order is None:
        return

    logger.info(f"order {order}")
    side = order["side"]
    try:
        if side == "buy":
            logger.info(f"buying {symbol} at {order['price']}")
            broker_api.submit_price_order(symbol, order["qty"], "buy", order["price"])
            return
        if side == "sell":
            logger.info(f"selling {symbol} at {order['price']}")
            broker_api.submit_price_order(symbol, order["qty"], "sell", order["price"])
            return
        logger.error(f"unknown side {side}")
        logger.error(f"order {order}")
    except Exception as exc:
        restored = _restore_queued_order_if_missing(symbol, order)
        logger.error(f"failed to submit {symbol} order: {exc}")
        logger.error(f"requeued_failed_order={restored} order={order}")


def crypto_order_loop():
    broker_api = _load_crypto_order_broker_api()
    try:
        while not _thread_stop_event.is_set():
            try:
                # get all data for symbols
                for symbol in symbols:
                    _process_queued_order(symbol, broker_api)
            except Exception as e:
                logger.error(e)
            _thread_stop_event.wait(10)
    finally:
        _thread_stopped_event.set()


_thread_lock = Lock()
_thread_stop_event = Event()
_thread_stopped_event = Event()
_thread_stopped_event.set()
thread_loop: Optional[Thread] = None


def ensure_crypto_order_loop_started() -> Optional[Thread]:
    global thread_loop
    with _thread_lock:
        if thread_loop is not None and thread_loop.is_alive():
            if _thread_stop_event.is_set() and not _thread_stopped_event.is_set():
                _thread_stop_event.clear()
            return thread_loop
        _thread_stop_event.clear()
        _thread_stopped_event.clear()
        thread_loop = Thread(target=crypto_order_loop, daemon=True, name="crypto-order-loop")
        thread_loop.start()
        return thread_loop


def stop_crypto_order_loop(timeout: float = 1.0) -> None:
    global thread_loop
    with _thread_lock:
        running_thread = thread_loop
        if running_thread is None:
            return
        _thread_stop_event.set()
    running_thread.join(timeout=timeout)
    with _thread_lock:
        if thread_loop is running_thread and not running_thread.is_alive():
            thread_loop = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    ensure_crypto_order_loop_started()
    try:
        yield
    finally:
        stop_crypto_order_loop()


app = FastAPI(lifespan=_lifespan)


class OrderRequest(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    price: float = Field(gt=0.0)
    qty: float = Field(gt=0.0)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        normalized = unmap_symbols(str(value).strip().upper())
        if normalized not in symbols:
            raise ValueError(f"unsupported symbol: {normalized}")
        return normalized

    @field_validator("side", mode="before")
    @classmethod
    def normalize_side(cls, value: object) -> object:
        if value is None:
            return value
        return str(value).strip().lower()

    @field_validator("price", "qty")
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


@app.post("/api/v1/stock_order")
def stock_order(order: OrderRequest):
    broker_api = _load_crypto_order_broker_api()
    symbol = order.symbol
    queued_order: QueuedOrderPayload = {
        "symbol": symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _set_queued_order(symbol, queued_order)
    mirror_symbol = _binance_mirror_symbol_for(symbol)
    mirrored_to_binance = False
    if mirror_symbol is not None and order.qty > _BINANCE_ALL_IN_MIRROR_MIN_QTY:
        broker_api.cancel_all_binance_orders()  # why cancel all crypto?
        broker_api.mirror_all_in_order(mirror_symbol, _binance_order_side_for(order.side), order.price)
        mirrored_to_binance = True

    log_payload: StockOrderLogPayload = {
        "symbol": symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "created_at": queued_order["created_at"],
        "mirrored_to_binance": mirrored_to_binance,
        "mirror_symbol": mirror_symbol if mirrored_to_binance else None,
    }
    _log_stock_order(log_payload)


@app.get("/api/v1/stock_orders")
def stock_orders():
    return JSONResponse(_snapshot_order_book())


@app.get("/api/v1/stock_order/{symbol}")
def get_stock_order(symbol: str):
    symbol = unmap_symbols(symbol)
    return JSONResponse(_get_queued_order(symbol))


@app.delete("/api/v1/stock_order/{symbol}")
def delete_stock_order(symbol: str):
    symbol = unmap_symbols(symbol)
    _clear_queued_order(symbol)


@app.get("/api/v1/stock_order/cancel_all")
def delete_stock_orders():
    for symbol in _snapshot_order_symbols():
        _clear_queued_order(symbol)


@app.get("/api/v1/forecasts/latest")
def forecasts_latest(
    symbols: Optional[str] = Query(default=None, description="Comma-separated list of symbols to include"),
):
    symbol_filter = _parse_symbols(symbols)
    return _serve_forecast_endpoint(
        "/api/v1/forecasts/latest",
        "forecasts",
        symbol_filter,
        lambda records: {
            "count": len(records),
            "forecasts": records,
        },
    )


@app.get("/api/v1/forecasts/prices")
def forecasts_prices(
    symbols: Optional[str] = Query(default=None, description="Comma-separated list of symbols to include"),
):
    symbol_filter = _parse_symbols(symbols)
    def build_prices_payload(records: List[Dict[str, object]]) -> Dict[str, object]:
        prices: List[Dict[str, object]] = []
        for record in records:
            entry = {"symbol": record.get("symbol")}
            if "last" in record:
                entry["last"] = record["last"]
            if "predicted" in record:
                entry["predicted"] = record["predicted"]
            prices.append(entry)
        return {
            "count": len(prices),
            "prices": prices,
        }

    return _serve_forecast_endpoint(
        "/api/v1/forecasts/prices",
        "prices",
        symbol_filter,
        build_prices_payload,
    )


@app.get("/api/v1/bot/forecasts")
def bot_forecasts(
    symbols: Optional[str] = Query(default=None, description="Comma-separated list of symbols to include"),
    min_profit: float = Query(default=0.0, description="Minimum profit threshold for BUY recommendations"),
):
    symbol_filter = _parse_symbols(symbols)
    def build_recommendation_payload(records: List[Dict[str, object]]) -> Dict[str, object]:
        forecasts, buy_list = _build_recommendations(records, min_profit)
        return {
            "count": len(forecasts),
            "buy_list": buy_list,
            "forecasts": forecasts,
        }

    return _serve_forecast_endpoint(
        "/api/v1/bot/forecasts",
        "forecasts",
        symbol_filter,
        build_recommendation_payload,
        include_buy_list=True,
    )
