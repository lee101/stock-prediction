""" fastapi server supporting

adding an order for a stock pair/side/price

polling untill the market is ready to accept the order then making a market order
cancelling all orders
cancelling an order
getting the current orders

"""
import csv
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, Iterable, List, Optional, Set, Tuple

from fastapi import FastAPI, Query
from loguru import logger
from pydantic import BaseModel
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

app = FastAPI()

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


def _parse_symbols(raw_symbols: Optional[str]) -> Optional[Set[str]]:
    if not raw_symbols:
        return None
    return {symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()}


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
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_generated_at(rows: List[Dict[str, str]], source_file: Path) -> str:
    for row in rows:
        generated_at = row.get("generated_at")
        if generated_at:
            return str(generated_at)
    return datetime.fromtimestamp(source_file.stat().st_mtime).isoformat()


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


def _load_forecast_records(symbols_filter: Optional[Set[str]]) -> Tuple[List[Dict[str, object]], Optional[str], Optional[str]]:
    source_file = _find_latest_prediction_file()
    if source_file is None:
        return [], None, None
    with source_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    generated_at = _extract_generated_at(rows, source_file)

    records: List[Dict[str, object]] = []
    for row in rows:
        record = _normalize_forecast_row(row)
        if record is None:
            continue
        symbol = record["symbol"]
        if symbols_filter and symbol not in symbols_filter:
            continue
        records.append(record)

    return records, str(source_file), generated_at


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


def crypto_order_loop():
    while True:
        try:
            # get all data for symbols
            for symbol in symbols:
                # very_latest_data = latest_data(symbol)
                order = crypto_symbol_to_order.get(symbol)
                if order:
                    logger.info(f"order {order}")
                    if order['side'] == "buy":
                        # if float(very_latest_data.ask_price) < order['price']:
                        logger.info(f"buying {symbol} at {order['price']}")
                        crypto_symbol_to_order[symbol] = None
                        del crypto_symbol_to_order[symbol]
                        open_order_at_price_or_all(symbol, order['qty'], "buy", order['price'])
                    elif order['side'] == "sell":
                        # if float(very_latest_data.bid_price) > order['price']:
                        logger.info(f"selling {symbol} at {order['price']}")
                        crypto_symbol_to_order[symbol] = None
                        del crypto_symbol_to_order[symbol]
                        open_order_at_price_or_all(symbol, order['qty'], "sell", order['price'])
                    else:
                        logger.error(f"unknown side {order['side']}")
                        logger.error(f"order {order}")
        except Exception as e:
            logger.error(e)
        time.sleep(10)


thread_loop = Thread(target=crypto_order_loop, daemon=True)
thread_loop.start()


class OrderRequest(BaseModel):
    symbol: str
    side: str
    price: float
    qty: float


@app.post("/api/v1/stock_order")
def stock_order(order: OrderRequest):
    symbol = unmap_symbols(order.symbol)
    crypto_symbol_to_order[symbol] = {
        "symbol": symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "created_at": datetime.now().isoformat(),
    }
    # convert to USDT - assume crypto
    usdt_symbol = symbol[:3] + "USDT"
    # order all on binance
    if order.qty > 0.03 and symbol == "BTCUSD":  # going all in on a bitcoin side
        binance_wrapper.cancel_all_orders()  # why cancel all crypto?
        # replicate order to binance account for free trading on btc
        binance_wrapper.create_all_in_order(usdt_symbol, order.side.upper(), order.price)


@app.get("/api/v1/stock_orders")
def stock_orders():
    return JSONResponse(crypto_symbol_to_order.__dict__)


@app.get("/api/v1/stock_order/{symbol}")
def get_stock_order(symbol: str):
    symbol = unmap_symbols(symbol)
    return JSONResponse(crypto_symbol_to_order.get(symbol))


@app.delete("/api/v1/stock_order/{symbol}")
def delete_stock_order(symbol: str):
    symbol = unmap_symbols(symbol)
    crypto_symbol_to_order[symbol] = None


@app.get("/api/v1/stock_order/cancel_all")
def delete_stock_orders():
    for symbol in crypto_symbol_to_order:
        crypto_symbol_to_order[symbol] = None


@app.get("/api/v1/forecasts/latest")
def forecasts_latest(
    symbols: Optional[str] = Query(default=None, description="Comma-separated list of symbols to include"),
):
    symbol_filter = _parse_symbols(symbols)
    records, source_file, generated_at = _load_forecast_records(symbol_filter)
    return {
        "generated_at": generated_at,
        "source_file": source_file,
        "count": len(records),
        "forecasts": records,
    }


@app.get("/api/v1/forecasts/prices")
def forecasts_prices(
    symbols: Optional[str] = Query(default=None, description="Comma-separated list of symbols to include"),
):
    symbol_filter = _parse_symbols(symbols)
    records, source_file, generated_at = _load_forecast_records(symbol_filter)
    prices: List[Dict[str, object]] = []
    for record in records:
        entry = {"symbol": record.get("symbol")}
        if "last" in record:
            entry["last"] = record["last"]
        if "predicted" in record:
            entry["predicted"] = record["predicted"]
        prices.append(entry)
    return {
        "generated_at": generated_at,
        "source_file": source_file,
        "count": len(prices),
        "prices": prices,
    }


@app.get("/api/v1/bot/forecasts")
def bot_forecasts(
    symbols: Optional[str] = Query(default=None, description="Comma-separated list of symbols to include"),
    min_profit: float = Query(default=0.0, description="Minimum profit threshold for BUY recommendations"),
):
    symbol_filter = _parse_symbols(symbols)
    records, source_file, generated_at = _load_forecast_records(symbol_filter)
    forecasts, buy_list = _build_recommendations(records, min_profit)
    return {
        "generated_at": generated_at,
        "source_file": source_file,
        "count": len(forecasts),
        "buy_list": buy_list,
        "forecasts": forecasts,
    }
