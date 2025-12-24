import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import alpaca_wrapper
from loguru import logger

from src.trading_obj_utils import filter_to_realistic_positions
from src.stock_utils import pairs_equal


STATUS_VERSION = 1


def _parse_args():
    parser = argparse.ArgumentParser(description="Expiry watcher for stockagent3 positions")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--expiry-at", required=True)
    parser.add_argument("--strategy", default="stockagent3")
    parser.add_argument("--reason", default="plan_expiry")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def _load_status(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed reading expiry watcher status %s: %s", path, exc)
        return {}


def _write_status(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temp.replace(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed writing expiry watcher status %s: %s", path, exc)


def _update_status(path: Path, status: dict, **changes) -> dict:
    status.update(changes)
    status["last_update"] = datetime.now(timezone.utc).isoformat()
    _write_status(path, status)
    return status


def _position_open(symbol: str, side: str) -> bool:
    positions = filter_to_realistic_positions(alpaca_wrapper.get_all_positions())
    for pos in positions:
        if not hasattr(pos, "symbol"):
            continue
        if not pairs_equal(pos.symbol, symbol):
            continue
        pos_side = getattr(pos, "side", "").lower()
        if side == "buy" and pos_side == "long":
            return True
        if side == "sell" and pos_side == "short":
            return True
    return False


def _trigger_backout(symbol: str):
    try:
        from scripts.alpaca_cli import backout_near_market
    except Exception as exc:  # pragma: no cover
        logger.error("Unable to import backout_near_market: %s", exc)
        return False
    try:
        backout_near_market(symbol)
        return True
    except Exception as exc:  # pragma: no cover
        logger.error("backout_near_market failed for %s: %s", symbol, exc)
        return False


def main():
    args = _parse_args()
    symbol = args.symbol.upper()
    side = args.side.lower()

    try:
        expiry_at = datetime.fromisoformat(args.expiry_at.replace("Z", "+00:00"))
    except ValueError:
        logger.error("Invalid expiry_at: %s", args.expiry_at)
        return

    status = _load_status(args.config_path)
    status.update(
        {
            "config_version": STATUS_VERSION,
            "symbol": symbol,
            "side": side,
            "strategy": args.strategy,
            "expiry_at": expiry_at.astimezone(timezone.utc).isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "reason": args.reason,
            "pid": os.getpid(),
            "active": True,
            "state": "waiting",
        }
    )
    _write_status(args.config_path, status)

    poll_seconds = max(5, int(args.poll_seconds))

    try:
        while True:
            now = datetime.now(timezone.utc)
            if now >= expiry_at:
                break
            _update_status(args.config_path, status, state="waiting")
            time.sleep(min(poll_seconds, max(1, int((expiry_at - now).total_seconds()))))

        status = _update_status(args.config_path, status, state="expired")

        if _position_open(symbol, side):
            status = _update_status(args.config_path, status, state="triggering_backout")
            success = _trigger_backout(symbol)
            status = _update_status(
                args.config_path,
                status,
                state="backout_complete" if success else "backout_failed",
            )
        else:
            status = _update_status(args.config_path, status, state="no_position")
    except KeyboardInterrupt:
        _update_status(args.config_path, status, state="cancelled", active=False)
        raise
    finally:
        if status.get("active", False):
            _update_status(args.config_path, status, active=False)


if __name__ == "__main__":
    main()
