#!/usr/bin/env python3
"""One-shot: cancel all market orders, place limit exit orders for all positions."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

import torch

LONG_ONLY = {"NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "DBX", "TSLA", "AAPL"}
SHORT_ONLY = {"YELP", "EBAY", "TRIP", "MTCH", "KIND", "ANGI", "Z", "EXPE", "BKNG", "NWSA", "NYT"}
STATE_FILE = Path("strategy_state/stock_portfolio_state.json")

api = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)

# Step 1: Show positions
positions = {}
for pos in api.get_all_positions():
    positions[pos.symbol] = {
        "qty": float(pos.qty),
        "side": pos.side.value,
        "avg_entry": float(pos.avg_entry_price),
        "market_value": float(pos.market_value),
        "unrealized_pl": float(pos.unrealized_pl),
    }

print(f"\n{'Symbol':<8} {'Qty':>8} {'Side':<6} {'Entry':>10} {'MktVal':>10} {'P&L':>10}")
print("-" * 60)
for sym, p in sorted(positions.items()):
    print(f"{sym:<8} {p['qty']:>8.2f} {p['side']:<6} ${p['avg_entry']:>9.2f} ${p['market_value']:>9.2f} ${p['unrealized_pl']:>9.2f}")

# Step 2: Cancel ALL existing orders
print("\nCancelling all existing orders...")
api.cancel_orders()
print("All orders cancelled.")

# Step 3: Load model
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, policy_config_from_payload
from binanceneural.inference import generate_latest_action
from src.torch_load_utils import torch_load_compat

ckpt_dir = Path("unified_hourly_experiment/checkpoints/portfolio_deploy")
meta = json.loads((ckpt_dir / "training_meta.json").read_text())
ckpt = torch_load_compat(ckpt_dir / "epoch_007.pt", map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

feature_columns = meta["feature_columns"]
seq_len = meta["sequence_length"]
horizons = [1] if not any("h24" in c for c in feature_columns) else [1, 24]

policy_cfg = policy_config_from_payload(meta, input_dim=len(feature_columns), state_dict=state_dict)
model = build_policy(policy_cfg)
model.load_state_dict(state_dict, strict=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"\nModel loaded: {len(feature_columns)} features, seq={seq_len}")

# Step 4: Generate exit orders for ALL positions
now = datetime.now(timezone.utc)
new_state = {"positions": {}, "pending_close": []}

for symbol, pos_info in sorted(positions.items()):
    qty = pos_info["qty"]
    if abs(qty) < 1:
        continue

    # All positions are long. For all of them, we want a sell limit exit.
    # For SHORT_ONLY stocks held long, we sell at the model's sell_price (exit at profit if possible).
    try:
        ds_cfg = DatasetConfig(
            symbol=symbol,
            data_root="trainingdatahourly/stocks",
            forecast_cache_root="unified_hourly_experiment/forecast_cache",
            forecast_horizons=horizons,
            sequence_length=seq_len,
            min_history_hours=100,
            validation_days=30,
            cache_only=True,
        )
        dm = BinanceHourlyDataModule(ds_cfg)
        frame = dm.frame.copy()
        frame["symbol"] = symbol

        action = generate_latest_action(
            model=model, frame=frame, feature_columns=feature_columns,
            normalizer=dm.normalizer, sequence_length=seq_len, horizon=1, device=device,
        )

        buy_price = action.get("buy_price", 0)
        sell_price = action.get("sell_price", 0)
        current_price = float(frame["close"].iloc[-1])

        # For long positions: exit at sell_price
        exit_price = sell_price
        exit_side = OrderSide.SELL

        print(f"{symbol}: cur=${current_price:.2f} entry=${pos_info['avg_entry']:.2f} exit=${exit_price:.2f}", end="")

        if exit_price > 0 and int(abs(qty)) >= 1:
            order = LimitOrderRequest(
                symbol=symbol,
                qty=int(abs(qty)),
                side=exit_side,
                limit_price=round(exit_price, 2),
                time_in_force=TimeInForce.GTC,
            )
            result = api.submit_order(order)
            short_note = " [SHORT_ONLY wrong-side]" if symbol in SHORT_ONLY else ""
            print(f" -> GTC sell {int(abs(qty))} @ ${exit_price:.2f}{short_note} (id={result.id})")

            new_state["positions"][symbol] = {
                "qty": qty,
                "entry_time": now.isoformat(),
                "exit_price": exit_price,
                "exit_order_id": str(result.id),
                "hold_hours": 6,
            }
        else:
            print(f" -> no valid exit price")
            new_state["positions"][symbol] = {
                "qty": qty,
                "entry_time": now.isoformat(),
                "exit_price": None,
                "exit_order_id": None,
                "hold_hours": 6,
            }

    except Exception as e:
        print(f"{symbol}: signal failed - {e}")
        new_state["positions"][symbol] = {
            "qty": qty,
            "entry_time": now.isoformat(),
            "exit_price": None,
            "exit_order_id": None,
            "hold_hours": 6,
        }

# Save state
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
STATE_FILE.write_text(json.dumps(new_state, indent=2, default=str))
print(f"\nState saved: {len(new_state['positions'])} tracked, {len(new_state['pending_close'])} pending_close")
print("Done!")
