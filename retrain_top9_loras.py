#!/usr/bin/env python3
"""Retrain Chronos2 LoRA models for top9 trading symbols."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_all_stock_loras_inprocess import main as sweep_main
import scripts.train_all_stock_loras_inprocess as mod

mod.SYMBOLS = [
    "NVDA", "PLTR", "GOOG", "NET", "DBX", "TRIP", "EBAY", "MTCH", "NYT",
]
mod.CONFIGS = [
    {"ctx": 128, "lr": "5e-5", "steps": 1000},
]
mod.RESULTS_FILE = Path("hyperparams/top9_lora_sweep_results.json")

if __name__ == "__main__":
    sweep_main()
