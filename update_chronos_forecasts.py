#!/usr/bin/env python3
"""
Update Chronos-2 forecasts for all symbols to latest data
"""
from pathlib import Path
from strategytrainingneural.collect_forecasts import main as collect_forecasts_main
import sys

# Override sys.argv to pass arguments programmatically
training_dir = Path("trainingdata/train")
symbol_files = list(training_dir.glob("*.csv"))
symbols = [f.stem for f in symbol_files]

print(f"Generating Chronos forecasts for {len(symbols)} symbols...")

# Build command line args
args = [
    "update_chronos_forecasts.py",  # program name
    "--data-dir", "trainingdata/train",
    "--cache-dir", "strategytraining/forecast_cache",
    "--context-length", "512",
    "--batch-size", "64",
    "--device-map", "cuda",
]

# Add all symbols
for symbol in symbols:
    args.extend(["--symbol", symbol])

sys.argv = args
collect_forecasts_main()
print("Chronos forecast generation complete!")
