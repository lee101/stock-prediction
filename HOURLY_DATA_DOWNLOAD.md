# Hourly Data Download from Alpaca

This guide explains how to download hourly stock and crypto data from Alpaca.

## Overview

The `download_hourly_data.py` script downloads hourly bars for stocks and crypto from Alpaca:
- **Stocks**: Up to 7 years of hourly data (~9,000-10,000 bars per symbol)
- **Crypto**: Up to 10 years of hourly data (~40,000-42,000 bars per symbol)

## Quick Start

```bash
# Activate the virtual environment
source .venv313/bin/activate

# Download all symbols (278 stocks + 17 crypto = 295 total)
python download_hourly_data.py

# Download specific symbols
python download_hourly_data.py --symbols AAPL BTCUSD NVDA

# Download only stocks
python download_hourly_data.py --only-stocks

# Download only crypto
python download_hourly_data.py --only-crypto
```

## Command Options

```bash
--symbols AAPL BTCUSD ...   # Download specific symbols
--only-stocks               # Download only stock symbols
--only-crypto              # Download only crypto symbols
--output-dir PATH          # Output directory (default: trainingdatahourly)
--stock-years N            # Years of stock history (default: 7)
--crypto-years N           # Years of crypto history (default: 10)
--sleep SECONDS            # Delay between requests (default: 0.5)
--force                    # Re-download even if files exist
```

## Output Structure

```
trainingdatahourly/
├── stocks/
│   ├── AAPL.csv
│   ├── NVDA.csv
│   └── ...
├── crypto/
│   ├── BTCUSD.csv
│   ├── ETHUSD.csv
│   └── ...
└── download_summary.csv   # Summary of all downloads
```

## CSV Format

Each CSV file contains hourly OHLCV data:
- `timestamp` - Hour timestamp (index)
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume
- `symbol` - Symbol name

## Monitoring Progress

```bash
# Check progress during download
./check_hourly_progress.sh

# View live download log
tail -f hourly_download.log

# Count downloaded files
ls trainingdatahourly/stocks/*.csv | wc -l
ls trainingdatahourly/crypto/*.csv | wc -l
```

## Running in Background

```bash
# Start download in background
source .venv313/bin/activate
nohup python download_hourly_data.py > hourly_download.log 2>&1 &

# Monitor progress
./check_hourly_progress.sh

# View the log
tail -f hourly_download.log
```

## Default Symbol Lists

The script uses symbol lists from `alpaca_wrapper.py`:

- **DEFAULT_STOCK_SYMBOLS**: 278 US stocks (tech, finance, consumer, energy, etc.)
- **DEFAULT_CRYPTO_SYMBOLS**: 17 crypto pairs (BTC, ETH, SOL, UNI, MATIC, etc.)

## Rate Limiting

Alpaca has rate limits on API requests. The script includes:
- Default 0.5 second delay between symbols
- Adjustable via `--sleep` parameter
- Automatic skip of already-downloaded files

## Example Downloads

Based on test runs (as of Nov 2025):
- AAPL: 9,970 hourly bars (2018-11-14 to 2025-11-12)
- BTCUSD: 42,611 hourly bars (2015-11-15 to 2025-11-12)
- ETHUSD: 42,609 hourly bars (2015-11-15 to 2025-11-12)

## Troubleshooting

### "No module named 'pandas'"
Make sure you activate the virtual environment first:
```bash
source .venv313/bin/activate
```

### Crypto symbols fail with "invalid symbol"
The script automatically converts symbols like `BTCUSD` to `BTC/USD` format required by Alpaca.

### Download stops partway
The script skips already-downloaded files by default. Just re-run it to continue:
```bash
python download_hourly_data.py
```

To force re-download:
```bash
python download_hourly_data.py --force
```

## Use Cases

This hourly data is useful for:
- Training time-series forecasting models
- Intraday strategy backtesting
- Feature engineering for ML models
- Higher-resolution market analysis
- Multi-timeframe indicator calculations

## Integration with Training Pipeline

The hourly data can be used alongside daily data:
- Daily data: `trainingdata/` (from `download_training_data.py`)
- Hourly data: `trainingdatahourly/` (from `download_hourly_data.py`)

Both use the same CSV format with OHLCV columns and timestamp index.
