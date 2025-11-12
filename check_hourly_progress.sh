#!/bin/bash
# Monitor hourly data download progress

echo "=== Hourly Data Download Progress ==="
echo ""

# Count downloaded files
stock_count=$(ls -1 trainingdatahourly/stocks/*.csv 2>/dev/null | wc -l)
crypto_count=$(ls -1 trainingdatahourly/crypto/*.csv 2>/dev/null | wc -l)
total_count=$((stock_count + crypto_count))

echo "Downloaded files:"
echo "  Stocks: $stock_count"
echo "  Crypto: $crypto_count"
echo "  Total:  $total_count / 295"
echo ""

# Show latest log entries
echo "Latest activity (last 20 lines):"
tail -20 hourly_download.log 2>/dev/null || echo "No log file found"
echo ""

# Check if process is still running
if pgrep -f "download_hourly_data.py" > /dev/null; then
    echo "Status: RUNNING ✓"
else
    echo "Status: COMPLETED or STOPPED"
fi
