# Quick Torch Compile Test Guide

## 🚀 Quick Start (5 minutes)

Test if torch.compile affects your strategy returns:

```bash
# Test BTCUSD only (fastest)
python scripts/compare_compile_backtest_returns.py --quick

# Test both BTC and ETH (recommended)
python scripts/compare_compile_backtest_returns.py

# Test specific symbols
python scripts/compare_compile_backtest_returns.py --symbols BTCUSD ETHUSD SOLUSD
```

## 📊 What It Tests

This compares **actual strategy PnL** between:
- ✅ Eager mode (torch.compile disabled)
- ⚠️ Compiled mode (torch.compile enabled)

For each symbol, it runs full backtests and compares:
- **Return** - Total strategy return
- **Sharpe** - Risk-adjusted return
- **Final Day** - Last day performance

## 🎯 Decision Criteria

After running, check the report at `evaltests/compile_backtest_comparison.md`:

### 🟢 PASS (Use torch.compile)
- Return delta < 1%
- Strategy performance unchanged
- Safe to use if performance benefits

### 🟡 WARNING (Use with caution)
- Return delta 1-5%
- Minor PnL impact
- Consider disabling if risk-averse

### 🔴 FAIL (Don't use torch.compile)
- Return delta > 5%
- Significant PnL impact
- **Must disable for production**

## 📝 Example Output

```
📊 BTCUSD Quick Summary:
  🟢 MaxDiff: Δ return = +0.0023
  🟢 Simple: Δ return = -0.0015
  🟡 CI Guard: Δ return = +0.0234

FINAL RECOMMENDATION
====================
🟢 torch.compile appears safe (PnL divergence <5%)
   Review report for details
```

## 🔧 End-to-End Testing

After backtest comparison, test with live forecasts:

```bash
# Test eager mode
export TOTO_DISABLE_COMPILE=1
PYTHONPATH=. .venv/bin/python scripts/alpaca_cli.py show_forecasts ETHUSD

# Test compiled mode
export TOTO_DISABLE_COMPILE=0
PYTHONPATH=. .venv/bin/python scripts/alpaca_cli.py show_forecasts ETHUSD

# Compare the forecasts manually
```

## ⚡ Quick Commands

```bash
# 1. Run backtest comparison (determines if safe to use)
python scripts/compare_compile_backtest_returns.py --quick

# 2. Check report
cat evaltests/compile_backtest_comparison.md

# 3. If PASS, test end-to-end
export TOTO_DISABLE_COMPILE=0
PYTHONPATH=. .venv/bin/python scripts/alpaca_cli.py show_forecasts ETHUSD

# 4. If FAIL, disable and use eager mode
export TOTO_DISABLE_COMPILE=1
python trade_stock_e2e.py
```

## 🐛 If Tests Fail

### Backtest doesn't run
```bash
# Check backtest script
python backtest_test3_inline.py --symbol BTCUSD --help

# Verify environment
python -c "import torch; print(torch.cuda.is_available())"
```

### Results file not found
```bash
# Check results directory
ls -la evaltests/backtests/

# Manually specify output suffix
# (edit compare_compile_backtest_returns.py if needed)
```

### Compilation errors
```bash
# Disable torch.compile
export TOTO_DISABLE_COMPILE=1

# Run basic test
python scripts/run_compile_stress_test.py --mode quick
```

## 📋 Full Test Sequence

Comprehensive testing before production deployment:

```bash
# 1. Stress test (validates accuracy)
python scripts/run_compile_stress_test.py --mode production-check

# 2. Backtest comparison (validates strategy returns)
python scripts/compare_compile_backtest_returns.py

# 3. End-to-end test (validates full pipeline)
export TOTO_DISABLE_COMPILE=0
PYTHONPATH=. .venv/bin/python scripts/alpaca_cli.py show_forecasts BTCUSD
PYTHONPATH=. .venv/bin/python scripts/alpaca_cli.py show_forecasts ETHUSD

# 4. Review all reports
cat tests/compile_stress_results/compile_stress_report.md
cat evaltests/compile_backtest_comparison.md

# 5. Make decision
./scripts/toggle_torch_compile.sh enable   # if all PASS
./scripts/toggle_torch_compile.sh disable  # if any FAIL
```

## ⏱️ Expected Runtime

| Test | Duration | Purpose |
|------|----------|---------|
| `--quick` | ~3-5 min | Fast validation (BTC only) |
| Default | ~10-15 min | Full validation (BTC + ETH) |
| Multiple symbols | ~5 min/symbol | Comprehensive testing |

## 🎓 Understanding Results

### Strategy Return Delta

```
MaxDiff: Compiled 0.1043, Eager 0.1020, Δ +0.0023 🟢
```

- **Δ +0.0023**: Compiled mode returned 0.23% more
- **🟢**: Delta is small (<1%), acceptable
- **Impact**: Minimal PnL difference

### Sharpe Ratio Delta

```
MaxDiff: Compiled 18.24, Eager 17.89, Δ +0.35
```

- **Δ +0.35**: Compiled has slightly better risk-adjusted returns
- Small Sharpe changes are normal
- Focus on return delta for PnL impact

## 🚨 Immediate Action

If you're experiencing slow production performance right now:

```bash
# 1. Quick disable (1 second)
export TOTO_DISABLE_COMPILE=1

# 2. Restart bot
python trade_stock_e2e.py

# 3. Run comparison later when convenient
python scripts/compare_compile_backtest_returns.py --quick
```

## 📚 More Information

- Full guide: `docs/TORCH_COMPILE_GUIDE.md`
- Status report: `TORCH_COMPILE_STATUS.md`
- Test README: `tests/compile_stress_tests_README.md`
