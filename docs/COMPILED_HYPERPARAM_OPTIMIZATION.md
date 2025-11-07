# Compiled Model Hyperparameter Optimization

This guide explains how to run comprehensive hyperparameter optimization for the compiled Toto and Kronos models.

## Overview

The compiled model optimization system:
- Uses `torch.compile` with `reduce-overhead` mode for maximum performance
- Searches extensively across hyperparameter space (100+ trials by default)
- Only updates configs if we achieve better PnL (measured by validation MAE)
- Runs in parallel across multiple symbols for efficiency
- Validates improvements using both validation and test sets

## Quick Start

### Optimize All Symbols

Run optimization across all symbols with default settings (100 trials each):

```bash
./run_hyperparam_search_compiled.sh
```

### Optimize Specific Symbols

Optimize only specific symbols:

```bash
./run_hyperparam_search_compiled.sh --symbols BTCUSD ETHUSD AAPL NVDA
```

### Extensive Search

For more thorough hyperparameter search (200 trials per model):

```bash
./run_hyperparam_search_compiled.sh --trials 200
```

### Test on Single Symbol

Test the optimization on a single symbol first:

```bash
python optimize_compiled_models.py --symbol BTCUSD --trials 50
```

## Advanced Usage

### More Parallel Workers

If you have sufficient GPU memory, increase workers for faster execution:

```bash
./run_hyperparam_search_compiled.sh --workers 4
```

**Warning:** Compiled models use more GPU memory. Monitor with `nvidia-smi`.

### Optimize Both Models

Optimize both Toto and Kronos (takes longer):

```bash
./run_hyperparam_search_compiled.sh --model both --trials 150
```

### Custom Configuration

You can also set options via environment variables:

```bash
TRIALS=200 WORKERS=3 MODEL=toto ./run_hyperparam_search_compiled.sh
```

## How It Works

### 1. Compilation Configuration

The system applies compilation optimizations from `toto_compile_config.py`:
- `TOTO_COMPILE=1` - Enable compilation
- `TOTO_COMPILE_MODE=reduce-overhead` - Balanced performance/stability
- `TOTO_COMPILE_BACKEND=inductor` - PyTorch Inductor backend
- `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` - Reduce graph breaks

### 2. Extensive Hyperparameter Search

For **Toto** (compiled), the system searches:

- **Aggregation methods** (30+ options):
  - Trimmed means: `trimmed_mean_5`, `trimmed_mean_10`, ..., `trimmed_mean_25`
  - Lower/upper trimmed means for directional bias
  - Quantiles: `quantile_0.10` through `quantile_0.90`
  - Winsorized means
  - Basic: `mean`, `median`

- **Sample counts**: 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096
  - Compiled models can handle larger sample sizes efficiently

- **Samples per batch**: Automatically selected based on num_samples
  - Small (≤128 samples): [16, 32, 64]
  - Medium (≤1024 samples): [64, 128, 256]
  - Large (>1024 samples): [128, 256, 512]

For **Kronos**, the system searches:
- **Temperature**: 0.05 to 0.40
- **Top-p**: 0.65 to 0.95
- **Top-k**: [0, 12, 16, 20, 24, 28, 32, 40, 48]
- **Sample count**: [96, 128, 160, 192, 224, 256, 320]
- **Max context**: [128, 192, 224, 256, 320]
- **Clip**: 1.0 to 3.0

### 3. Validation and Testing

Each configuration is evaluated on:
1. **Validation set** (30 most recent data points before test)
   - Used for hyperparameter selection
   - Metric: `pct_return_mae` (percentage return MAE)

2. **Test set** (20 most recent data points)
   - Used for final evaluation
   - Ensures no overfitting to validation set

### 4. Update Logic

A new configuration is only saved if:
- Validation MAE improves by >1% compared to existing config
- OR no existing config exists
- OR `--force` flag is used

## Output Files

### Updated Configs

Improved configurations are saved to:
- `hyperparams/best/{SYMBOL}.json` - Updated best config (used by trading system)
- `hyperparams/optimized_compiled/{SYMBOL}.json` - Backup copy

### Summary

Summary report saved to:
- `results/compiled_optimization_{TIMESTAMP}.json`

Contains:
- Results for each symbol
- Improvement statistics
- Configuration used
- Total execution time

### Logs

Detailed logs saved to:
- `logs/hyperparam_optimization_compiled_{TIMESTAMP}.log`

## Expected Performance

### Optimization Time

Per symbol (100 trials):
- **Toto compiled**: ~30-60 minutes
- **Kronos**: ~20-40 minutes

Total for all 24 symbols (2 workers):
- ~10-15 hours for comprehensive search

### GPU Memory Usage

- **Single Toto compiled model**: ~4-6 GB VRAM
- **2 parallel workers**: ~10-12 GB VRAM total
- Recommended: GPU with ≥16 GB VRAM for smooth operation

### Expected Improvements

Based on previous optimizations:
- **Typical improvement**: 5-15% reduction in MAE
- **Best cases**: 30-50% reduction in MAE
- **No improvement**: ~20-30% of symbols (already well-optimized)

## Monitoring

### Check Progress

Monitor running optimization:

```bash
# Watch log file
tail -f logs/hyperparam_optimization_compiled_*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Intermediate Results

Check intermediate results:

```bash
# List updated configs
ls -lt hyperparams/optimized_compiled/

# View specific result
cat hyperparams/optimized_compiled/BTCUSD.json | jq '.'
```

## Troubleshooting

### Out of Memory

If you get CUDA OOM errors:
1. Reduce workers: `--workers 1`
2. Close other GPU processes
3. Reduce max sample counts in `optimize_compiled_models.py`

### Slow Performance

If optimization is too slow:
1. Reduce trials: `--trials 50`
2. Test on subset: `--symbols BTCUSD ETHUSD`
3. Check GPU utilization with `nvidia-smi`

### No Improvements

If no symbols show improvement:
1. Check if existing configs are already well-optimized
2. Increase trials for more thorough search: `--trials 200`
3. Review validation window size (may be too small)

### Compilation Issues

If you see recompilation warnings:
- This is normal for first run (model is being compiled)
- Subsequent runs will use cached compiled code from `compiled_models/torch_inductor/`
- To force recompilation: `rm -rf compiled_models/torch_inductor/`

## Best Practices

### Initial Run

For first-time optimization:
1. Test on single symbol first
2. Review results and adjustment needs
3. Run on small subset (3-5 symbols)
4. Run full optimization overnight

### Regular Reoptimization

Reoptimize periodically:
- **Monthly**: Quick optimization (50 trials)
- **Quarterly**: Full optimization (100-200 trials)
- **After major market regime change**: Comprehensive search

### Symbol Priority

Optimize high-volume symbols first:
1. BTCUSD, ETHUSD (crypto - high volatility)
2. NVDA, AAPL, MSFT (high-volume tech stocks)
3. Other stocks

## Example Workflows

### Quick Test (Single Symbol)

```bash
# Test on BTCUSD with 30 trials (~15 minutes)
python optimize_compiled_models.py --symbol BTCUSD --trials 30
```

### Moderate Search (Top Symbols)

```bash
# Optimize top 5 symbols with 100 trials (~3-4 hours)
./run_hyperparam_search_compiled.sh \
    --trials 100 \
    --workers 2 \
    --symbols BTCUSD ETHUSD NVDA AAPL MSFT
```

### Comprehensive Search (All Symbols)

```bash
# Full overnight optimization (10-15 hours)
./run_hyperparam_search_compiled.sh \
    --trials 150 \
    --workers 2 \
    --model toto
```

### Force Update (Override Improvement Check)

```bash
# Force update even if no improvement
python optimize_compiled_models.py \
    --symbol BTCUSD \
    --trials 100 \
    --force
```

## Notes

- **Only Toto configs are updated** in `hyperparams/best/` by default (primary model)
- Kronos results are saved separately if `--model both` is used
- Compilation cache is stored in `compiled_models/torch_inductor/`
- First run will be slower due to compilation (cache is built)
- Subsequent runs use cached compiled code for faster execution
