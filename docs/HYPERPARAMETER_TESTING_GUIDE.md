# Hyperparameter Testing Guide

This guide explains how to use the enhanced hyperparameter testing tools to optimize MAE for Kronos and Toto models on your stock pairs.

## Overview

I've created three scripts for comprehensive hyperparameter exploration:

1. **test_hyperparameters_quick.py** - Fast strategic testing (11 Kronos configs, 16 Toto configs)
2. **test_hyperparameters_extended.py** - Comprehensive grid search (thousands of configs)
3. **analyze_hyperparam_results.py** - Result analysis and comparison

## Quick Start

### 1. Quick Test (Recommended for initial exploration)

Test on a few symbols with strategic parameter selections:

```bash
source .venv/bin/activate
python test_hyperparameters_quick.py --symbols AAPL MSFT BTCUSD
```

Results saved to: `hyperparams_quick/`

### 2. Comprehensive Test (For thorough optimization)

Test all combinations on specific symbols:

```bash
source .venv/bin/activate
python test_hyperparameters_extended.py --symbols AAPL --max-kronos-configs 100 --max-toto-configs 100
```

Results saved to: `hyperparams_extended/`

### 3. Test All Stock Pairs

Run on all stock pairs in trainingdata/:

```bash
source .venv/bin/activate
python test_hyperparameters_quick.py  # Tests all CSVs by default
```

### 4. Analyze Results

```bash
source .venv/bin/activate
python analyze_hyperparam_results.py --results-dir hyperparams_quick
```

Or compare two result sets:

```bash
python analyze_hyperparam_results.py --compare-dirs hyperparams_quick hyperparams_extended
```

## Hyperparameter Ranges Tested

### Kronos Parameters

**Quick Test (11 configs):**
- Temperature: 0.12 - 0.22 (conservative values)
- Top-P: 0.78 - 0.82
- Sample Count: 192 - 240
- Top-K: 20 - 24
- Context: 224 - 256
- Clip: 1.5 - 2.0

**Extended Test (thousands of configs):**
- Temperature: 0.10 - 0.30 (comprehensive range)
- Top-P: 0.70 - 0.90
- Sample Count: 128 - 320
- Top-K: 0, 16, 20, 24, 28, 32
- Context: 192, 224, 256, 288
- Clip: 1.2 - 2.5

### Toto Parameters

**Quick Test (16 configs):**
- Num Samples: 256 - 3072
- Aggregations:
  - quantile_0.15, quantile_0.18, quantile_0.20
  - trimmed_mean_10
  - lower_trimmed_mean_15
  - quantile_plus_std_0.15_0.12, quantile_plus_std_0.15_0.15
  - mean_quantile_mix_0.15_0.3

**Extended Test (hundreds of configs):**
- Num Samples: 64 - 4096
- Aggregations: 30+ different strategies including:
  - Quantiles: 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35
  - Trimmed means: 5%, 10%, 15%, 20%
  - Lower trimmed means: 10%, 15%, 20%
  - Mean ± std: various coefficients
  - Quantile + std combinations
  - Mean-quantile mixes

## Key Features

### Strategic Parameter Selection

The quick test focuses on **conservative configurations** based on prior research showing:
- Lower temperatures (0.12-0.18) generally perform better
- Moderate sample counts (192-224) balance speed and accuracy
- Conservative aggregations (lower quantiles, trimmed means) reduce outlier impact

### Comprehensive Exploration

The extended test explores:
- **Region 1**: Very conservative (temp 0.10-0.14) - best for stable predictions
- **Region 2**: Medium conservative (temp 0.15-0.18) - balanced exploration
- **Region 3**: Moderate (temp 0.20-0.24) - higher diversity

### Caching

Both scripts cache model instances to speed up testing:
- Kronos wrappers are cached by configuration
- Toto pipeline is a singleton
- CUDA memory is cleared between runs

## Test Results Interpretation

### MAE Metrics

Each configuration is evaluated on:
- **Validation MAE**: Performance on validation window (20 steps)
- **Test MAE**: Performance on held-out test window (20 steps)
- **Return MAE**: Mean absolute error on percentage returns

Lower MAE = better performance

### Example Results (AAPL Quick Test)

```
Best Kronos: kronos_temp0.12_p0.78_s192_k24_clip1.5_ctx224
  Validation MAE: 3.7948
  Test MAE: 4.2331
  Latency: 10.90s
```

This shows:
- Very low temperature (0.12) works best for AAPL
- Tight sampling (top_p=0.78) reduces variance
- Moderate sample count (192) is sufficient

## Advanced Usage

### Test Only Kronos or Toto

```bash
# Only Kronos
python test_hyperparameters_quick.py --symbols AAPL --skip-toto

# Only Toto
python test_hyperparameters_quick.py --symbols AAPL --skip-kronos
```

### Limit Configurations

```bash
python test_hyperparameters_extended.py \
  --symbols BTCUSD \
  --max-kronos-configs 50 \
  --max-toto-configs 50
```

### Export Analysis Results

```bash
python analyze_hyperparam_results.py \
  --results-dir hyperparams_quick \
  --export-csv results.csv
```

## File Structure

```
hyperparams_quick/
  kronos/
    AAPL.json      # Best Kronos config for AAPL
    MSFT.json
    ...
  toto/
    AAPL.json      # Best Toto config for AAPL
    MSFT.json
    ...

hyperparams_extended/
  kronos/
    ...
  toto/
    ...
```

Each JSON file contains:
- Selected configuration parameters
- Validation metrics
- Test metrics
- Window sizes
- Metadata

## Tips for Better Results

### 1. Start with Quick Test

Run the quick test first to identify promising parameter regions:

```bash
python test_hyperparameters_quick.py --symbols AAPL MSFT NVDA
```

### 2. Analyze Trends

Use the analysis script to understand which parameters work best:

```bash
python analyze_hyperparam_results.py --results-dir hyperparams_quick
```

Look for patterns like:
- "Lower temperatures consistently perform better"
- "Quantile 0.15 aggregation works well across symbols"
- "Context length 224 provides good balance"

### 3. Run Comprehensive Test on Best Candidates

Once you identify promising regions, run the extended test on specific symbols:

```bash
python test_hyperparameters_extended.py --symbols BTCUSD --max-kronos-configs 200
```

### 4. Compare Results

Compare quick vs extended results:

```bash
python analyze_hyperparam_results.py --compare-dirs hyperparams_quick hyperparams_extended
```

## Performance Considerations

### Memory Usage

- **Kronos**: ~4-6 GB GPU memory per wrapper
- **Toto**: ~8-10 GB GPU memory
- Both scripts use CUDA cache clearing between runs

### Timing

- **Quick Test**: ~2-5 minutes per symbol (27 configs total)
- **Extended Test**: ~30-120 minutes per symbol (hundreds of configs)

### Recommendations

1. Use quick test for initial exploration
2. Run extended test overnight for thorough optimization
3. Use `--max-*-configs` to limit testing time
4. Test on representative symbols first (high/low volatility, different asset classes)

## Understanding the Output

### During Testing

```
[INFO] Kronos 1/11: kronos_temp0.12_p0.78_s192_k24_clip1.5_ctx224
  -> MAE: 3.7948, Latency: 10.90s
```

- Configuration name describes all parameters
- MAE is validation MAE (lower is better)
- Latency is total inference time for validation window

### Best Results

```
[INFO] Best Kronos: kronos_temp0.12_p0.78_s192_k24_clip1.5_ctx224 (MAE: 3.7948)
[INFO] Test MAE: 4.2331
```

- Best config selected by validation MAE
- Test MAE shows generalization performance
- Small gap between validation and test is good (no overfitting)

### Analysis Output

```
KRONOS Results Summary
Total symbols tested: 5
Validation MAE Statistics:
  Mean:   4.2531
  Median: 3.9684
  Min:    3.7948 (AAPL)
  Max:    5.1234 (NVDA)
```

Shows aggregate statistics across all tested symbols.

## Next Steps

After finding good hyperparameters:

1. Use the best configs in your production trading scripts
2. Update default configs in `test_kronos_vs_toto.py`
3. Periodically re-run tests as market conditions change
4. Test on new stock pairs before trading them

## Troubleshooting

### Out of Memory Errors

Reduce sample counts or test fewer configs:
```bash
python test_hyperparameters_extended.py --max-kronos-configs 20
```

### Slow Performance

Use quick test instead of extended, or limit symbols:
```bash
python test_hyperparameters_quick.py --symbols AAPL MSFT
```

### Missing Dependencies

Activate virtual environment:
```bash
source .venv/bin/activate
```

## Example Workflow

Here's a complete workflow for optimizing a new stock pair:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Quick test on new symbol
python test_hyperparameters_quick.py --symbols NEWSTOCK

# 3. Analyze results
python analyze_hyperparam_results.py --results-dir hyperparams_quick

# 4. If promising, run comprehensive test
python test_hyperparameters_extended.py --symbols NEWSTOCK --max-kronos-configs 100

# 5. Compare results
python analyze_hyperparam_results.py --compare-dirs hyperparams_quick hyperparams_extended

# 6. Export for documentation
python analyze_hyperparam_results.py --results-dir hyperparams_extended --export-csv newstock_results.csv
```

## Summary

You now have three powerful tools for hyperparameter optimization:

- **Quick testing** for rapid iteration and initial exploration
- **Extended testing** for comprehensive optimization
- **Analysis tools** for understanding results and comparing approaches

Start with the quick test, analyze the results, then run extended tests on promising configurations. This iterative approach will help you find the best hyperparameters for each stock pair while managing computational resources efficiently.

Happy optimizing! 🚀
