# PufferLib3 C++ Market Simulator

High-performance C++ market simulator using LibTorch for GPU-accelerated reinforcement learning training.

## Features

- **GPU-Accelerated**: Runs entirely on GPU using LibTorch (CUDA 12.8)
- **Realistic Market Simulation**:
  - Trading fees: 0.05% (stocks), 0.15% (crypto)
  - Leverage modeling with 6.75% annual cost
  - No short selling or leverage for crypto
- **PnL Tracking**: Comprehensive logging of training and testing performance
- **PufferLib3 Style**: Vectorized environments with 4096 parallel simulations
- **High/Low Strategy Support**: Optional maxdiff-style execution

## Architecture

- `market_config.h`: Configuration constants and settings
- `market_state.h/cpp`: CSV data loading and OHLCV state management
- `portfolio.h/cpp`: Portfolio management, PnL calculation, leverage costs
- `pnl_logger.h/cpp`: Comprehensive logging system
- `market_env.h/cpp`: Main environment class for RL training
- `csv_loader.h/cpp`: Efficient CSV data loading

## Building

```bash
./build.sh
```

## Usage

### Testing
```bash
./build/test_market
```

### Training
```bash
./build/train_market
```

Logs will be saved to `./logs/` directory:
- `train_pnl.csv`: Per-step PnL during training
- `test_pnl.csv`: Per-step PnL during testing
- `episodes.csv`: Episode-level statistics
- `train_summary.txt` / `test_summary.txt`: Summary statistics

## Configuration

Edit `include/market_config.h` to adjust:
- Trading fees
- Leverage costs and limits
- Batch size (NUM_PARALLEL_ENVS)
- Lookback window
- Initial capital

## Training Data

Place CSV files in `../trainingdata/` with format:
```
timestamp,open,high,low,close,volume
```

## Performance

On RTX 5090 with batch size 4096:
- Expected throughput: >100K steps/second
- Memory usage: ~4GB VRAM
- Training speed: ~10x faster than Python implementations

## Integration

The trained policy can be exported as `.pt` and used for inference:
```cpp
auto policy = torch::load("market_policy.pt");
policy->to(device);
```
