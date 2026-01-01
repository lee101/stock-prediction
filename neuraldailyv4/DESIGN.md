# Neural Daily V4 - Chronos-2 Inspired Architecture

## Key Ideas from Chronos-2

1. **Patching**: Chunk time series into non-overlapping patches (not per-timestep)
2. **Direct multi-step**: Predict multiple future windows at once (not autoregressive)
3. **Quantile outputs**: Output distribution (quantiles) instead of point estimates
4. **Dual attention**: Time attention + Group attention (cross-symbol)
5. **Training = Inference**: Same aggregation process at both times

## V4 Architecture Changes

### 1. Input Patching

Instead of processing 256 individual days:
- Chunk into patches of 5 days (weekly resolution)
- 256 days → 51 patches (+ 1 partial)
- Each patch is embedded via small residual MLP

```
Input: (batch, 256, 20 features)
       ↓ reshape
Patches: (batch, 51, 5, 20)
       ↓ patch embed (linear + residual MLP)
Embedded: (batch, 51, hidden_dim)
```

Benefits:
- Faster processing (51 tokens vs 256)
- Forces model to learn weekly patterns
- More similar to Chronos-2

### 2. Multi-Window Output (Direct Multi-Step)

Output predictions for N windows simultaneously:
- N = 4 windows: [1-5 days, 5-10 days, 10-15 days, 15-20 days]
- Each window gets: buy_price, sell_price, position_size, confidence, exit_day

```
Output: (batch, num_windows=4, 5 logits)
```

### 3. Quantile Outputs

Instead of single point estimate for prices:
- Output 3 quantiles: [0.25, 0.5, 0.75] for buy/sell prices
- Allows uncertainty-aware trading

```
buy_price_quantiles: (batch, num_windows, 3)  # q25, q50, q75
sell_price_quantiles: (batch, num_windows, 3)
```

### 4. Learned Position Sizing

Position size as a learned function of:
- Model confidence
- Predicted volatility (from quantile spread)
- Asset class

```
position_size = f(confidence, quantile_spread, asset_class)
```

Constraints:
- Max 2x leverage for stocks
- Max 1x leverage for crypto (already losing money there)
- Scale by inverse volatility (smaller positions when uncertain)

### 5. Trimmed Mean Aggregation

At inference (and training):
1. Get predictions from all N windows
2. Simulate each window's trade
3. Aggregate via trimmed mean (discard highest/lowest)
4. Final prediction is aggregated result

This provides:
- Robustness to outlier predictions
- Ensemble-like benefits
- Same process at train and inference

## New Model Components

### PatchEmbedding
```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=5, input_dim=20, hidden_dim=256):
        self.patch_size = patch_size
        self.embed = nn.Linear(patch_size * input_dim, hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B, T, D = x.shape
        num_patches = T // self.patch_size

        # Reshape to patches
        x = x[:, :num_patches * self.patch_size]
        x = x.view(B, num_patches, self.patch_size * D)

        # Embed + residual
        h = self.embed(x)
        h = h + self.residual(h)
        return h
```

### MultiWindowHead
```python
class MultiWindowHead(nn.Module):
    def __init__(self, hidden_dim, num_windows=4, num_quantiles=3):
        self.num_windows = num_windows
        self.num_quantiles = num_quantiles

        # Per window: buy_quantiles(3) + sell_quantiles(3) + position(1) + confidence(1) + exit_day(1) = 9
        self.head = nn.Linear(hidden_dim, num_windows * 9)

    def forward(self, h):
        # h: (batch, hidden_dim) - pooled representation
        logits = self.head(h)
        logits = logits.view(-1, self.num_windows, 9)
        return {
            "buy_quantiles": logits[..., 0:3],
            "sell_quantiles": logits[..., 3:6],
            "position_size": logits[..., 6:7],
            "confidence": logits[..., 7:8],
            "exit_day": logits[..., 8:9],
        }
```

### TrimmedMeanAggregator
```python
class TrimmedMeanAggregator(nn.Module):
    def __init__(self, trim_fraction=0.25):
        self.trim_fraction = trim_fraction

    def forward(self, window_returns):
        # window_returns: (batch, num_windows)
        # Sort and trim
        sorted_returns = torch.sort(window_returns, dim=-1).values
        n = window_returns.size(-1)
        trim_n = int(n * self.trim_fraction)

        if trim_n > 0:
            trimmed = sorted_returns[:, trim_n:-trim_n]
        else:
            trimmed = sorted_returns

        return trimmed.mean(dim=-1)
```

## Training Changes

### Loss Function
```python
def compute_v4_loss(predictions, future_data, config):
    # Simulate each window
    window_results = []
    for w in range(config.num_windows):
        result = simulate_window(
            predictions[w],
            future_data,
            window_start=w * config.window_size,
            window_end=(w + 1) * config.window_size,
        )
        window_results.append(result)

    # Aggregate via trimmed mean
    returns = torch.stack([r.returns for r in window_results], dim=-1)
    aggregated_return = trimmed_mean(returns)

    # Loss components
    loss = -aggregated_return.mean()
    loss += config.sharpe_weight * -sharpe(aggregated_return)
    loss += config.uncertainty_weight * quantile_calibration_loss(predictions, future_data)

    return loss
```

### Quantile Calibration Loss
Ensures quantile predictions are calibrated:
```python
def quantile_calibration_loss(buy_quantiles, actual_lows, quantile_levels=[0.25, 0.5, 0.75]):
    # For each quantile level q, fraction of actuals below prediction should be ~q
    losses = []
    for i, q in enumerate(quantile_levels):
        pred_q = buy_quantiles[..., i]
        fraction_below = (actual_lows < pred_q).float().mean()
        losses.append((fraction_below - q).abs())
    return sum(losses) / len(losses)
```

## Inference Process

1. Build feature tensor from last 256 days
2. Patch and embed
3. Forward through transformer
4. Get multi-window predictions
5. For each window, simulate potential trade
6. Aggregate via trimmed mean (same as training)
7. Return aggregated trading plan

## File Structure

```
neuraldailyv4/
├── __init__.py
├── config.py         # V4 config with new params
├── model.py          # Patching + MultiWindow + Quantiles
├── simulation.py     # Window-based simulation
├── aggregation.py    # Trimmed mean + ensemble logic
├── data.py           # Same as V3 (reuse)
├── trainer.py        # Updated for V4 loss
└── runtime.py        # Updated for V4 inference
```

## Key Hyperparameters

```python
@dataclass
class V4Config:
    # Patching
    patch_size: int = 5           # 5-day patches
    num_windows: int = 4          # 4 prediction windows
    window_size: int = 5          # 5 days per window

    # Quantiles
    num_quantiles: int = 3        # q25, q50, q75
    quantile_levels: tuple = (0.25, 0.5, 0.75)

    # Aggregation
    trim_fraction: float = 0.25   # Trim top/bottom 25%

    # Position sizing
    position_temp: float = 1.0    # Temperature for position sizing
    min_position: float = 0.1     # Min position size
    max_position: float = 1.0     # Max position size

    # Architecture
    hidden_dim: int = 512         # Wide is better (from experiments)
    num_layers: int = 4           # Shallow is better
    num_heads: int = 8
```

## Migration Path

1. Start with V3 data pipeline (already has lookahead)
2. Replace model with V4 architecture
3. Update simulation for multi-window
4. Add aggregation layer
5. Train and validate
6. Compare backtest results

## Expected Benefits

1. **Robustness**: Trimmed mean reduces impact of bad predictions
2. **Uncertainty**: Quantile outputs capture price uncertainty
3. **Speed**: Patching reduces sequence length 5x
4. **Generalization**: Multi-window forces learning across timeframes
5. **Position sizing**: Learned sizing reduces losses on uncertain trades
