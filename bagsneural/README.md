# Bags Neural Trading

Train a lightweight neural model on Bags.fm OHLC data to produce:
- **signal** (buy vs hold) via sigmoid
- **size** (0..1 risk allocation) via sigmoid

The model is a simple MLP to keep behavior predictable. It learns from
future returns with an explicit cost penalty (swap + slippage in bps).

## Quick start

### Train
```bash
source .venv/bin/activate
python bagsneural/run_train.py \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --context 64 --horizon 3 --epochs 30
```

Artifacts:
- `bagsneural/checkpoints/bagsneural_<MINT>_best.pt`
- `bagsneural/checkpoints/bagsneural_<MINT>_best.json`

### Backtest
```bash
source .venv/bin/activate
python bagsneural/run_backtest.py \
  --mint HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS \
  --checkpoint bagsneural/checkpoints/bagsneural_HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS_best.pt \
  --buy-threshold 0.55 --sell-threshold 0.45
```

To evaluate only the holdout (last 20%) window:
```bash
python bagsneural/run_backtest.py --mint ... --checkpoint ... --test-split 0.2
```

### Multi-token training
```bash
python bagsneural/run_train_multi.py \
  --mints HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS,CZRsbB6BrHsAmGKeoxyfwzCyhttXvhfEukXCWnseBAGS,7pskt3A1Zsjhngazam7vHWjWHnfgiRump916Xj7ABAGS,W6n4FdEd7D6SetV5FxMMo8kNrduoj9qLWUu2v64BAGS,5ghpHpgiew7WuuKY1f3CSMgGHmDiZUeMjZrgDrfzBAGS \
  --context 64 --horizon 3 --epochs 25
```

## Notes
- `--min-return` and `--size-scale` control how aggressive sizing is.
- Backtest uses a **1 SOL max exposure** by default to match current safety.
- The model does not yet model LP/AMM curve impact explicitly; cost bps is the proxy.
