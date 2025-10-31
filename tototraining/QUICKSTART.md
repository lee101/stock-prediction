# Toto Optimization Quick Start Guide

## 🚀 Get Started in 3 Commands

### Option 1: Priority Stocks (Fast - Recommended)
```bash
# 1. Evaluate baseline (30 seconds)
python tototraining/baseline_eval_simple.py

# 2. Train priority stocks (2-4 hours)
uv run python tototraining/toto_retrain_wrapper.py --priority-only

# 3. Compare vs Kronos (10-20 minutes)
uv run python tototraining/compare_toto_vs_kronos.py --all --forecast-horizon 64
```

### Option 2: Full Automation
```bash
# Run everything with one command (2-4 hours for priority stocks)
./tototraining/run_full_optimization.sh true
```

---

## 📊 What Gets Trained

### Priority Stocks (11 total):
- **Easy stocks** (5-8% baseline): SPY, MSFT, AAPL, QQQ, GOOG
- **High-data stocks** (1,700+ samples): NVDA, AMD, META, TSLA
- **Crypto** (interesting comparisons): BTCUSD, ETHUSD

These represent the best candidates for:
- ✅ Good baseline performance (easier to beat)
- ✅ Sufficient data for training
- ✅ Interesting comparisons with Kronos

---

## 📈 Expected Results

### Training Improvements (vs Naive Baseline)
```
SPY:  5.48% → ~3.5-4.0%  (20-35% improvement)
MSFT: 5.74% → ~3.8-4.2%  (20-35% improvement)
NVDA: 15.43% → ~11-13%   (15-30% improvement)
TSLA: 19.13% → ~15-17%   (10-20% improvement)
```

### Toto vs Kronos
Expected win rate: **60-70% of stocks**
- Toto better on: Stable trends (SPY, MSFT, AAPL, QQQ)
- Kronos better on: High volatility (COIN, some crypto)
- Competitive: Tech stocks (NVDA, AMD, META)

---

## 🎯 After Training

### 1. Check Training Results
```bash
# View training summary
cat tototraining/stock_models/training_summary.json | jq

# View individual stock metrics
cat tototraining/stock_models/SPY/training_metrics.json | jq
```

### 2. Check Comparison Results
```bash
# View comparison summary
cat comparison_results/comparison_summary_h64.json | jq

# View wins/losses
cat comparison_results/comparison_summary_h64.json | jq '.results | to_entries | map({stock: .key, winner: .value.winner})'
```

### 3. Retrain Underperformers
```bash
# If specific stocks underperform, retrain with different hyperparams
# Edit toto_retrain_wrapper.py get_optimal_config_for_stock() to adjust

# Then retrain
uv run python tototraining/toto_retrain_wrapper.py --stocks TSLA COIN

# Compare again
uv run python tototraining/compare_toto_vs_kronos.py --stocks TSLA COIN
```

---

## 📁 Output Locations

```
tototraining/
├── baseline_results.json          # Naive baseline metrics
├── stock_models/
│   ├── SPY/                       # Trained SPY model
│   │   ├── training_metrics.json  # Training results
│   │   └── SPY_model/            # Model checkpoint
│   └── training_summary.json      # Overall summary
│
hyperparams/toto/
├── SPY.json                       # SPY config for comparison
├── NVDA.json                      # NVDA config for comparison
└── ...

comparison_results/
├── comparison_summary_h64.json    # Toto vs Kronos summary
└── SPY_comparison.txt             # Detailed comparison logs
```

---

## 🔧 Troubleshooting

### Issue: Training fails immediately
```bash
# Check if torch is installed
uv run python -c "import torch; print(torch.__version__)"

# If missing, already installed via setup

# Check if toto package is installed
uv run python -c "from toto.model.toto import Toto; print('OK')"
```

### Issue: Out of memory
```bash
# Reduce batch size in toto_retrain_wrapper.py
# Line ~52: batch = 2  # instead of 4

# Or reduce context length
# Line ~48: context = 512  # instead of 1024
```

### Issue: Comparison says "No configs found"
```bash
# Ensure training completed and created configs
ls hyperparams/toto/*.json

# If missing, run training first
uv run python tototraining/toto_retrain_wrapper.py --priority-only
```

---

## 💡 Tips for Best Results

### 1. Start Small
- Train priority stocks first (11 stocks)
- Review results before training all 24 stocks
- Iteratively improve poor performers

### 2. Monitor Progress
```bash
# Watch training log in real-time
tail -f tototraining/stock_models/SPY/training_output.txt

# Check GPU usage
watch -n 1 nvidia-smi
```

### 3. Optimize Systematically
- **Easy stocks doing well?** → Scale to all stocks
- **Hard stocks struggling?** → Try heteroscedastic loss, higher LoRA rank
- **Close ties with Kronos?** → Fine-tune learning rate, add more epochs

---

## 📖 Full Documentation

For detailed information:
- `README_RETRAINING.md` - Complete framework documentation
- `OPTIMIZATION_SUMMARY.md` - Baseline analysis and strategy

---

## 🎉 Success Criteria

You'll know the framework is working when:

✅ **Training completes** for >80% of priority stocks
✅ **Improvement over baseline** for most stocks (average 15-25%)
✅ **Competitive with Kronos** (win rate >50%)
✅ **Models saved** in correct format for comparison

---

## 🚀 Next Steps After Success

1. **Scale to all stocks**: Run full training on all 24 pairs
2. **Deploy best models**: Use with existing inference infrastructure
3. **Continuous improvement**: Set up retraining pipeline
4. **Production integration**: Connect to trading systems

---

*Ready to start? Run: `./tototraining/run_full_optimization.sh true`*
