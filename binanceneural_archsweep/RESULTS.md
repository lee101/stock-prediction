# Architecture Sweep Results - 2026-03-24

## Summary

Tested 7 architecture configs x 3 symbols (BTCUSD, ETHUSD, SOLUSD) with robust multi-scenario evaluation (fees 0-15bps, fill slippage 0-10bps, decision lag 0-2 bars, intensity scales 0.8-1.2).

## Key Findings

1. **SOLUSD is the most profitable symbol** - all 5 positive robust scores are SOL
2. **Classic transformer beats nano** on SOLUSD (+0.053 vs +0.026) - RoPE/GQA/RMSNorm don't help for 96-step sequences
3. **Fee-trained model** (proven_nano_fee) is the **only positive BTC config** (+0.005)
4. **Memory tokens hurt** across all symbols (-0.068 to -0.088)
5. **Deeper models don't help** - 8 layers ~= 4 layers for this data
6. **Wider models slightly help** for BTC (-0.014 vs -0.020)
7. **Remote 5090 with better data** improves all results (BTC nano: -0.001 remote vs -0.020 local)

## Top Models

| Rank | Config | Symbol | Robust Score | Mean Return |
|------|--------|--------|-------------|-------------|
| 1 | proven_classic | SOLUSD | +0.053 | +0.129 |
| 2 | proven_nano | SOLUSD | +0.026 | +0.009 |
| 3 | proven_nano_wide | SOLUSD | +0.026 | +0.046 |
| 4 | proven_nano_deep | SOLUSD | +0.024 | +0.015 |
| 5 | proven_nano_fee | BTCUSD | +0.005 | -0.024 |

## Best Checkpoints

- SOLUSD: `binanceneural/checkpoints/archsweep_proven_classic_SOLUSD_20260323_235635/epoch_004.pt`
- BTCUSD: `binanceneural/checkpoints/archsweep_proven_nano_fee_BTCUSD_20260324_030116/epoch_001.pt`
- ETHUSD: `binanceneural/checkpoints/archsweep_proven_nano_fee_ETHUSD_20260324_031912/epoch_003.pt`

## Strategy Recommendations

- **SOL**: Use proven_classic (256d, 4L, 8H), train without fees, intensity=1.2
- **BTC**: Use proven_nano_fee (256d, 4L, 8H, GQA), train WITH fees+lag, intensity=0.8
- **ETH**: Marginal edge at best; nano_fee is least-bad
- More/fresher training data helps significantly (remote results)
- Keep epochs low (5) to avoid overfitting
