# Alpaca Hourly Stop-Loss Experiments

This directory evaluates stop-loss overlays on top of the existing `newnanoalpacahourlyexp` hourly neural checkpoints.

It does not change live trading behavior. It:

- loads one or more Alpaca hourly checkpoints
- regenerates hourly actions on recent data
- replays those actions through `HourlyTraderMarketSimulator`
- sweeps stop-loss percent, slippage, cooldown, and fill-buffer settings

## Example

```bash
python -m newnanoalpacahourlystoploss.run_stoploss_sweep \
  --symbols ETHUSD,BTCUSD \
  --checkpoints "ETHUSD=binanceneural/checkpoints/alpaca_mig4_ethusd_10e_20260205_010817/epoch_010.pt,BTCUSD=binanceneural/checkpoints/alpaca_mig4_btcusd_10e_20260205_004511/epoch_010.pt" \
  --eval-days 10 \
  --fill-buffer-bps 0,5 \
  --stop-loss-pcts 0,0.01,0.015,0.02,0.03 \
  --stop-loss-slippage-pcts 0,0.0005 \
  --stop-loss-cooldown-bars 0,1,2 \
  --output-dir analysis/newnanoalpacahourlystoploss/eth_btc_10d
```

## Outputs

The sweep writes:

- `bars.csv`
- `actions.csv`
- `summary.csv`
- `summary.json`
- `best_result.json`

`summary.csv` includes baseline-relative deltas for runs where `stop_loss_pct=0`, so it is easy to see whether a stop-loss overlay actually helped.
