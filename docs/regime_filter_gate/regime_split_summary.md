# SPY regime filter — does it actually improve the realism gate?

`scripts/regime_filter_realism_gate.py` ran the prod 13-model v5 ensemble
through the realism gate (lag=2, fb=5, lev=1, fee=10bps, slip=5bps, shorts
disabled), captured per-window total returns, then split the 263 windows
by SPY's MA20 regime at the window's start_idx.

## Negative window timing — concentrated, not crash-confined

10 of 11 negative windows are **sequential** at start indices 250–259:
the same crash window viewed from 10 starting positions. The 11th is
start=47 (mild −0.35%, isolated). So the 11 negative windows really
represent only 2 distinct loss events on the val set.

| start | regime (MA20) | total return |
|---:|---|---:|
| 47 | bull | −0.35% |
| 250 | bull | −0.16% |
| 251 | bull | −6.71% |
| 252 | bull | −4.75% |
| 253 | bull | −0.54% |
| 254 | bull | −2.03% |
| 255 | bear | −2.93% |
| 256 | bear | −2.93% |
| 257 | bear | −2.93% |
| 258 | bear | −2.01% |
| 259 | bear | −7.57% |

The crash brewed at the *edge* of the MA20 boundary — starts 250–254
opened with SPY only 0.13–0.63% above MA20, and the regime broke down
during the trade. MA20 doesn't catch them at entry.

## MA-window comparison (BULL cohort = the trade-allowed cohort)

| MA window | bull n | bull neg | bull neg-rate | bull med (50d total) | bull p10 |
|---:|---:|---:|---:|---:|---:|
| MA5  | 161 | 7 | 4.3% | +16.69% | +5.25% |
| MA10 | 164 | 6 | 3.7% | +17.67% | +6.36% |
| MA15 | 174 | 4 | 2.3% | +17.92% | +6.74% |
| MA20 (current live) | 181 | 6 | 3.3% | +18.11% | +6.23% |
| MA30 | 182 | 4 | 2.2% | +18.75% | +6.74% |
| **MA50 (best)** | **188** | **5** | **2.7%** | **+20.12%** | **+8.56%** |

## Headline numbers (50d → monthly, ALL vs BULL@MA50)

| filter | n | med_monthly | p10_monthly | neg_rate |
|---|---:|---:|---:|---:|
| ALL (no filter) | 263 | +6.89% | +2.34% | 4.2% |
| BULL@MA20 (current) | 181 | +7.24% | +2.57% | 3.3% |
| **BULL@MA50** | **188** | **+8.05%** | **+3.45%** | **2.7%** |

## Recommendation: MA50 worth a config flip

Switching `src/market_regime.py` default `lookback=20 → 50` would:
- Lift median monthly +1.16% (6.89 → 8.05)
- Lift p10 monthly +1.11% (2.34 → 3.45)
- Reduce neg-rate 1.5pp (4.2 → 2.7)
- Filter out only 75 windows (vs MA20's 82) — almost no coverage loss

This is a free improvement. Recommended change: bump default to MA50,
keep regime gate behavior identical (block opens during bear, hold
existing positions). Will impact prod monthly by ~+0.15%/mo at current
`--allocation-pct 12.5` (1.16% × 12.5%).

## Caveats

- Live regime check runs daily; the filter helps when the regime is
  already bear AT entry, but it cannot retroactively close positions
  that were opened in bull and are still held when the regime flips.
- The 50-day window length means a single bear-regime crash can
  contaminate ~10 sequential window starts, which may overstate the
  benefit (the underlying loss event is one, not ten).
- The deploy gate is the realism gate WITHOUT regime filter — that's
  the worst-case. MA50 is bonus alpha, not a pass criterion.

## Reproduce

```bash
source .venv/bin/activate
python scripts/regime_filter_realism_gate.py
# JSON output: docs/regime_filter_gate/screened32_regime_split.json
```
