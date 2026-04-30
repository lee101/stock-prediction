package mixed

import "math"

// SimulateLimitBacktest is the verbatim port of trading_bot.go:1431
// (simulateLimitBacktest). Uses limit-fill semantics with fillBufferPct,
// per-side fees on entry+exit, and force-closes any open position at end.
// initialEquity scales the equity curve only; PnL is reported as percent of
// the unit-equity simulation. detail enables Trades/EquityCurve population.
func SimulateLimitBacktest(
	pair string,
	candles []Candle,
	start, end int,
	strategy LiveStrategy,
	initialEquity, feeRate, fillBufferPct float64,
	detail bool,
) BacktestResult {
	if len(candles) < 32 || strategy.Name == "" {
		return BacktestResult{}
	}
	if initialEquity <= 0 {
		initialEquity = 1
	}
	if feeRate < 0 {
		feeRate = 0
	}
	if fillBufferPct < 0 {
		fillBufferPct = 0
	}

	equity := 1.0
	peak := equity
	prevMarked := equity
	maxDD := 0.0
	trades, wins := 0, 0
	position := 0.0
	entryPrice := 0.0
	entryTs := int64(0)

	returns := make([]float64, 0, maxInt(0, end-start))
	tradesOut := make([]Trade, 0)
	curve := make([]EquityPoint, 0)

	start = maxInt(24, start)
	end = minInt(end, len(candles)-2)
	rebalance := RebalanceHours(strategy)

	for i := start; i <= end; i++ {
		if candles[i].Close <= 0 || candles[i+1].Close <= 0 {
			continue
		}
		target := position
		if (i-start)%rebalance == 0 {
			target = StrategyTargetExposure(candles, i, strategy)
		}

		next := candles[i+1]
		buyLimit, sellLimit := StrategyLimitPrices(candles, i)

		if position == 0 && target != 0 {
			if target > 0 && next.Low <= buyLimit*(1-fillBufferPct) {
				position = target
				entryPrice = buyLimit
				entryTs = next.Timestamp
			} else if target < 0 && next.High >= sellLimit*(1+fillBufferPct) {
				position = target
				entryPrice = sellLimit
				entryTs = next.Timestamp
			}
		} else if position > 0 && target <= 0 && next.High >= sellLimit*(1+fillBufferPct) {
			net := ((sellLimit-entryPrice)/entryPrice*math.Abs(position)) - feeRate*2*math.Abs(position)
			equity *= maxFloat(0, 1+net)
			trades++
			if net > 0 {
				wins++
			}
			if detail {
				tradesOut = append(tradesOut, Trade{
					Pair: pair, Side: "long", EntryTimestamp: entryTs, ExitTimestamp: next.Timestamp,
					EntryPrice: entryPrice, ExitPrice: sellLimit, PnLPct: net * 100,
				})
			}
			position = 0
			entryPrice = 0
		} else if position < 0 && target >= 0 && next.Low <= buyLimit*(1-fillBufferPct) {
			net := ((entryPrice-buyLimit)/entryPrice*math.Abs(position)) - feeRate*2*math.Abs(position)
			equity *= maxFloat(0, 1+net)
			trades++
			if net > 0 {
				wins++
			}
			if detail {
				tradesOut = append(tradesOut, Trade{
					Pair: pair, Side: "short", EntryTimestamp: entryTs, ExitTimestamp: next.Timestamp,
					EntryPrice: entryPrice, ExitPrice: buyLimit, PnLPct: net * 100,
				})
			}
			position = 0
			entryPrice = 0
		}

		marked := equity
		if position > 0 && entryPrice > 0 {
			marked = equity * maxFloat(0, 1+((next.Close-entryPrice)/entryPrice*math.Abs(position))-feeRate*2*math.Abs(position))
		} else if position < 0 && entryPrice > 0 {
			marked = equity * maxFloat(0, 1+((entryPrice-next.Close)/entryPrice*math.Abs(position))-feeRate*2*math.Abs(position))
		}
		ret := 0.0
		if prevMarked > 0 {
			ret = marked/prevMarked - 1
		}
		returns = append(returns, ret)
		prevMarked = marked
		if marked > peak {
			peak = marked
		}
		if peak > 0 {
			maxDD = maxFloat(maxDD, (peak-marked)/peak*100)
		}
		if detail {
			curve = append(curve, EquityPoint{
				Timestamp: next.Timestamp,
				Value:     math.Round(marked*initialEquity*100) / 100,
			})
		}
	}

	if position != 0 && entryPrice > 0 {
		last := candles[minInt(len(candles)-1, end+1)]
		exitPrice := last.Close
		side := "long"
		var net float64
		if position > 0 {
			net = ((exitPrice-entryPrice)/entryPrice*math.Abs(position)) - feeRate*2*math.Abs(position)
		} else {
			net = ((entryPrice-exitPrice)/entryPrice*math.Abs(position)) - feeRate*2*math.Abs(position)
			side = "short"
		}
		equity *= maxFloat(0, 1+net)
		trades++
		if net > 0 {
			wins++
		}
		if detail {
			tradesOut = append(tradesOut, Trade{
				Pair: pair, Side: side, EntryTimestamp: entryTs, ExitTimestamp: last.Timestamp,
				EntryPrice: entryPrice, ExitPrice: exitPrice, PnLPct: net * 100,
			})
		}
	}

	return BacktestResult{
		Perf:        buildPerformance((equity-1)*100, trades, wins, maxDD, returns),
		Trades:      tradesOut,
		EquityCurve: curve,
	}
}

// SimulateLiveStrategyRange is the simple wrapper used by the walk-forward selector.
func SimulateLiveStrategyRange(candles []Candle, start, end int, strategy LiveStrategy) Performance {
	return SimulateLimitBacktest("", candles, start, end, strategy, 1, DefaultTradeFeeRate, DefaultFillBufferPct, false).Perf
}

// BuyAndHoldPerformance mirrors trading_bot.go:1289 buyAndHoldPerf, used as a
// fallback in bestLiveStrategyInRange when no strategy produced trades.
func BuyAndHoldPerformance(candles []Candle, start, end int, feeRate float64) (Performance, bool) {
	start = maxInt(0, start)
	end = minInt(end, len(candles)-1)
	if end-start < 2 {
		return Performance{}, false
	}
	entry := candles[start].Close
	if entry <= 0 {
		return Performance{}, false
	}
	returns := make([]float64, 0, end-start)
	peak, equity, maxDD := 1.0, 1.0, 0.0
	for i := start + 1; i <= end; i++ {
		c := candles[i].Close
		p := candles[i-1].Close
		if p <= 0 || c <= 0 {
			continue
		}
		ret := (c - p) / p
		returns = append(returns, ret)
		equity *= 1 + ret
		if equity > peak {
			peak = equity
		}
		if peak > 0 {
			maxDD = maxFloat(maxDD, (peak-equity)/peak*100)
		}
	}
	finalClose := candles[end].Close
	if finalClose <= 0 {
		return Performance{}, false
	}
	net := (finalClose-entry)/entry - feeRate*2
	wins := 0
	if net > 0 {
		wins = 1
	}
	return buildPerformance(net*100, 1, wins, maxDD, returns), true
}
