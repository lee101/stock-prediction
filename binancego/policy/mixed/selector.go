package mixed

import "math"

// StrategyScore mirrors trading_bot.go:1331.
func StrategyScore(p Performance) float64 {
	if p.Trades == 0 {
		return math.Inf(-1)
	}
	tradePenalty := math.Sqrt(float64(p.Trades)) * 0.13
	winBonus := (p.WinRate - 50.0) * 0.025
	return p.PnLPct - p.MaxDrawdownPct*1.05 + p.Sortino*2.65 + winBonus - tradePenalty
}

// BestLiveStrategyInRange mirrors trading_bot.go:1242.
func BestLiveStrategyInRange(candles []Candle, start, end int, gated bool) (LiveStrategy, Performance) {
	bestScore := math.Inf(-1)
	var best LiveStrategy
	var bestPerf Performance
	for _, s := range LiveStrategies() {
		perf := SimulateLiveStrategyRange(candles, start, end, s)
		if perf.Trades == 0 {
			continue
		}
		score := StrategyScore(perf)
		if gated {
			rebalance := RebalanceHours(s)
			minTrades := maxInt(3, minInt(8, (end-start)/maxInt(1, rebalance*5)))
			maxDD := 6.0
			if rebalance >= 4 {
				maxDD = 7.5
			}
			if perf.Trades < minTrades ||
				perf.PnLPct < 0.35 ||
				perf.WinRate < 45.0 ||
				perf.MaxDrawdownPct > maxDD ||
				perf.Sortino < 0.35 {
				continue
			}
		}
		if score > bestScore {
			bestScore = score
			best = s
			bestPerf = perf
		}
	}
	if best.Name == "" && !gated {
		if bh, ok := BuyAndHoldPerformance(candles, start, end, DefaultTradeFeeRate); ok {
			best = LiveStrategy{
				Name:         "buy_and_hold",
				LongOnly:     true,
				RebalanceHrs: maxInt(1, end-start),
				Exposure:     1,
			}
			bestPerf = bh
		}
	}
	return best, bestPerf
}

// BestWalkForwardStrategy mirrors trading_bot.go:1230. Picks the strategy with
// the highest StrategyScore on the trailing trainHours window prior to idx,
// using the gated filter (min trades, win rate, drawdown, sortino).
func BestWalkForwardStrategy(candles []Candle, idx, trainHours int) (LiveStrategy, Performance) {
	if idx < 96 || idx >= len(candles) {
		return LiveStrategy{}, Performance{}
	}
	if trainHours <= 0 {
		trainHours = 168
	}
	start := maxInt(24, idx-trainHours)
	return BestLiveStrategyInRange(candles, start, idx, true)
}

// BestLiveStrategy mirrors trading_bot.go:1205 — picks unconstrained best on
// the trailing lookback ending at len(candles)-1.
func BestLiveStrategy(candles []Candle, lookbackHours int) (LiveStrategy, Performance) {
	if len(candles) < 32 {
		return LiveStrategy{}, Performance{}
	}
	if lookbackHours <= 0 {
		lookbackHours = 168
	}
	end := len(candles) - 1
	start := maxInt(24, end-lookbackHours)
	return BestLiveStrategyInRange(candles, start, end, false)
}

// SimulateWalkForwardPerformance mirrors trading_bot.go:1345 — runs the
// walk-forward selector across a trailing lookback window with bar-level
// turnover at 0.0006 per side. Used to evaluate end-to-end strategy quality.
func SimulateWalkForwardPerformance(candles []Candle, lookbackHours, trainHours int) Performance {
	if len(candles) < 120 {
		return Performance{}
	}
	if lookbackHours <= 0 {
		lookbackHours = 168
	}
	if trainHours <= 0 {
		trainHours = 168
	}
	equity, peak := 1.0, 1.0
	returns := make([]float64, 0, lookbackHours)
	trades, wins := 0, 0
	maxDD := 0.0
	position := 0.0
	tradeEquity := 1.0
	start := maxInt(96, len(candles)-lookbackHours)
	for i := start; i+1 < len(candles); i++ {
		strategy, _ := BestWalkForwardStrategy(candles, i, trainHours)
		if candles[i].Close <= 0 || candles[i+1].Close <= 0 {
			continue
		}
		cost := 0.0
		if strategy.Name == "" {
			if position != 0 {
				trades++
				if tradeEquity > 1 {
					wins++
				}
				cost = 0.0006 * math.Abs(position)
				position = 0
				tradeEquity = 1
			}
		} else {
			rebalance := RebalanceHours(strategy)
			if (i-start)%rebalance == 0 {
				target := StrategyTargetExposure(candles, i, strategy)
				if target != position {
					if position != 0 {
						trades++
						if tradeEquity > 1 {
							wins++
						}
						tradeEquity = 1
					}
					turnover := math.Abs(target - position)
					if turnover > 0 {
						cost = 0.0006 * turnover
					}
					position = target
				}
			}
		}
		rawRet := position * (candles[i+1].Close - candles[i].Close) / candles[i].Close
		ret := rawRet - cost
		if position != 0 {
			tradeEquity *= 1 + rawRet
		}
		returns = append(returns, ret)
		equity *= 1 + ret
		if equity > peak {
			peak = equity
		}
		if peak > 0 {
			maxDD = maxFloat(maxDD, (peak-equity)/peak*100)
		}
	}
	if position != 0 {
		trades++
		if tradeEquity > 1 {
			wins++
		}
	}
	return buildPerformance((equity-1)*100, trades, wins, maxDD, returns)
}
