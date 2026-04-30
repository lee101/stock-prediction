package mixed

// Signal-side functions: turn a strategy + candle history into a target
// position (-1, 0, +1) × exposure and the limit prices used for execution.
// Direct port of trading_bot.go:1167-1652.

func RebalanceHours(s LiveStrategy) int {
	if s.RebalanceHrs <= 0 {
		return 1
	}
	return s.RebalanceHrs
}

func StrategyExposure(s LiveStrategy) float64 {
	if s.Exposure <= 0 {
		return 1
	}
	return clamp(s.Exposure, 0.25, 1.25)
}

// StrategyThreshold returns max(MinThreshold, vol*ThresholdMul) with the
// canonical defaults filled in (matches trading_bot.go:1167).
func StrategyThreshold(s LiveStrategy, vol float64) float64 {
	minT := s.MinThreshold
	if minT <= 0 {
		minT = 0.00035
	}
	mul := s.ThresholdMul
	if mul <= 0 {
		mul = 0.20
	}
	return maxFloat(minT, vol*mul)
}

// StrategyExpectedReturn evaluates the strategy at index idx of candles.
// Mirrors trading_bot.go:1179.
func StrategyExpectedReturn(candles []Candle, idx int, s LiveStrategy) float64 {
	if idx <= 1 || idx >= len(candles) {
		return 0
	}
	expected := 0.0
	for _, term := range s.Terms {
		if term.Lookback <= 0 || idx < term.Lookback {
			continue
		}
		base := candles[idx-term.Lookback].Close
		if base <= 0 || candles[idx].Close <= 0 {
			continue
		}
		expected += term.Weight * ((candles[idx].Close - base) / base / float64(term.Lookback))
	}
	return clamp(expected, -0.02, 0.02)
}

// StrategySide returns -1/0/+1 (mirrors trading_bot.go:1630).
func StrategySide(candles []Candle, idx int, s LiveStrategy) float64 {
	if idx <= 1 || idx >= len(candles) {
		return 0
	}
	expected := StrategyExpectedReturn(candles, idx, s)
	rets := hourlyReturns(candles[maxInt(0, idx-72) : idx+1])
	threshold := StrategyThreshold(s, stddev(rets))
	if expected > threshold {
		return 1
	}
	if expected < -threshold && !s.LongOnly {
		return -1
	}
	return 0
}

// StrategyTargetExposure: side × exposure (mirrors trading_bot.go:1646).
func StrategyTargetExposure(candles []Candle, idx int, s LiveStrategy) float64 {
	side := StrategySide(candles, idx, s)
	if side == 0 {
		return 0
	}
	return side * StrategyExposure(s)
}

// StrategyOrderSpread mirrors trading_bot.go:1573 — clamped vol-based spread.
func StrategyOrderSpread(candles []Candle, idx int) float64 {
	if idx <= 1 {
		return 0.0018
	}
	start := maxInt(0, idx-72)
	rets := hourlyReturns(candles[start : idx+1])
	return clamp(maxFloat(0.0018, stddev(rets)*3.0), 0.0018, 0.035)
}

// StrategyLimitPrices returns (buyLimit, sellLimit) from close ± spread/2.
func StrategyLimitPrices(candles []Candle, idx int) (float64, float64) {
	close := candles[idx].Close
	spread := StrategyOrderSpread(candles, idx)
	return close * (1 - spread/2), close * (1 + spread/2)
}
