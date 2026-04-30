package mixed

import "math"

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxFloat(values ...float64) float64 {
	m := values[0]
	for _, v := range values[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func minFloat(values ...float64) float64 {
	m := values[0]
	for _, v := range values[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func meanFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var s float64
	for _, v := range values {
		s += v
	}
	return s / float64(len(values))
}

func avgFloat(values []float64) float64 { return meanFloat(values) }

func stddev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	var sq float64
	for _, v := range values {
		d := v - mean
		sq += d * d
	}
	return math.Sqrt(sq / float64(len(values)))
}

// hourlyReturns mirrors trading_bot.go:1031.
func hourlyReturns(candles []Candle) []float64 {
	out := make([]float64, 0, len(candles)-1)
	for i := 1; i < len(candles); i++ {
		prev := candles[i-1].Close
		if prev > 0 && candles[i].Close > 0 {
			out = append(out, (candles[i].Close-prev)/prev)
		}
	}
	return out
}

// sortinoRatio mirrors trading_bot.go:1604 — including the magic clamp at ±20
// and the "all-positive" sentinel of 20 when downsideCount==0 and mean>0.
func sortinoRatio(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	mean := avgFloat(returns)
	downSq := 0.0
	downCount := 0
	for _, r := range returns {
		if r < 0 {
			downSq += r * r
			downCount++
		}
	}
	if downCount == 0 {
		if mean > 0 {
			return 20
		}
		return 0
	}
	dd := math.Sqrt(downSq / float64(downCount))
	if dd <= 0 {
		return 0
	}
	return clamp(mean/dd*math.Sqrt(float64(len(returns))), -20, 20)
}

func buildPerformance(pnlPct float64, trades, wins int, maxDDPct float64, returns []float64) Performance {
	winRate := 0.0
	if trades > 0 {
		winRate = float64(wins) / float64(trades) * 100
	}
	return Performance{
		PnLPct:         pnlPct,
		Trades:         trades,
		Wins:           wins,
		WinRate:        winRate,
		MaxDrawdownPct: maxDDPct,
		Sortino:        sortinoRatio(returns),
	}
}
