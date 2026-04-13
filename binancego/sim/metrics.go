package sim

import "math"

const HourlyPeriodsPerYear = 8760.0

// Sortino computes annualized Sortino ratio matching src/metrics_utils.py.
// downside_dev = sqrt(sum_neg_sq / total_n) -- RMS of negatives / total count.
func Sortino(equityCurve []float64, periodsPerYear float64) float64 {
	n := len(equityCurve)
	if n < 3 {
		return 0
	}
	returns := StepReturns(equityCurve)
	if len(returns) < 2 {
		return 0
	}
	var sumRet, sumNegSq float64
	hasNeg := false
	for _, r := range returns {
		sumRet += r
		if r < 0 {
			sumNegSq += r * r
			hasNeg = true
		}
	}
	if !hasNeg {
		// All positive: fall back to Sharpe
		return Sharpe(equityCurve, periodsPerYear)
	}
	meanRet := sumRet / float64(len(returns))
	// canonical: sqrt(sum_neg_sq / total_n)
	downsideDev := math.Sqrt(sumNegSq / float64(len(returns)))
	if downsideDev <= 0 {
		return 0
	}
	return (meanRet / downsideDev) * math.Sqrt(periodsPerYear)
}

// Sharpe computes annualized Sharpe ratio.
func Sharpe(equityCurve []float64, periodsPerYear float64) float64 {
	returns := StepReturns(equityCurve)
	if len(returns) < 2 {
		return 0
	}
	var sum, sumSq float64
	for _, r := range returns {
		sum += r
		sumSq += r * r
	}
	n := float64(len(returns))
	mean := sum / n
	variance := sumSq/n - mean*mean
	if variance <= 0 {
		return 0
	}
	// ddof=1 for Sharpe (matching metrics_utils.py)
	stdDev := math.Sqrt(sumSq/(n-1) - (sum*sum)/(n*(n-1)))
	if stdDev <= 0 {
		return 0
	}
	return (mean * periodsPerYear) / (stdDev * math.Sqrt(periodsPerYear))
}

// MaxDrawdown returns the maximum peak-to-trough drawdown (negative value).
func MaxDrawdown(equityCurve []float64) float64 {
	if len(equityCurve) == 0 {
		return 0
	}
	peak := equityCurve[0]
	maxDD := 0.0
	for _, v := range equityCurve {
		if v > peak {
			peak = v
		}
		dd := (v - peak) / (peak + 1e-10)
		if dd < maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

// StepReturns converts equity curve to percentage returns.
func StepReturns(equityCurve []float64) []float64 {
	n := len(equityCurve)
	if n < 2 {
		return nil
	}
	ret := make([]float64, n-1)
	for i := 0; i < n-1; i++ {
		denom := equityCurve[i]
		if denom == 0 {
			denom = 1e-10
		}
		r := (equityCurve[i+1] - equityCurve[i]) / denom
		if math.IsInf(r, 0) || math.IsNaN(r) {
			r = 0
		}
		ret[i] = r
	}
	return ret
}
