// Package mixed is a Go port of the bitbankgo "mixed" trading algorithm.
// See /nvme0n1-disk/code/bitbankgo/webapp/handlers/trading_bot.go for the
// canonical reference implementation. The port preserves the exact strategy
// definitions, threshold logic, walk-forward selector, and multi-pair mixer
// so that bit-for-bit parity is verifiable.
package mixed

const (
	DefaultTradeFeeRate  = 0.0006
	DefaultFillBufferPct = 0.0005
)

// ReturnTerm is one component of a strategy: weight * (close[idx]-close[idx-Lookback]) / close[idx-Lookback] / Lookback.
type ReturnTerm struct {
	Lookback int
	Weight   float64
}

// LiveStrategy mirrors bitbankgo's liveStrategy struct.
type LiveStrategy struct {
	Name         string
	Terms        []ReturnTerm
	ThresholdMul float64
	MinThreshold float64
	LongOnly     bool
	RebalanceHrs int
	Exposure     float64
}

// Performance mirrors bitbankgo's strategyPerformance.
type Performance struct {
	PnLPct         float64
	Trades         int
	Wins           int
	WinRate        float64
	MaxDrawdownPct float64
	Sortino        float64
}

// Trade mirrors bitbankgo's strategyTrade for diagnostics.
type Trade struct {
	Pair           string
	Side           string
	EntryTimestamp int64
	ExitTimestamp  int64
	EntryPrice     float64
	ExitPrice      float64
	PnLPct         float64
}

// Candle is the OHLC bar shape consumed by the algorithm. Timestamp is
// preserved as nanoseconds since epoch (or whatever unit the caller uses) so
// the decision side can pass it through trade records.
type Candle struct {
	Timestamp int64
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// BacktestResult bundles a Performance with optional trade detail / equity curve.
type BacktestResult struct {
	Perf        Performance
	Trades      []Trade
	EquityCurve []EquityPoint
}

type EquityPoint struct {
	Timestamp int64
	Value     float64
}

// liveStrategiesCache caches the strategy list for repeated lookups.
var liveStrategiesCache = buildLiveStrategies()

// LiveStrategies returns the cached, immutable strategy definitions.
func LiveStrategies() []LiveStrategy {
	return liveStrategiesCache
}

// buildLiveStrategies is the verbatim port of bitbankgo's buildLiveStrategies()
// at trading_bot.go:1112. Order is preserved so deterministic ties match.
func buildLiveStrategies() []LiveStrategy {
	return []LiveStrategy{
		{Name: "rev2h", Terms: []ReturnTerm{{Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.15, MinThreshold: 0.00035},
		{Name: "rev2h_size115", Terms: []ReturnTerm{{Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.15, MinThreshold: 0.00035, Exposure: 1.15},
		{Name: "rev2h_fast", Terms: []ReturnTerm{{Lookback: 2, Weight: -0.15}}, ThresholdMul: 0.10, MinThreshold: 0.00035},
		{Name: "rev1h_2h", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.20}, {Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.15, MinThreshold: 0.00035},
		{Name: "rev1h_2h_size115", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.20}, {Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.15, MinThreshold: 0.00035, Exposure: 1.15},
		{Name: "rev1h_2h_size125", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.20}, {Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.18, MinThreshold: 0.00045, Exposure: 1.25},
		{Name: "rev2h_24h", Terms: []ReturnTerm{{Lookback: 2, Weight: -0.20}, {Lookback: 24, Weight: -0.15}}, ThresholdMul: 0.15, MinThreshold: 0.00035},
		{Name: "rev2h_4h", Terms: []ReturnTerm{{Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.24, MinThreshold: 0.00055, RebalanceHrs: 4},
		{Name: "rev1h_2h_4h", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.20}, {Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.24, MinThreshold: 0.00055, RebalanceHrs: 4},
		{Name: "rev1h_2h_4h_size115", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.20}, {Lookback: 2, Weight: -0.20}}, ThresholdMul: 0.26, MinThreshold: 0.0006, RebalanceHrs: 4, Exposure: 1.15},
		{Name: "fast_revert_slow_mom", Terms: []ReturnTerm{{Lookback: 2, Weight: -0.20}, {Lookback: 24, Weight: 0.24}}, ThresholdMul: 0.35, MinThreshold: 0.0006, RebalanceHrs: 2},
		{Name: "downshock_rebound_long", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.30}, {Lookback: 3, Weight: -0.20}}, ThresholdMul: 0.30, MinThreshold: 0.0008, LongOnly: true, RebalanceHrs: 2},
		{Name: "rev4h_slow", Terms: []ReturnTerm{{Lookback: 4, Weight: -1.00}}, ThresholdMul: 0.90, MinThreshold: 0.00035, RebalanceHrs: 4},
		{Name: "rev4h_swing", Terms: []ReturnTerm{{Lookback: 4, Weight: -0.75}, {Lookback: 24, Weight: 0.16}}, ThresholdMul: 0.55, MinThreshold: 0.0008, RebalanceHrs: 6},
		{Name: "rev24h", Terms: []ReturnTerm{{Lookback: 24, Weight: -0.35}}, ThresholdMul: 0.15, MinThreshold: 0.00035, RebalanceHrs: 4},
		{Name: "mom6h", Terms: []ReturnTerm{{Lookback: 6, Weight: 1.00}}, ThresholdMul: 0.65, MinThreshold: 0.00035},
		{Name: "mom6h_size115", Terms: []ReturnTerm{{Lookback: 6, Weight: 1.00}}, ThresholdMul: 0.72, MinThreshold: 0.00055, Exposure: 1.15},
		{Name: "mom6h_clear", Terms: []ReturnTerm{{Lookback: 6, Weight: 1.00}}, ThresholdMul: 0.70, MinThreshold: 0.0025, RebalanceHrs: 3},
		{Name: "mom6h_clear_long", Terms: []ReturnTerm{{Lookback: 6, Weight: 1.00}}, ThresholdMul: 0.70, MinThreshold: 0.0025, LongOnly: true, RebalanceHrs: 3},
		{Name: "mom6h_clear_long_size115", Terms: []ReturnTerm{{Lookback: 6, Weight: 1.00}}, ThresholdMul: 0.80, MinThreshold: 0.0028, LongOnly: true, RebalanceHrs: 3, Exposure: 1.15},
		{Name: "mom12h_swing", Terms: []ReturnTerm{{Lookback: 12, Weight: 0.70}, {Lookback: 24, Weight: 0.25}}, ThresholdMul: 0.60, MinThreshold: 0.0011, RebalanceHrs: 6},
		{Name: "mom12h_lowturn", Terms: []ReturnTerm{{Lookback: 12, Weight: 0.75}, {Lookback: 36, Weight: 0.20}}, ThresholdMul: 0.85, MinThreshold: 0.0018, RebalanceHrs: 12},
		{Name: "mom12h_lowturn_long", Terms: []ReturnTerm{{Lookback: 12, Weight: 0.75}, {Lookback: 36, Weight: 0.20}}, ThresholdMul: 0.85, MinThreshold: 0.0018, LongOnly: true, RebalanceHrs: 12},
		{Name: "slow_mom_long", Terms: []ReturnTerm{{Lookback: 12, Weight: 0.60}, {Lookback: 24, Weight: 0.45}, {Lookback: 48, Weight: 0.30}}, ThresholdMul: 0.70, MinThreshold: 0.0012, LongOnly: true, RebalanceHrs: 6},
		{Name: "slow_mom_long_12h", Terms: []ReturnTerm{{Lookback: 12, Weight: 0.60}, {Lookback: 24, Weight: 0.45}, {Lookback: 48, Weight: 0.30}}, ThresholdMul: 0.80, MinThreshold: 0.0015, LongOnly: true, RebalanceHrs: 12},
		{Name: "slow_mom_daily_long", Terms: []ReturnTerm{{Lookback: 24, Weight: 0.60}, {Lookback: 72, Weight: 0.35}, {Lookback: 168, Weight: 0.18}}, ThresholdMul: 0.95, MinThreshold: 0.0016, LongOnly: true, RebalanceHrs: 24},
		{Name: "slow_mom_fast_revert", Terms: []ReturnTerm{{Lookback: 2, Weight: -0.12}, {Lookback: 12, Weight: 0.45}, {Lookback: 36, Weight: 0.30}}, ThresholdMul: 0.55, MinThreshold: 0.0009, RebalanceHrs: 4},
		{Name: "swing_revert_12h", Terms: []ReturnTerm{{Lookback: 12, Weight: -0.45}, {Lookback: 48, Weight: 0.16}}, ThresholdMul: 0.70, MinThreshold: 0.0012, RebalanceHrs: 12},
		{Name: "daily_revert", Terms: []ReturnTerm{{Lookback: 24, Weight: -0.38}, {Lookback: 72, Weight: 0.10}}, ThresholdMul: 0.80, MinThreshold: 0.0014, RebalanceHrs: 24},
		{Name: "asym_trend_long", Terms: []ReturnTerm{{Lookback: 6, Weight: 0.40}, {Lookback: 24, Weight: 0.55}}, ThresholdMul: 0.55, MinThreshold: 0.0010, LongOnly: true, RebalanceHrs: 4},
		{Name: "asym_trend_lowturn_long", Terms: []ReturnTerm{{Lookback: 12, Weight: 0.35}, {Lookback: 48, Weight: 0.55}}, ThresholdMul: 0.85, MinThreshold: 0.0016, LongOnly: true, RebalanceHrs: 12},
		{Name: "mom1h", Terms: []ReturnTerm{{Lookback: 1, Weight: 0.35}}, ThresholdMul: 0.65, MinThreshold: 0.00035},
		{Name: "rev_short_mom_day", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.15}, {Lookback: 3, Weight: -0.15}, {Lookback: 24, Weight: 0.15}}, ThresholdMul: 0.40, MinThreshold: 0.00035, RebalanceHrs: 2},
		{Name: "mixed_clear", Terms: []ReturnTerm{{Lookback: 1, Weight: -0.15}, {Lookback: 3, Weight: -0.10}, {Lookback: 24, Weight: 0.15}}, ThresholdMul: 0.40, MinThreshold: 0.00035, RebalanceHrs: 2},
		{Name: "mixed_swing", Terms: []ReturnTerm{{Lookback: 3, Weight: -0.12}, {Lookback: 12, Weight: 0.20}, {Lookback: 36, Weight: 0.18}}, ThresholdMul: 0.45, MinThreshold: 0.0009, RebalanceHrs: 6},
		{Name: "mixed_swing_12h", Terms: []ReturnTerm{{Lookback: 6, Weight: -0.10}, {Lookback: 24, Weight: 0.22}, {Lookback: 72, Weight: 0.12}}, ThresholdMul: 0.70, MinThreshold: 0.0014, RebalanceHrs: 12},
	}
}

// FindStrategy returns the strategy with the given name, or zero-value if missing.
func FindStrategy(name string) (LiveStrategy, bool) {
	for _, s := range liveStrategiesCache {
		if s.Name == name {
			return s, true
		}
	}
	return LiveStrategy{}, false
}
