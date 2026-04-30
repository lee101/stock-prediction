package mixed

import (
	"math"
	"sort"
)

// PortfolioConfig controls the multi-symbol walk-forward backtest.
type PortfolioConfig struct {
	// LookbackHours used for per-symbol backward-window stats fed into the
	// MixSignal candidate scoring (mirrors bitbankgo's 7-day window when
	// LookbackHours = 168).
	LookbackHours int

	// TrainHours used by BestWalkForwardStrategy (mirrors bitbankgo's 168).
	TrainHours int

	// FeeRate applied per side of every fill (default 0.001 = 10bps stocks).
	FeeRate float64

	// FillBufferPct extra padding past the limit price required for a fill.
	FillBufferPct float64

	// MaxLeverage caps total |allocation| across selected pairs (e.g. 2.0 for Alpaca).
	MaxLeverage float64

	// CanShort turns shorts on/off. For Alpaca cash accounts set false.
	CanShort bool

	// RebalanceEveryBars how often (in bars) to re-pick the portfolio. 1 = every bar.
	RebalanceEveryBars int
}

func DefaultPortfolioConfig() PortfolioConfig {
	return PortfolioConfig{
		LookbackHours:      168,
		TrainHours:         168,
		FeeRate:            0.001,
		FillBufferPct:      DefaultFillBufferPct,
		MaxLeverage:        2.0,
		CanShort:           true,
		RebalanceEveryBars: 1,
	}
}

// PortfolioResult is the equity curve + summary stats for the full window.
type PortfolioResult struct {
	EquityCurve  []float64
	Returns      []float64
	Timestamps   []int64
	TotalReturn  float64
	MonthlyMed   float64 // median of 30-bar*24 (calendar-month proxy) chunks
	Sortino      float64
	MaxDrawdown  float64
	NumTrades    int
	NumDecisions int
}

// SymbolBars is one symbol's full hourly bar series. Bars must be aligned to
// the same UTC index across symbols (zero-fill missing bars in the loader).
type SymbolBars struct {
	Symbol string
	Bars   []Candle
}

type heldEntry struct {
	symIdx int
	side   float64 // -1, 0, 1
	weight float64 // |allocation| in [0, 1+]
}

// SimulatePortfolio runs the bitbankgo-style multi-symbol walk-forward
// backtest. Each bar in [startBar, endBar):
//  1. For each symbol, compute a fresh PairSignal using
//     BestWalkForwardStrategy on the trailing TrainHours window, then
//     evaluate that strategy on the trailing LookbackHours window for
//     pnl/sortino/dd stats.
//  2. Feed the per-symbol signals into MixSignal to pick which pairs to
//     hold for this bar (BestMixedCandidates).
//  3. Allocate equity equally across the selected pairs (capped at
//     MaxLeverage / count).
//  4. Apply per-symbol next-bar return (with side from each strategy's
//     target exposure), deducting fees on turnover.
//
// This is intentionally a coarse aggregator — the per-symbol fine-grained
// limit-fill simulator lives in SimulateLimitBacktest. For per-100d-window
// portfolio gating against the 27%/mo HARD RULE we use this function.
func SimulatePortfolio(symbols []SymbolBars, startBar, endBar int, cfg PortfolioConfig) PortfolioResult {
	if cfg.LookbackHours <= 0 {
		cfg.LookbackHours = 168
	}
	if cfg.TrainHours <= 0 {
		cfg.TrainHours = 168
	}
	if cfg.MaxLeverage <= 0 {
		cfg.MaxLeverage = 1
	}
	if cfg.RebalanceEveryBars <= 0 {
		cfg.RebalanceEveryBars = 1
	}

	if len(symbols) == 0 {
		return PortfolioResult{}
	}
	// Align to the longest series; assume all are same length (loader's job).
	totalBars := 0
	for _, s := range symbols {
		if len(s.Bars) > totalBars {
			totalBars = len(s.Bars)
		}
	}
	if endBar > totalBars-1 {
		endBar = totalBars - 1
	}
	startBar = maxInt(startBar, 96)
	if endBar <= startBar {
		return PortfolioResult{}
	}

	equity := 1.0
	peak := equity
	maxDD := 0.0
	curve := []float64{equity}
	rets := make([]float64, 0, endBar-startBar)
	timestamps := make([]int64, 0, endBar-startBar)

	var held []heldEntry
	totalTrades := 0
	decisions := 0

	for i := startBar; i < endBar; i++ {
		// (Re)pick portfolio at rebalance ticks.
		if (i-startBar)%cfg.RebalanceEveryBars == 0 {
			signals := make([]PairSignal, 0, len(symbols))
			sides := make(map[string]float64, len(symbols))
			for sIdx, sym := range symbols {
				if i >= len(sym.Bars) {
					continue
				}
				strat, _ := BestWalkForwardStrategy(sym.Bars, i, cfg.TrainHours)
				if strat.Name == "" {
					continue
				}
				// Trailing-lookback stats for portfolio scoring.
				lbStart := maxInt(24, i-cfg.LookbackHours)
				perf := SimulateLimitBacktest("", sym.Bars, lbStart, i, strat, 1, cfg.FeeRate, cfg.FillBufferPct, false).Perf
				if perf.Trades == 0 {
					continue
				}
				target := StrategyTargetExposure(sym.Bars, i, strat)
				if target == 0 && !cfg.CanShort {
					// allow zero — just hold cash for that pair.
				}
				if !cfg.CanShort && target < 0 {
					target = 0
				}
				sigType := "hold"
				if target > 0 {
					sigType = "buy"
				} else if target < 0 {
					sigType = "sell"
				}
				buyLim, sellLim := StrategyLimitPrices(sym.Bars, i)
				signals = append(signals, PairSignal{
					Pair:           sym.Symbol,
					SignalType:     sigType,
					BuyPrice:       buyLim,
					SellPrice:      sellLim,
					PnL7DPct:       perf.PnLPct,
					MaxDD7DPct:     perf.MaxDrawdownPct,
					Sortino7D:      perf.Sortino,
					Trades7D:       perf.Trades,
					Confidence:     clamp(perf.Sortino/4.0, 0, 1),
					RebalanceHours: RebalanceHours(strat),
				})
				sides[sym.Symbol] = target
				_ = sIdx
			}
			decisions++
			decision := MixSignal(signals)
			newHeld := []heldEntry{}
			if decision != nil && decision.SignalType != "hold" {
				dirFilter := 1.0
				if decision.SignalType == "sell" {
					dirFilter = -1.0
				}
				active := []PairSignal{}
				for _, a := range decision.Actions {
					side := sides[a.Pair]
					if dirFilter > 0 && side > 0 {
						active = append(active, a)
					} else if dirFilter < 0 && side < 0 {
						active = append(active, a)
					}
				}
				if len(active) > 0 {
					perPairWeight := cfg.MaxLeverage / float64(len(active))
					for _, a := range active {
						symIdx := indexBySymbol(symbols, a.Pair)
						if symIdx < 0 {
							continue
						}
						newHeld = append(newHeld, heldEntry{
							symIdx: symIdx,
							side:   sides[a.Pair],
							weight: perPairWeight * math.Abs(sides[a.Pair]),
						})
					}
				}
			}
			// Charge fee on turnover (sum of |new − old| weights).
			turnover := computeTurnover(held, newHeld)
			equity *= maxFloat(0, 1-cfg.FeeRate*turnover)
			totalTrades += turnoverTrades(held, newHeld)
			held = newHeld
		}

		// Apply this bar's MTM ret on whatever is currently held.
		barRet := 0.0
		var ts int64
		for _, h := range held {
			sym := symbols[h.symIdx]
			if i+1 >= len(sym.Bars) {
				continue
			}
			c0 := sym.Bars[i].Close
			c1 := sym.Bars[i+1].Close
			ts = sym.Bars[i+1].Timestamp
			if c0 <= 0 || c1 <= 0 {
				continue
			}
			r := (c1 - c0) / c0
			barRet += h.side * h.weight * r
		}
		equity *= maxFloat(0, 1+barRet)
		rets = append(rets, barRet)
		timestamps = append(timestamps, ts)
		curve = append(curve, equity)
		if equity > peak {
			peak = equity
		}
		if peak > 0 {
			ddNow := (peak - equity) / peak * 100
			if ddNow > maxDD {
				maxDD = ddNow
			}
		}
	}

	return PortfolioResult{
		EquityCurve:  curve,
		Returns:      rets,
		Timestamps:   timestamps,
		TotalReturn:  equity - 1,
		MonthlyMed:   monthlyMedianFromHourlyReturns(rets),
		Sortino:      sortinoRatio(rets),
		MaxDrawdown:  maxDD,
		NumTrades:    totalTrades,
		NumDecisions: decisions,
	}
}

func indexBySymbol(syms []SymbolBars, name string) int {
	for i, s := range syms {
		if s.Symbol == name {
			return i
		}
	}
	return -1
}

func computeTurnover(prev, next []heldEntry) float64 {
	prevW := map[int]float64{}
	for _, h := range prev {
		prevW[h.symIdx] += h.side * h.weight
	}
	nextW := map[int]float64{}
	for _, h := range next {
		nextW[h.symIdx] += h.side * h.weight
	}
	turnover := 0.0
	seen := map[int]bool{}
	for k, v := range prevW {
		seen[k] = true
		turnover += math.Abs(nextW[k] - v)
	}
	for k, v := range nextW {
		if !seen[k] {
			turnover += math.Abs(v)
		}
	}
	return turnover
}

func turnoverTrades(prev, next []heldEntry) int {
	prevSet := map[int]bool{}
	for _, h := range prev {
		prevSet[h.symIdx] = true
	}
	nextSet := map[int]bool{}
	for _, h := range next {
		nextSet[h.symIdx] = true
	}
	count := 0
	for k := range prevSet {
		if !nextSet[k] {
			count++
		}
	}
	for k := range nextSet {
		if !prevSet[k] {
			count++
		}
	}
	return count
}

// monthlyMedianFromHourlyReturns groups returns into 30*24-bar (≈1mo)
// buckets and returns the median compounded return per bucket. If we don't
// have enough bars for a full month, falls back to compounding the lot.
func monthlyMedianFromHourlyReturns(rets []float64) float64 {
	const barsPerMonth = 30 * 24
	if len(rets) < barsPerMonth {
		eq := 1.0
		for _, r := range rets {
			eq *= 1 + r
		}
		if len(rets) == 0 {
			return 0
		}
		// scale to 30d
		scale := float64(barsPerMonth) / float64(len(rets))
		return math.Pow(eq, scale) - 1
	}
	monthlyRets := make([]float64, 0, len(rets)/barsPerMonth)
	for i := 0; i+barsPerMonth <= len(rets); i += barsPerMonth {
		eq := 1.0
		for _, r := range rets[i : i+barsPerMonth] {
			eq *= 1 + r
		}
		monthlyRets = append(monthlyRets, eq-1)
	}
	if len(monthlyRets) == 0 {
		return 0
	}
	sort.Float64s(monthlyRets)
	n := len(monthlyRets)
	if n%2 == 1 {
		return monthlyRets[n/2]
	}
	return (monthlyRets[n/2-1] + monthlyRets[n/2]) / 2
}
