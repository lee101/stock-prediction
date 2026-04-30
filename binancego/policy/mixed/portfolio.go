package mixed

import (
	"math"
	"sort"
)

// PairSignal is the per-symbol decision summary the portfolio layer reads.
// It mirrors the heterogeneous map[string]interface{} payload that bitbankgo
// passes to mixedSignal() but as a typed struct.
type PairSignal struct {
	Pair           string
	SignalType     string  // "buy", "sell", "hold"
	BuyPrice       float64 // limit price (close * (1 - spread/2))
	SellPrice      float64 // limit price (close * (1 + spread/2))
	PnL7DPct       float64 // backward-window backtest pnl percent
	MaxDD7DPct     float64 // backward-window max drawdown
	Sortino7D      float64 // backward-window sortino
	Trades7D       int     // backward-window trade count
	Confidence     float64 // 0..1 from selector (use Sortino-derived if absent)
	RebalanceHours int
	XGBPnL30DPct   float64 // optional XGB bonus inputs (zero if unused)
	XGBMaxDD30DPct float64
	XGBSortino30D  float64
	XGBTrades30D   int
}

// MixedCandidate is bitbankgo's mixedCandidate.
type MixedCandidate struct {
	Pair       string
	Signal     PairSignal
	Score      float64
	PnL        float64
	DD         float64
	Sortino    float64
	Confidence float64
	Spread     float64 // (sell-buy)/mid
	Trades     int
	Rebalance  int
}

// MixedDecision is the aggregated buy/sell/hold call across selected pairs.
type MixedDecision struct {
	SignalType     string  // "buy" | "sell" | "hold"
	BuyPrice       float64 // 100*(1-spread/2)
	SellPrice      float64 // 100*(1+spread/2)
	Confidence     float64
	PnL7DPct       float64
	Trades7D       int
	MaxDD7DPct     float64
	Sortino7D      float64
	RebalanceHours float64
	Constituents   []string
	Actions        []PairSignal
}

// MixedCandidateScore mirrors trading_bot.go:774.
func MixedCandidateScore(pnl, dd, sortino float64, trades int) float64 {
	if trades < 0 {
		trades = 0
	}
	return pnl*1.12 - dd*0.65 + sortino*2.05 - math.Sqrt(float64(trades))*0.06
}

// XGBCandidateBonus mirrors trading_bot.go:781.
func XGBCandidateBonus(s PairSignal) float64 {
	if s.XGBTrades30D < 8 || s.XGBSortino30D <= 0 || s.XGBPnL30DPct <= 0 {
		return 0
	}
	return clamp(s.XGBPnL30DPct*0.16-s.XGBMaxDD30DPct*0.08+s.XGBSortino30D*0.85-math.Sqrt(float64(s.XGBTrades30D))*0.015, 0, 12)
}

// BestMixedCandidates mirrors trading_bot.go:926. Picks size in [3,6] that
// maximises (mean pnl − maxDD*1.05 + mean sortino*2.65 − sqrt(size)*0.18).
// If the best score is non-positive, falls back to the top-3 (or fewer).
func BestMixedCandidates(candidates []MixedCandidate) []MixedCandidate {
	if len(candidates) == 0 {
		return nil
	}
	bestScore := math.Inf(-1)
	bestSize := minInt(3, len(candidates))
	for size := 3; size <= minInt(6, len(candidates)); size++ {
		var pnlSum, sortinoSum, ddMax float64
		for _, c := range candidates[:size] {
			pnlSum += c.PnL
			sortinoSum += c.Sortino
			ddMax = maxFloat(ddMax, c.DD)
		}
		div := float64(size)
		score := pnlSum/div - ddMax*1.05 + sortinoSum/div*2.65 - math.Sqrt(div)*0.18
		if score > bestScore {
			bestScore = score
			bestSize = size
		}
	}
	if bestScore <= 0 {
		return candidates[:minInt(3, len(candidates))]
	}
	return candidates[:bestSize]
}

// MixSignal aggregates per-pair signals into a portfolio-level decision.
// Mirrors trading_bot.go:666 mixedSignal but with typed inputs.
func MixSignal(signals []PairSignal) *MixedDecision {
	candidates := make([]MixedCandidate, 0, len(signals))
	for _, sig := range signals {
		buy, sell := sig.BuyPrice, sig.SellPrice
		if buy <= 0 || sell <= 0 {
			continue
		}
		mid := (buy + sell) / 2
		if mid <= 0 {
			continue
		}
		score := MixedCandidateScore(sig.PnL7DPct, sig.MaxDD7DPct, sig.Sortino7D, sig.Trades7D) + XGBCandidateBonus(sig)
		candidates = append(candidates, MixedCandidate{
			Pair:       sig.Pair,
			Signal:     sig,
			Score:      score,
			PnL:        sig.PnL7DPct,
			DD:         sig.MaxDD7DPct,
			Sortino:    sig.Sortino7D,
			Confidence: sig.Confidence,
			Spread:     (sell - buy) / mid,
			Trades:     sig.Trades7D,
			Rebalance:  sig.RebalanceHours,
		})
	}
	if len(candidates) == 0 {
		return nil
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].Score == candidates[j].Score {
			return candidates[i].Pair < candidates[j].Pair
		}
		return candidates[i].Score > candidates[j].Score
	})
	selected := BestMixedCandidates(candidates)

	buys, sells := 0, 0
	var spreadSum, pnlSum, ddMax, sortinoSum, confSum float64
	trades := 0
	rebalanceSum := 0
	constituents := make([]string, 0, len(selected))
	actions := make([]PairSignal, 0, len(selected))
	for _, c := range selected {
		constituents = append(constituents, c.Pair)
		trades += c.Trades
		rebalanceSum += maxInt(1, c.Rebalance)
		spreadSum += c.Spread
		pnlSum += c.PnL
		ddMax = maxFloat(ddMax, c.DD)
		sortinoSum += c.Sortino
		confSum += c.Confidence
		actions = append(actions, c.Signal)
		switch c.Signal.SignalType {
		case "buy":
			buys++
		case "sell":
			sells++
		}
	}
	count := len(selected)
	signal := "hold"
	if buys > sells && buys >= 2 {
		signal = "buy"
	} else if sells > buys && sells >= 2 {
		signal = "sell"
	}
	div := float64(count)
	spread := clamp(spreadSum/div, 0.0018, 0.04)
	return &MixedDecision{
		SignalType:     signal,
		BuyPrice:       100 * (1 - spread/2),
		SellPrice:      100 * (1 + spread/2),
		Confidence:     confSum / div,
		PnL7DPct:       pnlSum / div,
		Trades7D:       trades,
		MaxDD7DPct:     ddMax,
		Sortino7D:      sortinoSum / div,
		RebalanceHours: math.Round(float64(rebalanceSum)/div*100) / 100,
		Constituents:   constituents,
		Actions:        actions,
	}
}
